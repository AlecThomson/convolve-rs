use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use convolve_rs::{
    beam::Beam,
    common_beam::{common_beam, fits_in_beam},
    convolve_uv::FftPlans,
    cube_io::{self, CubeElem, CubeMeta, CubeMode},
    fits_io::{output_path, read_fits, write_fits},
    smooth::{smooth, smooth_with_plans},
};

// ── Top-level CLI ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "convolvers",
    about = "Convolve FITS images/cubes to a common beam",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Smooth 2D FITS images to a common beam resolution.
    #[command(name = "2d")]
    TwoD(TwoDArgs),
    /// Smooth 3D/4D FITS spectral cubes to a common beam.
    #[command(name = "3d")]
    ThreeD(ThreeDArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::TwoD(args) => cmd_2d(args),
        Commands::ThreeD(args) => cmd_3d(args),
    }
}

// ── Shared args ────────────────────────────────────────────────────────────────

#[derive(Args, Debug, Clone)]
struct SharedArgs {
    /// Output filename suffix.
    #[arg(short, long, default_value = "sm")]
    suffix: String,

    /// Output filename prefix.
    #[arg(short, long)]
    prefix: Option<String>,

    /// Output directory [default: same as input].
    #[arg(short, long)]
    outdir: Option<PathBuf>,

    /// Target BMAJ in arcsec (must also specify --bmin and --bpa).
    #[arg(long)]
    bmaj: Option<f64>,

    /// Target BMIN in arcsec.
    #[arg(long)]
    bmin: Option<f64>,

    /// Target BPA in degrees.
    #[arg(long)]
    bpa: Option<f64>,

    /// Circularise the final beam (BMIN = BMAJ, BPA = 0).
    #[arg(long)]
    circularise: bool,

    /// Beam size cutoff in arcsec — blank images/channels with BMAJ larger than this.
    #[arg(short, long)]
    cutoff: Option<f64>,

    /// Compute common beam and report without writing output files.
    #[arg(short, long)]
    dryrun: bool,

    /// Tolerance for MVE common-beam algorithm.
    #[arg(long, default_value_t = 1e-4)]
    tolerance: f64,

    /// Number of ellipse edge samples per beam for MVE.
    #[arg(long, default_value_t = 200)]
    nsamps: usize,

    /// Epsilon (edge inflation) for MVE.
    #[arg(long, default_value_t = 5e-4)]
    epsilon: f64,

    /// Verbose output (-v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

// ── 2D subcommand ─────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
struct TwoDArgs {
    /// Input FITS image(s).
    #[arg(required = true, num_args = 1..)]
    infile: Vec<PathBuf>,

    /// Treat a single infile as a text file listing one path per line.
    #[arg(long)]
    listfile: bool,

    #[command(flatten)]
    shared: SharedArgs,

    /// Path to write a beamlog.
    #[arg(long)]
    log: Option<PathBuf>,
}

struct BeamLogEntry2D {
    filename: PathBuf,
    old_beam: Beam,
    new_beam: Beam,
    conv_beam: Beam,
}

fn cmd_2d(args: TwoDArgs) -> Result<()> {
    init_logging(args.shared.verbose);

    let files = collect_files(&args.infile, args.listfile)?;
    let target_beam = parse_target_beam(&args.shared)?;

    let sp = spinner(format!(
        "Reading beam parameters from {} file(s)…",
        files.len()
    ));
    let all_beams: Vec<Beam> = files
        .iter()
        .map(|f| {
            let data = read_fits(f).with_context(|| format!("reading {}", f.display()))?;
            if let Some(cutoff) = args.shared.cutoff
                && data.beam.major_arcsec() > cutoff
            {
                sp.suspend(|| {
                    warn!(
                        "{}: BMAJ={:.1}\" > cutoff={:.1}\" — will be blanked",
                        f.display(),
                        data.beam.major_arcsec(),
                        cutoff
                    )
                });
            }
            Ok(data.beam)
        })
        .collect::<Result<Vec<_>>>()?;
    sp.finish_and_clear();

    let mut common = match target_beam {
        Some(b) => {
            if !fits_in_beam(&all_beams, &b) {
                bail!("target beam is too small — some images cannot reach it");
            }
            b
        }
        None => {
            let valid: Vec<Beam> = all_beams
                .iter()
                .filter(|b| {
                    b.is_finite()
                        && !b.is_zero()
                        && args.shared.cutoff.is_none_or(|c| b.major_arcsec() <= c)
                })
                .cloned()
                .collect();
            anyhow::ensure!(!valid.is_empty(), "all beams are flagged or invalid");
            let sp = spinner("Solving for the common beam…");
            let cb = common_beam(
                &valid,
                args.shared.tolerance,
                args.shared.nsamps,
                args.shared.epsilon,
            )
            .context("could not find common beam")?;
            sp.finish_and_clear();
            cb
        }
    };

    common = apply_beam_rounding(common, args.shared.circularise)?;

    info!("Common beam: {common}");

    if args.shared.dryrun {
        // Emit the result on stdout (tracing logs to stderr) so `--dryrun` stays
        // machine-readable for callers that capture it.
        println!("{common}");
        info!("Dry run — no files written.");
        return Ok(());
    }

    let pb = progress_bar(files.len() as u64);

    let results: Vec<BeamLogEntry2D> = files
        .par_iter()
        .zip(all_beams.par_iter())
        .map(|(file, old_beam)| {
            pb.suspend(|| debug!("Reading {}", file.display()));
            let data = read_fits(file).with_context(|| format!("reading {}", file.display()))?;
            let out = output_path(
                file,
                Some(&args.shared.suffix),
                args.shared.prefix.as_deref(),
                args.shared.outdir.as_deref(),
            );
            let conv_beam = common.deconvolve_or_zero(old_beam);
            pb.suspend(|| {
                debug!(
                    "{}: current {old_beam} | target {common} | kernel {conv_beam}",
                    file.display()
                )
            });
            let smoothed = smooth(
                &data.image,
                old_beam,
                &common,
                data.dx_deg,
                data.dy_deg,
                args.shared.cutoff,
                data.unit,
            )
            .with_context(|| format!("smoothing {}", file.display()))?;
            pb.suspend(|| debug!("Writing {}", out.display()));
            write_fits(&smoothed, &out, file, &common, data.is_4d)
                .with_context(|| format!("writing {}", out.display()))?;
            pb.suspend(|| info!("{} → {}", file.display(), out.display()));
            pb.inc(1);
            Ok(BeamLogEntry2D {
                filename: out,
                old_beam: *old_beam,
                new_beam: common,
                conv_beam,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_with_message("done");

    if let Some(log_path) = &args.log {
        use std::fmt::Write as _;
        let mut out = String::from(
            "# FileName OldBMAJ[deg] OldBMIN[deg] OldBPA[deg] TargetBMAJ[deg] TargetBMIN[deg] TargetBPA[deg] ConvBMAJ[deg] ConvBMIN[deg] ConvBPA[deg]\n",
        );
        for e in &results {
            writeln!(
                out,
                "{} {} {} {} {} {} {} {} {} {}",
                e.filename.display(),
                e.old_beam.major_deg,
                e.old_beam.minor_deg,
                e.old_beam.pa_deg,
                e.new_beam.major_deg,
                e.new_beam.minor_deg,
                e.new_beam.pa_deg,
                e.conv_beam.major_deg,
                e.conv_beam.minor_deg,
                e.conv_beam.pa_deg,
            )?;
        }
        std::fs::write(log_path, out)?;
        info!("Beamlog written to {}", log_path.display());
    }

    Ok(())
}

// ── 3D subcommand ─────────────────────────────────────────────────────────────

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum ModeArg {
    /// Per-channel common beam across all input cubes.
    Natural,
    /// Single common beam across all channels and cubes.
    Total,
}

#[derive(Parser, Debug)]
struct ThreeDArgs {
    /// Input FITS spectral cube(s).
    #[arg(required = true, num_args = 1..)]
    infile: Vec<PathBuf>,

    /// Treat a single infile as a text file listing one path per line.
    #[arg(long)]
    listfile: bool,

    #[command(flatten)]
    shared: SharedArgs,

    /// Common-beam mode.
    #[arg(long, default_value = "natural", value_enum)]
    mode: ModeArg,
}

fn cmd_3d(args: ThreeDArgs) -> Result<()> {
    init_logging(args.shared.verbose);

    let files = collect_files(&args.infile, args.listfile)?;

    let sp = spinner(format!("Reading metadata from {} cube(s)…", files.len()));
    let metas: Vec<CubeMeta> = files
        .iter()
        .map(|f| {
            sp.suspend(|| debug!("Reading metadata + per-channel beams from {}", f.display()));
            let m = cube_io::read_cube_meta(f)
                .with_context(|| format!("reading metadata from {}", f.display()))?;
            sp.suspend(|| {
                debug!(
                    "{}: {}×{} px, {} channels, {} Stokes",
                    f.display(),
                    m.nx,
                    m.ny,
                    m.nfreq,
                    m.nstokes
                )
            });
            Ok(m)
        })
        .collect::<Result<_>>()?;
    sp.finish_and_clear();
    info!("Read metadata from {} cube(s)", files.len());

    let nfreq = metas[0].nfreq;
    for (f, m) in files.iter().zip(metas.iter()) {
        anyhow::ensure!(
            m.nfreq == nfreq,
            "{}: expected {} channels, got {}",
            f.display(),
            nfreq,
            m.nfreq
        );
        anyhow::ensure!(
            m.nstokes <= 1,
            "{}: NAXIS4={} (multiple Stokes) is not supported — only Stokes 0 \
             would be convolved while the other Stokes planes are written as \
             zeros, producing a misleading cube. Extract a single Stokes plane \
             first.",
            f.display(),
            m.nstokes
        );
    }

    let target_beam = parse_target_beam(&args.shared)?;

    let target_beams: Vec<Option<Beam>> = if let Some(b) = target_beam {
        let all_valid: Vec<Beam> = metas
            .iter()
            .flat_map(|m| m.beams.iter())
            .filter_map(|b| *b)
            .filter(|b| b.is_finite() && !b.is_zero())
            .collect();
        if !fits_in_beam(&all_valid, &b) {
            bail!("target beam is too small — some channels cannot reach it");
        }
        vec![Some(b); nfreq]
    } else {
        let mode = match args.mode {
            ModeArg::Natural => CubeMode::Natural,
            ModeArg::Total => CubeMode::Total,
        };
        let sp = spinner(match mode {
            CubeMode::Natural => "Solving for per-channel common beams…".to_string(),
            CubeMode::Total => "Solving for the common beam across all channels…".to_string(),
        });
        let beams = compute_target_beams(
            &metas,
            mode,
            args.shared.cutoff,
            args.shared.circularise,
            args.shared.tolerance,
            args.shared.nsamps,
            args.shared.epsilon,
        )?;
        sp.finish_and_clear();
        beams
    };

    // Report the target beam(s).  In `total` mode (or with an explicit target) all
    // channels share one beam, so print it directly.  In `natural` mode every channel
    // has its own target, so summarise the count and defer the per-channel detail to
    // verbose logging in the processing loop below.
    let n_valid = target_beams.iter().filter(|b| b.is_some()).count();
    let all_same = target_beams
        .iter()
        .filter_map(|b| *b)
        .collect::<Vec<_>>()
        .windows(2)
        .all(|w| w[0] == w[1]);
    match target_beams.iter().find_map(|b| *b) {
        Some(b) if all_same => info!("Target beam (all channels): {b}"),
        Some(b) => {
            info!("Target beam varies per channel ({n_valid} valid channels); e.g. channel 0: {b}");
            info!("Run with -v to log the current/target/kernel beam for every channel.");
        }
        None => {}
    }

    if args.shared.dryrun {
        // Emit the resolved target beam(s) on stdout (tracing logs to stderr) so
        // `--dryrun` stays machine-readable: one beam when all channels share it,
        // otherwise one `<channel> <beam>` line per channel.
        if all_same {
            if let Some(b) = target_beams.iter().find_map(|b| *b) {
                println!("{b}");
            }
        } else {
            for (c, b) in target_beams.iter().enumerate() {
                match b {
                    Some(b) => println!("{c} {b}"),
                    None => println!("{c} masked"),
                }
            }
        }
        info!("Dry run — no files written.");
        return Ok(());
    }

    let cube_mode = match args.mode {
        ModeArg::Natural => CubeMode::Natural,
        ModeArg::Total => CubeMode::Total,
    };

    let pb = progress_bar((files.len() * nfreq) as u64);

    for (file, meta) in files.iter().zip(metas.iter()) {
        let out = output_path(
            file,
            Some(&args.shared.suffix),
            args.shared.prefix.as_deref(),
            args.shared.outdir.as_deref(),
        );

        // Cube initialisation copies the full primary header and (in natural mode)
        // writes the BEAMS table — a potentially slow IO step on large cubes, so
        // announce it.  `pb.suspend` keeps the log line from clobbering the bar.
        pb.suspend(|| info!("Initialising output cube {} …", out.display()));
        pb.set_message("initialising");
        cube_io::init_output_cube(file, &out, &target_beams, cube_mode, meta)
            .with_context(|| format!("initialising output cube {}", out.display()))?;

        // Stream channels through a bounded pipeline instead of materialising the
        // whole cube in RAM (see `process_cube`).  Dispatch on the cube's pixel
        // precision so the FFT runs at the data's native precision: f32 cubes
        // (the common case) transform in f32, genuine f64 cubes in f64.
        pb.set_message("processing");

        match meta.dtype {
            cube_io::PixelType::F32 => {
                process_cube::<f32>(file, &out, meta, &target_beams, args.shared.cutoff, &pb)?
            }
            cube_io::PixelType::F64 => {
                process_cube::<f64>(file, &out, meta, &target_beams, args.shared.cutoff, &pb)?
            }
        }

        let beamlog = {
            let dir = out.parent().unwrap_or(Path::new("."));
            let stem = out.file_stem().unwrap_or_default();
            dir.join(format!("beamlog.{}.txt", stem.to_string_lossy()))
        };
        cube_io::write_beamlog(&beamlog, &target_beams)
            .with_context(|| format!("writing beamlog {}", beamlog.display()))?;
        pb.suspend(|| debug!("Beamlog written to {}", beamlog.display()));

        pb.suspend(|| info!("{} → {}", file.display(), out.display()));
    }

    pb.finish_with_message("done");
    Ok(())
}

/// Stream every channel of one cube through the bounded convolution pipeline at
/// pixel precision `T`.
///
/// rayon convolves planes in parallel (CPU- and memory-bandwidth bound) and
/// sends each finished plane to a single writer thread that owns the output cube
/// and writes sequentially (cfitsio is not thread-safe).  The bounded channel
/// caps peak memory to the in-flight planes — not the whole cube — and overlaps
/// convolution with disk IO.  Threading (not async) fits: the work is CPU-bound
/// and FITS IO is blocking, so an async runtime would buy nothing.
///
/// One [`FftPlans`] is built up front and shared by reference across all workers,
/// so the FFTs for every channel reuse the same plans instead of re-planning per
/// channel.
fn process_cube<T: CubeElem>(
    file: &Path,
    out: &Path,
    meta: &CubeMeta,
    target_beams: &[Option<Beam>],
    cutoff: Option<f64>,
    pb: &ProgressBar,
) -> Result<()> {
    // Plans depend only on the image dimensions, shared by all channels.
    let plans = FftPlans::<T>::new(meta.ny, meta.nx);

    // Bound the in-flight queue by a byte budget, not a plane count, so it
    // actually back-pressures (see `channel_cap`).
    let plane_bytes = meta.ny * meta.nx * std::mem::size_of::<T>();
    let cap = channel_cap(plane_bytes);

    // Native-float cubes (BITPIX -32/-64) store IEEE floats verbatim, so a plane
    // can be byte-swapped to big-endian in the parallel producers and the writer
    // just `pwrite`s the bytes — moving the swap off the single writer thread,
    // which was the cube pipeline's wall.  Integer cubes need cfitsio's
    // float→int conversion + scaling, so they keep the (slower) cfitsio writer.
    // `pwrite`-based positioned writes need a Unix `FileExt`, so gate on it.
    if cfg!(unix) && meta.is_native_float() {
        process_cube_raw::<T>(file, out, meta, target_beams, cutoff, pb, &plans, cap)
    } else {
        process_cube_cfitsio::<T>(file, out, meta, target_beams, cutoff, pb, &plans, cap)
    }
}

/// Convolve one channel to its target beam, returning the finished plane.
///
/// Returns an all-NaN plane (explicit no-data) when the channel cannot be
/// convolved — masked/zero source or target beam, or a beam above `cutoff`. A
/// zero source beam would otherwise make the analytic UV filter's gain diverge
/// to infinity, so blanking is deliberate rather than zero-fill or inf.
fn convolve_channel<T: CubeElem>(
    file: &Path,
    c: usize,
    meta: &CubeMeta,
    target_beams: &[Option<Beam>],
    cutoff: Option<f64>,
    plans: &FftPlans<T>,
    pb: &ProgressBar,
) -> Result<Array2<T>> {
    let nan_plane = || Array2::from_elem((meta.ny, meta.nx), T::nan());

    let old_beam = match meta.beams[c] {
        Some(b) if !b.is_zero() => b,
        _ => return Ok(nan_plane()),
    };
    let target = match target_beams[c] {
        Some(b) if !b.is_zero() => b,
        _ => return Ok(nan_plane()),
    };

    // Verbose (-v) per-channel beam report: current, target, and the convolving
    // kernel (target deconvolved from the current beam). Route through
    // `pb.suspend` so the log never corrupts the live bar.
    let kernel = target.deconvolve_or_zero(&old_beam);
    pb.suspend(|| debug!("Channel {c}: current {old_beam} | target {target} | kernel {kernel}"));

    if let Some(cut) = cutoff
        && old_beam.major_arcsec() > cut
    {
        pb.suspend(|| {
            warn!(
                "Channel {c}: BMAJ={:.1}\" > cutoff — blanking",
                old_beam.major_arcsec()
            )
        });
        return Ok(nan_plane());
    }

    let raw = cube_io::read_channel_as::<T>(file, c, meta)
        .with_context(|| format!("reading channel {c} from {}", file.display()))?;
    smooth_with_plans(
        &raw,
        &old_beam,
        &target,
        meta.dx_deg,
        meta.dy_deg,
        cutoff,
        meta.unit,
        plans,
    )
    .with_context(|| format!("smoothing channel {c}"))
}

/// Fast path for native-float cubes: producers byte-swap each plane to FITS
/// big-endian in parallel, and a single writer thread `pwrite`s the raw bytes at
/// the channel's offset — pure I/O, no per-element swap on the writer.
#[allow(clippy::too_many_arguments)]
fn process_cube_raw<T: CubeElem>(
    file: &Path,
    out: &Path,
    meta: &CubeMeta,
    target_beams: &[Option<Beam>],
    cutoff: Option<f64>,
    pb: &ProgressBar,
    plans: &FftPlans<T>,
    cap: usize,
) -> Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;

    let nfreq = meta.nfreq;
    let plane_bytes = meta.ny * meta.nx * std::mem::size_of::<T>();
    let data_offset = cube_io::primary_data_offset(out)
        .with_context(|| format!("locating data unit in {}", out.display()))?;

    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<u8>)>(cap);

    std::thread::scope(|s| {
        // Single writer thread — owns one OS handle and writes already-swapped
        // bytes at disjoint offsets. `write_all_at` (pwrite) does not move a
        // shared cursor, so the work here is I/O only.
        let writer_handle = s.spawn(move || -> Result<()> {
            #[cfg(unix)]
            {
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .open(out)
                    .with_context(|| format!("opening output cube {}", out.display()))?;
                for (c, bytes) in rx {
                    let off = data_offset + (c * plane_bytes) as u64;
                    f.write_all_at(&bytes, off)
                        .with_context(|| format!("writing channel {c} to {}", out.display()))?;
                }
                Ok(())
            }
            #[cfg(not(unix))]
            {
                let _ = (&rx, data_offset, plane_bytes, out);
                unreachable!("raw write path is gated on unix in process_cube")
            }
        });

        // Parallel producers — convolve, byte-swap, and stream raw bytes.
        let produce: Result<()> = (0..nfreq).into_par_iter().try_for_each(|c| {
            let plane = convolve_channel::<T>(file, c, meta, target_beams, cutoff, plans, pb)?;
            let bytes = T::plane_to_be_bytes(&plane);
            tx.send((c, bytes))
                .map_err(|_| anyhow::anyhow!("writer thread stopped before channel {c}"))?;
            pb.inc(1);
            Ok(())
        });

        // Close the channel so the writer loop ends, then join it. Prefer the
        // writer's error (the real cause) over the producers' generic "writer
        // stopped" when both fail.
        drop(tx);
        let writer_result = writer_handle
            .join()
            .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
        writer_result.and(produce)
    })
}

/// Fallback path for integer cubes: a single cfitsio writer applies the
/// float→int conversion and scaling that raw byte writes cannot.
#[allow(clippy::too_many_arguments)]
fn process_cube_cfitsio<T: CubeElem>(
    file: &Path,
    out: &Path,
    meta: &CubeMeta,
    target_beams: &[Option<Beam>],
    cutoff: Option<f64>,
    pb: &ProgressBar,
    plans: &FftPlans<T>,
    cap: usize,
) -> Result<()> {
    let nfreq = meta.nfreq;
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Array2<T>)>(cap);

    std::thread::scope(|s| {
        // Single writer thread — owns the output FITS handle.  `FitsFile` holds a
        // raw cfitsio pointer and is not `Send`, so it is opened *on* this thread
        // and never crosses a thread boundary.
        let writer_handle = s.spawn(move || -> Result<()> {
            let mut writer = cube_io::CubeWriter::open(out)
                .with_context(|| format!("opening output cube {}", out.display()))?;
            for (c, plane) in rx {
                writer
                    .write_channel_as::<T>(c, &plane, meta)
                    .with_context(|| format!("writing channel {c} to {}", out.display()))?;
            }
            Ok(())
        });

        // Parallel producers — convolve and stream finished planes to the writer.
        let produce: Result<()> = (0..nfreq).into_par_iter().try_for_each(|c| {
            let plane = convolve_channel::<T>(file, c, meta, target_beams, cutoff, plans, pb)?;
            tx.send((c, plane))
                .map_err(|_| anyhow::anyhow!("writer thread stopped before channel {c}"))?;
            pb.inc(1);
            Ok(())
        });

        drop(tx);
        let writer_result = writer_handle
            .join()
            .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
        writer_result.and(produce)
    })
}

/// In-flight queue depth for convolved planes, chosen so the bounded channel
/// actually back-pressures.
///
/// The old `2 * num_threads` cap counted *planes*: when `nfreq` was below it the
/// channel never blocked, so the whole convolved cube plus per-worker scratch
/// buffered in RAM (observed: 113 GB RSS for a 27 GB cube). Bounding by a byte
/// budget instead keeps peak memory near `budget` regardless of plane size,
/// while the `2 * num_threads` ceiling preserves enough buffering to keep the
/// convolvers fed and the floor of 4 keeps small cubes pipelined.
fn channel_cap(plane_bytes: usize) -> usize {
    cap_from_budget(
        mem_budget_bytes(),
        plane_bytes,
        (rayon::current_num_threads() * 2).max(4),
    )
}

/// Pure core of [`channel_cap`]: how many planes fit in `budget`, clamped to
/// `[4, ceiling]`. Split out from the env/thread lookups so it can be tested.
fn cap_from_budget(budget: u64, plane_bytes: usize, ceiling: usize) -> usize {
    let by_budget = (budget / plane_bytes.max(1) as u64) as usize;
    by_budget.clamp(4, ceiling)
}

/// Memory budget (bytes) for the in-flight plane queue.
///
/// Override with `CONVOLVERS_MEM_BUDGET_MB`; otherwise use a quarter of the
/// system's available memory (Linux `/proc/meminfo`), falling back to 4 GiB when
/// that cannot be read. This bounds only the queue — per-worker FFT scratch is
/// separate — so a conservative fraction leaves headroom for it.
fn mem_budget_bytes() -> u64 {
    const FALLBACK: u64 = 4 << 30; // 4 GiB
    if let Ok(s) = std::env::var("CONVOLVERS_MEM_BUDGET_MB")
        && let Ok(mb) = s.trim().parse::<u64>()
        && mb > 0
    {
        return mb.saturating_mul(1 << 20);
    }
    available_memory_bytes().map_or(FALLBACK, |avail| (avail / 4).max(1 << 30))
}

/// Available system memory in bytes from Linux `/proc/meminfo` (`MemAvailable`),
/// or `None` where it cannot be determined.
fn available_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    // Format: "MemAvailable:   12345678 kB"
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb.saturating_mul(1024))
}

fn compute_target_beams(
    metas: &[CubeMeta],
    mode: CubeMode,
    cutoff: Option<f64>,
    circularise: bool,
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
) -> Result<Vec<Option<Beam>>> {
    let nfreq = metas[0].nfreq;
    match mode {
        CubeMode::Natural => (0..nfreq)
            .map(|c| {
                let valid: Vec<Beam> = metas
                    .iter()
                    .filter_map(|m| m.beams[c])
                    .filter(|b| b.is_finite() && !b.is_zero())
                    .filter(|b| cutoff.is_none_or(|cut| b.major_arcsec() <= cut))
                    .collect();
                if valid.is_empty() {
                    return Ok(None);
                }
                let cb = common_beam(&valid, tolerance, nsamps, epsilon)
                    .with_context(|| format!("finding common beam for channel {c}"))?;
                Ok(Some(apply_beam_rounding(cb, circularise)?))
            })
            .collect(),
        CubeMode::Total => {
            let valid: Vec<Beam> = metas
                .iter()
                .flat_map(|m| m.beams.iter())
                .filter_map(|b| *b)
                .filter(|b| b.is_finite() && !b.is_zero())
                .filter(|b| cutoff.is_none_or(|cut| b.major_arcsec() <= cut))
                .collect();
            anyhow::ensure!(
                !valid.is_empty(),
                "no valid beams found across all cubes/channels"
            );
            let cb = common_beam(&valid, tolerance, nsamps, epsilon)
                .context("finding total common beam")?;
            let cb = apply_beam_rounding(cb, circularise)?;
            Ok(vec![Some(cb); nfreq])
        }
    }
}

// ── Shared utilities ──────────────────────────────────────────────────────────

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => tracing::Level::INFO,
        1 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();
}

fn collect_files(infile: &[PathBuf], listfile: bool) -> Result<Vec<PathBuf>> {
    let files = if listfile {
        anyhow::ensure!(infile.len() == 1, "only one listfile argument supported");
        std::fs::read_to_string(&infile[0])?
            .lines()
            .map(|l| PathBuf::from(l.trim()))
            .collect()
    } else {
        infile.to_vec()
    };
    anyhow::ensure!(!files.is_empty(), "no input files found");
    Ok(files)
}

fn parse_target_beam(args: &SharedArgs) -> Result<Option<Beam>> {
    match (args.bmaj, args.bmin, args.bpa) {
        (None, None, None) => Ok(None),
        (Some(bmaj), Some(bmin), Some(bpa)) => Ok(Some(
            Beam::from_arcsec(bmaj, bmin, bpa).context("invalid target beam")?,
        )),
        _ => bail!("--bmaj, --bmin, and --bpa must all be specified together"),
    }
}

fn apply_beam_rounding(b: Beam, circularise: bool) -> Result<Beam> {
    let b = Beam::from_arcsec(
        ceil_to(b.major_arcsec(), 1),
        ceil_to(b.minor_arcsec(), 1),
        round_up(b.pa_deg, 2),
    )
    .context("rounding common beam")?;
    if circularise {
        Beam::from_arcsec(b.major_arcsec(), b.major_arcsec(), 0.0).context("circularising beam")
    } else {
        Ok(b)
    }
}

fn progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    // Steady tick animates the `{spinner}` even when `pos` is not advancing, so
    // idle-but-busy phases (e.g. `init_output_cube` before the first channel is
    // written) still show live activity rather than a frozen bar.
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// An indeterminate spinner for blocking phases that have no item count and run
/// outside the main progress bar (e.g. reading metadata, solving for a common
/// beam).  Caller drives it with `finish_and_clear` when the work completes.
fn spinner(msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed}] {msg}")
            .unwrap(),
    );
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

fn ceil_to(x: f64, precision: i32) -> f64 {
    let factor = 10_f64.powi(precision);
    (x * factor).ceil() / factor
}

fn round_up(x: f64, decimals: i32) -> f64 {
    let factor = 10_f64.powi(decimals);
    (x * factor).ceil() / factor
}

#[cfg(test)]
mod tests {
    use super::cap_from_budget;

    const GIB: u64 = 1 << 30;

    /// The byte budget back-pressures: a budget smaller than the cube means the
    /// cap is well below the channel count, so the queue actually blocks. This
    /// is the regression the byte-budget cap fixes — the old plane-count cap
    /// (`2 * threads`) never blocked when `nfreq` was below it.
    #[test]
    fn cap_bounded_by_byte_budget_not_plane_count() {
        let plane = 419 * (1 << 20); // ~419 MiB (10240² f32)
        // 4 GiB budget over a 27 GiB / 64-plane cube → ~9 planes in flight,
        // far below both the 64 channels and a 192 plane-count ceiling.
        let cap = cap_from_budget(4 * GIB, plane, 192);
        assert_eq!(cap, 9);
        assert!(cap < 64, "must block before buffering the whole cube");
    }

    #[test]
    fn cap_has_floor_of_four() {
        // A plane larger than the whole budget still keeps a small pipeline.
        assert_eq!(cap_from_budget(GIB, 4 * GIB as usize, 192), 4);
    }

    #[test]
    fn cap_capped_by_ceiling_for_tiny_planes() {
        // Tiny planes would allow a huge cap; the thread-derived ceiling holds.
        assert_eq!(cap_from_budget(64 * GIB, 1 << 16, 192), 192);
    }
}
