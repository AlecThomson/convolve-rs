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
    cube_io::PixelType,
    cube_io::{self, CubeElem, CubeMeta, CubeMode},
    fits_io::{FitsImageHeader, output_path, read_fits_header, read_fits_plane, write_fits},
    smooth::smooth_owned_with_plans,
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
    /// `None` for a blanked image.
    conv_beam: Option<Beam>,
}

fn cmd_2d(args: TwoDArgs) -> Result<()> {
    init_logging(args.shared.verbose);

    let files = collect_files(&args.infile, args.listfile)?;
    let target_beam = parse_target_beam(&args.shared)?;
    let cutoff = args.shared.cutoff;

    let sp = spinner(format!(
        "Reading beam parameters from {} file(s)…",
        files.len()
    ));
    let headers: Vec<FitsImageHeader> = files
        .iter()
        .map(|f| read_fits_header(f).with_context(|| format!("reading {}", f.display())))
        .collect::<Result<_>>()?;
    sp.finish_and_clear();
    for h in &headers {
        if !usable_beam(&h.beam, None) {
            warn!(
                "{}: beam {} has zero size; will be blanked",
                h.path.display(),
                h.beam
            );
        } else if !usable_beam(&h.beam, cutoff) {
            warn!(
                "{}: BMAJ={:.1}\" > cutoff={:.1}\"; will be blanked",
                h.path.display(),
                h.beam.major_arcsec(),
                cutoff.unwrap_or_default()
            );
        }
    }

    let valid: Vec<Beam> = headers
        .iter()
        .map(|h| h.beam)
        .filter(|b| usable_beam(b, cutoff))
        .collect();
    anyhow::ensure!(!valid.is_empty(), "all beams are flagged or invalid");
    let common = match target_beam {
        Some(b) => b,
        None => {
            let sp = spinner("Solving for the common beam…");
            let cb = common_beam(
                &valid,
                args.shared.tolerance,
                args.shared.nsamps,
                args.shared.epsilon,
            )
            .context("could not find common beam")?;
            sp.finish_and_clear();
            round_common_beam(cb, &valid)?
        }
    };
    let common = finish_target(common, &valid, args.shared.circularise)?;

    info!("Common beam: {common}");

    if args.shared.dryrun {
        // Emit the result on stdout (tracing logs to stderr) so `--dryrun` stays
        // machine-readable for callers that capture it.
        println!("{common}");
        info!("Dry run: no files written.");
        return Ok(());
    }

    let pb = progress_bar(files.len() as u64);

    let results: Vec<BeamLogEntry2D> = headers
        .par_iter()
        .map(|header| {
            let file = &header.path;
            let out = output_path(
                file,
                Some(&args.shared.suffix),
                args.shared.prefix.as_deref(),
                args.shared.outdir.as_deref(),
            );
            let old_beam = header.beam;
            let conv_beam =
                usable_beam(&old_beam, cutoff).then(|| common.deconvolve_or_zero(&old_beam));
            if let Some(conv_beam) = conv_beam {
                pb.suspend(|| {
                    debug!(
                        "{}: current {old_beam} | target {common} | kernel {conv_beam}",
                        file.display()
                    )
                });
            }
            let blank = conv_beam.is_none();
            match header.dtype {
                PixelType::F32 => smooth_image::<f32>(header, &out, &common, blank),
                PixelType::F64 => smooth_image::<f64>(header, &out, &common, blank),
            }?;
            pb.suspend(|| info!("{} → {}", file.display(), out.display()));
            pb.inc(1);
            Ok(BeamLogEntry2D {
                filename: out,
                old_beam,
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
            let conv = e.conv_beam.map_or("nan nan nan".to_string(), |b| {
                format!("{} {} {}", b.major_deg, b.minor_deg, b.pa_deg)
            });
            writeln!(
                out,
                "{} {} {} {} {} {} {} {conv}",
                e.filename.display(),
                e.old_beam.major_deg,
                e.old_beam.minor_deg,
                e.old_beam.pa_deg,
                e.new_beam.major_deg,
                e.new_beam.minor_deg,
                e.new_beam.pa_deg,
            )?;
        }
        std::fs::write(log_path, out)?;
        info!("Beamlog written to {}", log_path.display());
    }

    Ok(())
}

/// Smooth one image to `target` at pixel precision `T`, or write it as NaN if
/// `blank` is set.
fn smooth_image<T: CubeElem>(
    header: &FitsImageHeader,
    out: &Path,
    target: &Beam,
    blank: bool,
) -> Result<()> {
    let file = &header.path;
    let image = if blank {
        Array2::from_elem((header.ny, header.nx), T::nan())
    } else {
        let data =
            read_fits_plane::<T>(header).with_context(|| format!("reading {}", file.display()))?;
        let plans = FftPlans::new(header.ny, header.nx);
        smooth_owned_with_plans(
            data,
            &header.beam,
            target,
            header.dx_deg,
            header.dy_deg,
            None,
            header.unit,
            &plans,
        )
        .with_context(|| format!("smoothing {}", file.display()))?
    };
    write_fits(&image, out, file, target, header.is_4d)
        .with_context(|| format!("writing {}", out.display()))
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
    info!(
        "Read metadata from {} cube(s), {} channel(s) each",
        files.len(),
        metas[0].nfreq
    );

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
            "{}: {} Stokes planes are not supported — only Stokes 0 \
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
            .filter(|b| usable_beam(b, args.shared.cutoff))
            .collect();
        let b = finish_target(b, &all_valid, args.shared.circularise)?;
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

        // The output cube is created and held open by the writer thread inside
        // `process_cube` (see `CubeWriter::create`): the single FITS handle stays
        // open from creation through every channel write, so cfitsio writes the
        // data unit exactly once instead of flushing a full pass of zeros on a
        // create-close before the real planes overwrite it.  Announce the step.
        pb.suspend(|| info!("Initialising output cube {} …", out.display()));
        pb.set_message("processing");

        // Stream channels through a bounded pipeline instead of materialising the
        // whole cube in RAM (see `process_cube`).  Dispatch on the cube's pixel
        // precision so the FFT runs at the data's native precision: f32 cubes
        // (the common case) transform in f32, genuine f64 cubes in f64.
        match meta.dtype {
            cube_io::PixelType::F32 => process_cube::<f32>(
                file,
                &out,
                meta,
                &target_beams,
                cube_mode,
                args.shared.cutoff,
                &pb,
            )?,
            cube_io::PixelType::F64 => process_cube::<f64>(
                file,
                &out,
                meta,
                &target_beams,
                cube_mode,
                args.shared.cutoff,
                &pb,
            )?,
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
    cube_mode: CubeMode,
    cutoff: Option<f64>,
    pb: &ProgressBar,
) -> Result<()> {
    // Plans depend only on the image dimensions, shared by all channels.
    let plans = FftPlans::<T>::new(meta.ny, meta.nx);
    let nfreq = meta.nfreq;

    let cap = (rayon::current_num_threads() * 2).max(4);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Array2<T>)>(cap);

    std::thread::scope(|s| {
        // Single writer thread — creates and owns the output FITS handle.
        // `FitsFile` holds a raw cfitsio pointer and is not `Send`, so the handle
        // is created *on* this thread and never crosses a thread boundary. Keeping
        // it open from creation through every channel write to `finish` means
        // cfitsio writes the data unit exactly once (no wasted zero-fill pass).
        let writer_handle = s.spawn(move || -> Result<()> {
            let mut writer = cube_io::CubeWriter::create(file, out, target_beams, cube_mode, meta)
                .with_context(|| format!("initialising output cube {}", out.display()))?;
            for (c, plane) in rx {
                writer
                    .write_channel_as::<T>(c, &plane, meta)
                    .with_context(|| format!("writing channel {c} to {}", out.display()))?;
            }
            writer
                .finish()
                .with_context(|| format!("finalising output cube {}", out.display()))?;
            Ok(())
        });

        // Parallel producers — convolve and stream finished planes to the writer.
        let produce: Result<()> = (0..nfreq).into_par_iter().try_for_each(|c| {
            // Hand a finished plane to the writer; a send error means the writer
            // thread already exited (i.e. a write failed) — surfaced below.
            let send = |plane: Array2<T>| -> Result<()> {
                tx.send((c, plane))
                    .map_err(|_| anyhow::anyhow!("writer thread stopped before channel {c}"))?;
                pb.inc(1);
                Ok(())
            };
            let nan_plane = || Array2::from_elem((meta.ny, meta.nx), T::nan());

            // A channel with no source beam, no target, or a degenerate zero beam
            // cannot be convolved — a zero source beam makes the analytic UV
            // filter's gain (g_ratio) diverge to infinity. Blank it with NaNs so
            // it is written as explicit no-data rather than left as the output's
            // zero-fill or corrupted to infinity.
            let old_beam = match meta.beams[c] {
                Some(b) if usable_beam(&b, None) => b,
                _ => return send(nan_plane()),
            };
            let target = match target_beams[c] {
                Some(b) if usable_beam(&b, None) => b,
                _ => return send(nan_plane()),
            };

            // Verbose (-v) per-channel beam report: current, target, and the
            // convolving kernel (target deconvolved from the current beam).
            // Route through `pb.suspend` so the log never corrupts the live bar.
            let kernel = target.deconvolve_or_zero(&old_beam);
            pb.suspend(|| {
                debug!("Channel {c}: current {old_beam} | target {target} | kernel {kernel}")
            });

            let plane = if let Some(cut) = cutoff
                && old_beam.major_arcsec() > cut
            {
                pb.suspend(|| {
                    warn!(
                        "Channel {c}: BMAJ={:.1}\" > cutoff — blanking",
                        old_beam.major_arcsec()
                    )
                });
                nan_plane()
            } else {
                let raw = cube_io::read_channel_as::<T>(file, c, meta)
                    .with_context(|| format!("reading channel {c} from {}", file.display()))?;
                // Owned: the channel is smoothed in the buffer it was read into.
                smooth_owned_with_plans(
                    raw,
                    &old_beam,
                    &target,
                    meta.dx_deg,
                    meta.dy_deg,
                    cutoff,
                    meta.unit,
                    &plans,
                )
                .with_context(|| format!("smoothing channel {c}"))?
            };

            send(plane)
        });

        // Close the channel so the writer loop ends, then join it.  Prefer the
        // writer's error (the real cause) over the producers' generic "writer
        // stopped" when both fail.
        drop(tx);
        let writer_result = writer_handle
            .join()
            .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
        writer_result.and(produce)
    })
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
                    .filter(|b| usable_beam(b, cutoff))
                    .collect();
                if valid.is_empty() {
                    return Ok(None);
                }
                let cb = common_beam(&valid, tolerance, nsamps, epsilon)
                    .with_context(|| format!("finding common beam for channel {c}"))?;
                let cb = round_common_beam(cb, &valid)?;
                Ok(Some(finish_target(cb, &valid, circularise)?))
            })
            .collect(),
        CubeMode::Total => {
            let valid: Vec<Beam> = metas
                .iter()
                .flat_map(|m| m.beams.iter())
                .filter_map(|b| *b)
                .filter(|b| usable_beam(b, cutoff))
                .collect();
            anyhow::ensure!(
                !valid.is_empty(),
                "no valid beams found across all cubes/channels"
            );
            let cb = common_beam(&valid, tolerance, nsamps, epsilon)
                .context("finding total common beam")?;
            let cb = round_common_beam(cb, &valid)?;
            let cb = finish_target(cb, &valid, circularise)?;
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

/// True for a beam that can be convolved and is within the cutoff (arcsec).
fn usable_beam(b: &Beam, cutoff: Option<f64>) -> bool {
    b.is_finite() && b.minor_deg > 0.0 && cutoff.is_none_or(|c| b.major_arcsec() <= c)
}

/// Round a computed common beam up to 0.1″ axes and 0.01° PA, as racs_tools
/// does. If the rotated PA leaves an input outside, add another 0.1″.
fn round_common_beam(b: Beam, inputs: &[Beam]) -> Result<Beam> {
    let round = |step: f64| {
        Beam::from_arcsec(
            ceil_to(b.major_arcsec() + step, 1),
            ceil_to(b.minor_arcsec() + step, 1),
            ceil_to(b.pa_deg, 2),
        )
        .context("rounding common beam")
    };
    let rounded = round(0.0)?;
    if fits_in_beam(inputs, &rounded) {
        Ok(rounded)
    } else {
        round(0.1)
    }
}

/// Circularise `b` if asked, then check every input beam can reach it.
fn finish_target(b: Beam, inputs: &[Beam], circularise: bool) -> Result<Beam> {
    let b = if circularise {
        Beam::from_arcsec(b.major_arcsec(), b.major_arcsec(), 0.0).context("circularising beam")?
    } else {
        b
    };
    if !fits_in_beam(inputs, &b) {
        bail!("target beam {b} is too small: some inputs cannot be convolved to it");
    }
    Ok(b)
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

/// Round `x` up to `decimals` places, ignoring round-off below 1e-9 of the
/// last place (so 30.500000000000004 stays 30.5).
fn ceil_to(x: f64, decimals: i32) -> f64 {
    let factor = 10_f64.powi(decimals);
    (x * factor - 1e-9).ceil() / factor
}
