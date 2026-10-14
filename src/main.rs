use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rayon::prelude::*;
use tracing::{Level, debug, info, warn};

use convolve_rs::{
    beam::Beam,
    common_beam::{common_beam, fits_in_beam},
    convolve_uv::FftPlans,
    cube_io::{self, CubeElem, CubeMeta, CubeMode, CubeReader, CubeWriter, PixelType},
    fits_io::{FitsImageHeader, output_path, read_fits_header, read_fits_plane, write_fits},
    smooth::smooth,
};

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
    match Cli::parse().command {
        Commands::TwoD(args) => cmd_2d(args),
        Commands::ThreeD(args) => cmd_3d(args),
    }
}

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

    /// Beam size cutoff in arcsec: blank images/channels with BMAJ larger than this.
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

impl SharedArgs {
    fn output_for(&self, file: &Path) -> PathBuf {
        output_path(
            file,
            Some(&self.suffix),
            self.prefix.as_deref(),
            self.outdir.as_deref(),
        )
    }

    fn common_beam(&self, beams: &[Beam]) -> Result<Beam> {
        let cb = common_beam(beams, self.tolerance, self.nsamps, self.epsilon)?;
        round_common_beam(cb, beams)
    }
}

// ── 2D ────────────────────────────────────────────────────────────────────────

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
    let cutoff = args.shared.cutoff;

    let sp = spinner(format!("Reading headers of {} file(s)…", files.len()));
    let headers: Vec<FitsImageHeader> = files
        .par_iter()
        .map(|f| read_fits_header(f).with_context(|| format!("reading {}", f.display())))
        .collect::<Result<_>>()?;
    sp.finish_and_clear();
    warn_blanked_images(&headers, cutoff);

    let valid: Vec<Beam> = headers
        .iter()
        .map(|h| h.beam)
        .filter(|b| usable_beam(b, cutoff))
        .collect();
    anyhow::ensure!(!valid.is_empty(), "all beams are flagged or invalid");
    let common = match parse_target_beam(&args.shared)? {
        Some(b) => b,
        None => {
            let sp = spinner("Solving for the common beam…");
            let cb = args.shared.common_beam(&valid)?;
            sp.finish_and_clear();
            cb
        }
    };
    let common = finish_target(common, &valid, args.shared.circularise)?;
    info!("Common beam: {common}");

    if args.shared.dryrun {
        // stdout, so callers can capture it; logs go to stderr.
        println!("{common}");
        info!("Dry run: no files written.");
        return Ok(());
    }

    let pb = progress_bar(files.len() as u64);
    let process = |header: &FitsImageHeader| -> Result<BeamLogEntry2D> {
        let out = args.shared.output_for(&header.path);
        let entry = smooth_image(header, &out, &common, cutoff)?;
        pb.suspend(|| info!("{} → {}", header.path.display(), out.display()));
        pb.inc(1);
        Ok(entry)
    };
    // With fewer images than threads, smooth one at a time and let each
    // convolution use the whole pool.
    let results: Vec<BeamLogEntry2D> = if headers.len() >= rayon::current_num_threads() {
        headers.par_iter().map(process).collect::<Result<_>>()?
    } else {
        headers.iter().map(process).collect::<Result<_>>()?
    };
    pb.finish_with_message("done");

    if let Some(log_path) = &args.log {
        write_2d_beamlog(log_path, &results)?;
        info!("Beamlog written to {}", log_path.display());
    }
    Ok(())
}

fn warn_blanked_images(headers: &[FitsImageHeader], cutoff: Option<f64>) {
    for h in headers {
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
}

/// Smooth one image to `target`, or write it as NaN if its beam is unusable.
fn smooth_image(
    header: &FitsImageHeader,
    out: &Path,
    target: &Beam,
    cutoff: Option<f64>,
) -> Result<BeamLogEntry2D> {
    let old_beam = header.beam;
    let conv_beam = usable_beam(&old_beam, cutoff).then(|| target.deconvolve_or_zero(&old_beam));
    if let Some(conv) = conv_beam {
        debug!(
            "{}: current {old_beam} | target {target} | kernel {conv}",
            header.path.display()
        );
    }
    let blank = conv_beam.is_none();
    match header.dtype {
        PixelType::F32 => smooth_image_as::<f32>(header, out, target, blank),
        PixelType::F64 => smooth_image_as::<f64>(header, out, target, blank),
    }?;
    Ok(BeamLogEntry2D {
        filename: out.to_path_buf(),
        old_beam,
        new_beam: *target,
        conv_beam,
    })
}

fn smooth_image_as<T: CubeElem>(
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
        smooth(
            data,
            &header.beam,
            target,
            header.dx_deg,
            header.dy_deg,
            header.unit,
            None,
        )
        .with_context(|| format!("smoothing {}", file.display()))?
    };
    write_fits(&image, out, file, target).with_context(|| format!("writing {}", out.display()))
}

fn write_2d_beamlog(path: &Path, entries: &[BeamLogEntry2D]) -> Result<()> {
    let deg = |b: &Beam| format!("{} {} {}", b.major_deg(), b.minor_deg(), b.pa_deg());
    let mut out = String::from(
        "# FileName OldBMAJ[deg] OldBMIN[deg] OldBPA[deg] TargetBMAJ[deg] TargetBMIN[deg] \
         TargetBPA[deg] ConvBMAJ[deg] ConvBMIN[deg] ConvBPA[deg]\n",
    );
    for e in entries {
        let conv = e.conv_beam.as_ref().map_or("nan nan nan".into(), deg);
        out += &format!(
            "{} {} {} {conv}\n",
            e.filename.display(),
            deg(&e.old_beam),
            deg(&e.new_beam)
        );
    }
    std::fs::write(path, out)?;
    Ok(())
}

// ── 3D ────────────────────────────────────────────────────────────────────────

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum ModeArg {
    /// Per-channel common beam across all input cubes.
    Natural,
    /// Single common beam across all channels and cubes.
    Total,
}

impl From<ModeArg> for CubeMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Natural => CubeMode::Natural,
            ModeArg::Total => CubeMode::Total,
        }
    }
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
    let metas = read_cube_metas(&files)?;
    let nfreq = metas[0].nfreq;
    let mode = CubeMode::from(args.mode);

    let target_beams = match parse_target_beam(&args.shared)? {
        Some(b) => {
            let valid: Vec<Beam> = metas
                .iter()
                .flat_map(|m| m.beams.iter().flatten())
                .copied()
                .filter(|b| usable_beam(b, args.shared.cutoff))
                .collect();
            vec![Some(finish_target(b, &valid, args.shared.circularise)?); nfreq]
        }
        None => {
            let sp = spinner(match mode {
                CubeMode::Natural => "Solving for per-channel common beams…",
                CubeMode::Total => "Solving for the common beam across all channels…",
            });
            let beams = compute_target_beams(&metas, mode, &args.shared)?;
            sp.finish_and_clear();
            beams
        }
    };

    let all_same = report_targets(&target_beams);
    if args.shared.dryrun {
        // stdout, so callers can capture it: one beam if all channels share
        // it, else one `<channel> <beam>` line per channel.
        match target_beams.iter().find_map(|b| *b) {
            Some(b) if all_same => println!("{b}"),
            _ => {
                for (c, b) in target_beams.iter().enumerate() {
                    match b {
                        Some(b) => println!("{c} {b}"),
                        None => println!("{c} masked"),
                    }
                }
            }
        }
        info!("Dry run: no files written.");
        return Ok(());
    }

    let pb = progress_bar((files.len() * nfreq) as u64);
    for meta in &metas {
        let file = &meta.path;
        let out = args.shared.output_for(file);
        pb.suspend(|| info!("Initialising output cube {} …", out.display()));
        pb.set_message("processing");
        match meta.dtype {
            PixelType::F32 => {
                process_cube::<f32>(&out, meta, &target_beams, mode, args.shared.cutoff, &pb)?
            }
            PixelType::F64 => {
                process_cube::<f64>(&out, meta, &target_beams, mode, args.shared.cutoff, &pb)?
            }
        }
        let beamlog = cube_io::beamlog_path(&out);
        cube_io::write_beamlog(&beamlog, &target_beams)
            .with_context(|| format!("writing beamlog {}", beamlog.display()))?;
        pb.suspend(|| info!("{} → {}", file.display(), out.display()));
    }
    pb.finish_with_message("done");
    Ok(())
}

/// Read every cube's metadata and check they can be smoothed together.
fn read_cube_metas(files: &[PathBuf]) -> Result<Vec<CubeMeta>> {
    let sp = spinner(format!("Reading metadata from {} cube(s)…", files.len()));
    let metas: Vec<CubeMeta> = files
        .par_iter()
        .map(|f| {
            cube_io::read_cube_meta(f)
                .with_context(|| format!("reading metadata from {}", f.display()))
        })
        .collect::<Result<_>>()?;
    sp.finish_and_clear();

    let nfreq = metas[0].nfreq;
    for m in &metas {
        debug!(
            "{}: {}×{} px, {} channels, {} Stokes",
            m.path.display(),
            m.nx,
            m.ny,
            m.nfreq,
            m.nstokes
        );
        anyhow::ensure!(
            m.nfreq == nfreq,
            "{}: expected {nfreq} channels, got {}",
            m.path.display(),
            m.nfreq
        );
        // Only the first Stokes plane is convolved, so the others would be
        // written as zeros.
        anyhow::ensure!(
            m.nstokes <= 1,
            "{}: {} Stokes planes are not supported; extract a single Stokes plane first.",
            m.path.display(),
            m.nstokes
        );
    }
    info!(
        "Read metadata from {} cube(s), {nfreq} channel(s) each",
        metas.len()
    );
    Ok(metas)
}

/// Log the target beams and return whether every channel shares one.
fn report_targets(target_beams: &[Option<Beam>]) -> bool {
    let valid: Vec<Beam> = target_beams.iter().flatten().copied().collect();
    let all_same = valid.windows(2).all(|w| w[0].approx_eq(&w[1]));
    match valid.first() {
        Some(b) if all_same => info!("Target beam (all channels): {b}"),
        Some(b) => {
            info!(
                "Target beam varies per channel ({} valid channels); e.g. channel 0: {b}",
                valid.len()
            );
            info!("Run with -v to log the current/target/kernel beam for every channel.");
        }
        None => {}
    }
    all_same
}

/// Smooth every channel of one cube at pixel precision `T`.
///
/// rayon workers convolve channels and send them to one writer thread, which
/// owns the output (cfitsio handles are not `Send`). The bounded channel caps
/// memory at the planes in flight.
fn process_cube<T: CubeElem>(
    out: &Path,
    meta: &CubeMeta,
    target_beams: &[Option<Beam>],
    mode: CubeMode,
    cutoff: Option<f64>,
    pb: &ProgressBar,
) -> Result<()> {
    let plans = FftPlans::<T>::new(meta.ny, meta.nx);
    let cap = (rayon::current_num_threads() * 2).max(4);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Array2<T>)>(cap);

    std::thread::scope(|s| {
        let writer = s.spawn(move || -> Result<()> {
            let mut writer = CubeWriter::create(&meta.path, out, target_beams, mode, meta)
                .with_context(|| format!("initialising output cube {}", out.display()))?;
            for (c, plane) in rx {
                writer
                    .write_channel(c, &plane, meta)
                    .with_context(|| format!("writing channel {c} to {}", out.display()))?;
            }
            writer
                .finish()
                .with_context(|| format!("finalising output cube {}", out.display()))
        });

        let produce = (0..meta.nfreq).into_par_iter().try_for_each_init(
            || CubeReader::open(&meta.path),
            |reader, c| {
                let reader = reader.as_mut().map_err(|e| anyhow::anyhow!("{e}"))?;
                let plane = smooth_channel(reader, c, meta, target_beams[c], cutoff, &plans, pb)?;
                // A send error means the writer stopped; its error is reported below.
                tx.send((c, plane))
                    .map_err(|_| anyhow::anyhow!("writer thread stopped before channel {c}"))?;
                pb.inc(1);
                Ok(())
            },
        );

        drop(tx);
        let written = writer
            .join()
            .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
        written.and(produce)
    })
}

/// Channel `c` smoothed to `target`, or NaN if it has no usable beam or is
/// over the cutoff.
fn smooth_channel<T: CubeElem>(
    reader: &mut CubeReader,
    c: usize,
    meta: &CubeMeta,
    target: Option<Beam>,
    cutoff: Option<f64>,
    plans: &FftPlans<T>,
    pb: &ProgressBar,
) -> Result<Array2<T>> {
    let nan_plane = || Array2::from_elem((meta.ny, meta.nx), T::nan());
    let (Some(old_beam), Some(target)) = (meta.beams[c], target) else {
        return Ok(nan_plane());
    };
    if !usable_beam(&old_beam, None) || !usable_beam(&target, None) {
        return Ok(nan_plane());
    }
    if !usable_beam(&old_beam, cutoff) {
        pb.suspend(|| {
            warn!(
                "Channel {c}: BMAJ={:.1}\" > cutoff; blanking",
                old_beam.major_arcsec()
            )
        });
        return Ok(nan_plane());
    }
    if tracing::enabled!(Level::DEBUG) {
        let kernel = target.deconvolve_or_zero(&old_beam);
        pb.suspend(|| {
            debug!("Channel {c}: current {old_beam} | target {target} | kernel {kernel}")
        });
    }
    let raw = reader
        .read_channel::<T>(c, meta)
        .with_context(|| format!("reading channel {c} from {}", meta.path.display()))?;
    smooth(
        raw,
        &old_beam,
        &target,
        meta.dx_deg,
        meta.dy_deg,
        meta.unit,
        Some(plans),
    )
    .with_context(|| format!("smoothing channel {c}"))
}

fn compute_target_beams(
    metas: &[CubeMeta],
    mode: CubeMode,
    args: &SharedArgs,
) -> Result<Vec<Option<Beam>>> {
    let nfreq = metas[0].nfreq;
    let usable = |c: usize| {
        metas
            .iter()
            .filter_map(move |m| m.beams[c])
            .filter(|b| usable_beam(b, args.cutoff))
    };
    match mode {
        CubeMode::Natural => (0..nfreq)
            .map(|c| {
                let valid: Vec<Beam> = usable(c).collect();
                if valid.is_empty() {
                    return Ok(None);
                }
                let cb = args
                    .common_beam(&valid)
                    .with_context(|| format!("finding common beam for channel {c}"))?;
                Ok(Some(finish_target(cb, &valid, args.circularise)?))
            })
            .collect(),
        CubeMode::Total => {
            let valid: Vec<Beam> = (0..nfreq).flat_map(usable).collect();
            anyhow::ensure!(
                !valid.is_empty(),
                "no valid beams found across all cubes/channels"
            );
            let cb = args
                .common_beam(&valid)
                .context("finding total common beam")?;
            Ok(vec![
                Some(finish_target(cb, &valid, args.circularise)?);
                nfreq
            ])
        }
    }
}

// ── Shared ────────────────────────────────────────────────────────────────────

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => Level::INFO,
        1 => Level::DEBUG,
        _ => Level::TRACE,
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
    b.has_area() && cutoff.is_none_or(|c| b.major_arcsec() <= c)
}

/// Round a computed common beam up to 0.1″ axes and 0.01° PA, as racs_tools
/// does. If the rotated PA leaves an input outside, add another 0.1″.
fn round_common_beam(b: Beam, inputs: &[Beam]) -> Result<Beam> {
    let round = |step: f64| {
        Beam::from_arcsec(
            ceil_to(b.major_arcsec() + step, 1),
            ceil_to(b.minor_arcsec() + step, 1),
            ceil_to(b.pa_deg(), 2),
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

/// Round `x` up to `decimals` places, ignoring round-off below 1e-9 of the
/// last place (so 30.500000000000004 stays 30.5).
fn ceil_to(x: f64, decimals: i32) -> f64 {
    let factor = 10_f64.powi(decimals);
    (x * factor - 1e-9).ceil() / factor
}

fn progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
    );
    // Keeps the spinner moving while no channel has finished yet.
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// A spinner for steps with no item count.
fn spinner(msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed}] {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}
