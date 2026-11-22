use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing::{Level, info};

use convolve_rs::{
    beam::Beam,
    cube_io::{CubeMode, RefChan},
    pipeline::{CubeOptions, SmoothOptions, SmoothedImage, smooth_cubes, smooth_images},
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

    /// Blank pixels where more than this fraction of the kernel's weight fell
    /// on missing data (NaN pixels, or the sky beyond the image).
    #[arg(long, value_parser = parse_fraction, default_value_t = convolve_rs::DEFAULT_NAN_FRACTION)]
    nan_fraction: f64,

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

    /// Threads to use [default: all cores, or RAYON_NUM_THREADS].
    #[arg(short = 'n', long, value_parser = clap::value_parser!(u16).range(1..))]
    ncores: Option<u16>,
}

fn parse_fraction(s: &str) -> Result<f64, String> {
    let f: f64 = s.parse().map_err(|e| format!("{e}"))?;
    (0.0..=1.0)
        .contains(&f)
        .then_some(f)
        .ok_or_else(|| format!("{f} is not between 0 and 1"))
}

impl SharedArgs {
    /// Start logging and gather the options.
    fn init(&self) -> Result<SmoothOptions> {
        init_logging(self.verbose);
        Ok(SmoothOptions {
            target_beam: parse_target_beam(self)?,
            circularise: self.circularise,
            cutoff: self.cutoff,
            suffix: self.suffix.clone(),
            prefix: self.prefix.clone(),
            outdir: self.outdir.clone(),
            tolerance: self.tolerance,
            nsamps: self.nsamps,
            epsilon: self.epsilon,
            nan_fraction: self.nan_fraction,
            dryrun: self.dryrun,
            ncores: self.ncores.map(usize::from),
            progress: true,
        })
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

fn cmd_2d(args: TwoDArgs) -> Result<()> {
    let opts = args.shared.init()?;
    let files = collect_files(&args.infile, args.listfile)?;
    let result = smooth_images(&files, &opts)?;
    if opts.dryrun {
        // stdout, so callers can capture it; logs go to stderr.
        println!("{}", result.target);
        return Ok(());
    }
    if let Some(log_path) = &args.log {
        write_2d_beamlog(log_path, &result.target, &result.images)?;
        info!("Beamlog written to {}", log_path.display());
    }
    Ok(())
}

fn write_2d_beamlog(path: &Path, target: &Beam, entries: &[SmoothedImage]) -> Result<()> {
    let deg = |b: &Beam| format!("{} {} {}", b.major_deg(), b.minor_deg(), b.pa_deg());
    let mut out = String::from(
        "# FileName OldBMAJ[deg] OldBMIN[deg] OldBPA[deg] TargetBMAJ[deg] TargetBMIN[deg] \
         TargetBPA[deg] ConvBMAJ[deg] ConvBMIN[deg] ConvBPA[deg]\n",
    );
    for e in entries {
        let conv = e.kernel.as_ref().map_or("nan nan nan".into(), deg);
        out += &format!(
            "{} {} {} {conv}\n",
            e.output.display(),
            deg(&e.old_beam),
            deg(target)
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

    /// Channel whose beam goes in the output header's BMAJ/BMIN/BPA
    /// [default: the spectral axis' CRPIX].
    #[arg(long, value_enum)]
    ref_chan: Option<RefChanArg>,

    /// Reuse the target beams of a previous run, read from each output cube's
    /// beamlog, instead of solving for them.
    #[arg(long, conflicts_with_all = ["bmaj", "bmin", "bpa"])]
    uselogs: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RefChanArg {
    /// The first channel.
    First,
    /// The last channel.
    Last,
    /// The middle channel (the upper middle for an even count).
    Mid,
}

impl From<RefChanArg> for RefChan {
    fn from(r: RefChanArg) -> Self {
        match r {
            RefChanArg::First => RefChan::First,
            RefChanArg::Last => RefChan::Last,
            RefChanArg::Mid => RefChan::Mid,
        }
    }
}

fn cmd_3d(args: ThreeDArgs) -> Result<()> {
    let opts = CubeOptions {
        smooth: args.shared.init()?,
        mode: args.mode.into(),
        ref_chan: args.ref_chan.map_or(RefChan::default(), RefChan::from),
        uselogs: args.uselogs,
        target_beams: None,
    };
    let files = collect_files(&args.infile, args.listfile)?;
    let cubes = smooth_cubes(&files, &opts)?;
    if opts.smooth.dryrun {
        // stdout, so callers can capture it: one beam if all channels share
        // it, else one `<channel> <beam>` line per channel.
        let targets = &cubes[0].targets;
        let valid: Vec<Beam> = targets.iter().flatten().copied().collect();
        let all_same = valid.windows(2).all(|w| w[0].approx_eq(&w[1]));
        match valid.first() {
            Some(b) if all_same => println!("{b}"),
            _ => {
                for (c, b) in targets.iter().enumerate() {
                    match b {
                        Some(b) => println!("{c} {b}"),
                        None => println!("{c} masked"),
                    }
                }
            }
        }
    }
    Ok(())
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
