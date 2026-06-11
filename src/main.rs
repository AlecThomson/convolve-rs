use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use convolve_rs::{
    beam::Beam,
    common_beam::{common_beam, fits_in_beam},
    cube_io::{self, CubeMeta, CubeMode},
    fits_io::{output_path, read_fits, write_fits},
    smooth::smooth,
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

    info!("Reading beam parameters from {} files", files.len());
    let all_beams: Vec<Beam> = files
        .iter()
        .map(|f| {
            let data = read_fits(f).with_context(|| format!("reading {}", f.display()))?;
            if let Some(cutoff) = args.shared.cutoff
                && data.beam.major_arcsec() > cutoff
            {
                warn!(
                    "{}: BMAJ={:.1}\" > cutoff={:.1}\" — will be blanked",
                    f.display(),
                    data.beam.major_arcsec(),
                    cutoff
                );
            }
            Ok(data.beam)
        })
        .collect::<Result<Vec<_>>>()?;

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
            common_beam(
                &valid,
                args.shared.tolerance,
                args.shared.nsamps,
                args.shared.epsilon,
            )
            .context("could not find common beam")?
        }
    };

    common = apply_beam_rounding(common, args.shared.circularise)?;

    info!("Common beam: {common}");
    println!("Common beam: {common}");

    if args.shared.dryrun {
        println!("Dry run — no files written.");
        return Ok(());
    }

    let pb = progress_bar(files.len() as u64);

    let results: Vec<BeamLogEntry2D> = files
        .par_iter()
        .zip(all_beams.par_iter())
        .map(|(file, old_beam)| {
            let data = read_fits(file).with_context(|| format!("reading {}", file.display()))?;
            let out = output_path(
                file,
                Some(&args.shared.suffix),
                args.shared.prefix.as_deref(),
                args.shared.outdir.as_deref(),
            );
            let conv_beam = common.deconvolve_or_zero(old_beam);
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
            write_fits(&smoothed, &out, file, &common, data.is_4d)
                .with_context(|| format!("writing {}", out.display()))?;
            info!("{} → {}", file.display(), out.display());
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

    info!("Reading cube metadata from {} file(s)", files.len());
    let metas: Vec<CubeMeta> = files
        .iter()
        .map(|f| {
            cube_io::read_cube_meta(f)
                .with_context(|| format!("reading metadata from {}", f.display()))
        })
        .collect::<Result<_>>()?;

    let nfreq = metas[0].nfreq;
    for (f, m) in files.iter().zip(metas.iter()) {
        anyhow::ensure!(
            m.nfreq == nfreq,
            "{}: expected {} channels, got {}",
            f.display(),
            nfreq,
            m.nfreq
        );
        if m.nstokes > 1 {
            warn!(
                "{}: NAXIS4={} — only Stokes 0 will be convolved",
                f.display(),
                m.nstokes
            );
        }
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
        compute_target_beams(
            &metas,
            mode,
            args.shared.cutoff,
            args.shared.circularise,
            args.shared.tolerance,
            args.shared.nsamps,
            args.shared.epsilon,
        )?
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
        Some(b) if all_same => println!("Target beam (all channels): {b}"),
        Some(b) => {
            println!(
                "Target beam varies per channel ({n_valid} valid channels); e.g. channel 0: {b}"
            );
            println!("Run with -vv to log the current/target/kernel beam for every channel.");
        }
        None => {}
    }

    if args.shared.dryrun {
        println!("Dry run — no files written.");
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

        cube_io::init_output_cube(file, &out, &target_beams, cube_mode, meta)
            .with_context(|| format!("initialising output cube {}", out.display()))?;

        // Process channels in parallel, write sequentially.
        let channel_results: Vec<Option<Array2<f32>>> = (0..nfreq)
            .into_par_iter()
            .map(|c| -> Result<Option<Array2<f32>>> {
                let old_beam = match meta.beams[c] {
                    Some(b) => b,
                    None => {
                        pb.inc(1);
                        return Ok(None);
                    }
                };
                let target = match target_beams[c] {
                    Some(b) => b,
                    None => {
                        pb.inc(1);
                        return Ok(None);
                    }
                };

                // Verbose (-vv) per-channel beam report: current, target, and the
                // convolving kernel (target deconvolved from the current beam).
                let kernel = target.deconvolve_or_zero(&old_beam);
                debug!("Channel {c}: current {old_beam} | target {target} | kernel {kernel}");

                if let Some(cutoff) = args.shared.cutoff
                    && old_beam.major_arcsec() > cutoff
                {
                    warn!(
                        "Channel {c}: BMAJ={:.1}\" > cutoff — blanking",
                        old_beam.major_arcsec()
                    );
                    pb.inc(1);
                    return Ok(Some(Array2::from_elem((meta.ny, meta.nx), f32::NAN)));
                }

                let plane = cube_io::read_channel(file, c, meta)
                    .with_context(|| format!("reading channel {c} from {}", file.display()))?;
                let smoothed = smooth(
                    &plane,
                    &old_beam,
                    &target,
                    meta.dx_deg,
                    meta.dy_deg,
                    args.shared.cutoff,
                    meta.unit,
                )
                .with_context(|| format!("smoothing channel {c}"))?;
                pb.inc(1);
                Ok(Some(smoothed))
            })
            .collect::<Result<_>>()?;

        for (c, maybe_plane) in channel_results.into_iter().enumerate() {
            if let Some(plane) = maybe_plane {
                cube_io::write_channel(&out, c, &plane, meta)
                    .with_context(|| format!("writing channel {c} to {}", out.display()))?;
            }
        }

        let beamlog = {
            let dir = out.parent().unwrap_or(Path::new("."));
            let stem = out.file_stem().unwrap_or_default();
            dir.join(format!("beamlog.{}.txt", stem.to_string_lossy()))
        };
        cube_io::write_beamlog(&beamlog, &target_beams)
            .with_context(|| format!("writing beamlog {}", beamlog.display()))?;

        info!("{} → {}", file.display(), out.display());
    }

    pb.finish_with_message("done");
    Ok(())
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
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        _ => tracing::Level::DEBUG,
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
