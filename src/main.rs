use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tracing::{info, warn};

use convolve_rs::{
    beam::Beam,
    common_beam::{common_beam, fits_in_beam},
    fits_io::{output_path, read_fits, write_fits},
    smooth::smooth,
};

#[derive(Parser, Debug)]
#[command(
    name = "beamcon_2d",
    about = "Smooth 2D FITS images to a common beam resolution",
    long_about = None,
)]
struct Cli {
    /// Input FITS image(s).
    #[arg(required = true, num_args = 1..)]
    infile: Vec<PathBuf>,

    /// Treat a single infile as a text file listing one path per line.
    #[arg(long)]
    listfile: bool,

    /// Output filename suffix [default: sm].
    #[arg(short, long, default_value = "sm")]
    suffix: String,

    /// Output filename prefix.
    #[arg(short, long)]
    prefix: Option<String>,

    /// Output directory [default: same as input].
    #[arg(short, long)]
    outdir: Option<PathBuf>,

    /// Target BMAJ in arcsec (must also set --bmin and --bpa).
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

    /// Beam size cutoff in arcsec — blank images with BMAJ larger than this.
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

    /// Path to write a beamlog CSV.
    #[arg(long)]
    log: Option<PathBuf>,

    /// Verbose output (-v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let level = match cli.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        _ => tracing::Level::DEBUG,
    };
    tracing_subscriber::fmt().with_max_level(level).with_target(false).init();

    // Collect files.
    let files: Vec<PathBuf> = if cli.listfile {
        anyhow::ensure!(cli.infile.len() == 1, "only one listfile argument supported");
        std::fs::read_to_string(&cli.infile[0])?
            .lines()
            .map(|l| PathBuf::from(l.trim()))
            .collect()
    } else {
        cli.infile.clone()
    };
    anyhow::ensure!(!files.is_empty(), "no input files found");

    // Validate target beam args.
    let target_beam = match (cli.bmaj, cli.bmin, cli.bpa) {
        (None, None, None) => None,
        (Some(bmaj), Some(bmin), Some(bpa)) => {
            Some(Beam::from_arcsec(bmaj, bmin, bpa).context("invalid target beam")?)
        }
        _ => bail!("--bmaj, --bmin, and --bpa must all be specified together"),
    };

    // Read beams from all FITS headers.
    info!("Reading beam parameters from {} files", files.len());
    let all_beams: Vec<Beam> = files
        .iter()
        .map(|f| {
            let data = read_fits(f).with_context(|| format!("reading {}", f.display()))?;
            if let Some(cutoff) = cli.cutoff {
                if data.beam.major_arcsec() > cutoff {
                    warn!("{}: BMAJ={:.1}\" > cutoff={:.1}\" — will be blanked",
                          f.display(), data.beam.major_arcsec(), cutoff);
                }
            }
            Ok(data.beam)
        })
        .collect::<Result<Vec<_>>>()?;

    // Determine common beam.
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
                        && cli.cutoff.map_or(true, |c| b.major_arcsec() <= c)
                })
                .cloned()
                .collect();
            anyhow::ensure!(!valid.is_empty(), "all beams are flagged or invalid");
            common_beam(&valid, cli.tolerance, cli.nsamps, cli.epsilon)
                .context("could not find common beam")?
        }
    };

    // Round up to 0.1 arcsec precision (matches Python beamcon_2D).
    common = Beam::from_arcsec(
        ceil_to(common.major_arcsec(), 1),
        ceil_to(common.minor_arcsec(), 1),
        round_up(common.pa_deg, 2),
    )
    .context("rounding common beam")?;

    if cli.circularise {
        common = Beam::from_arcsec(common.major_arcsec(), common.major_arcsec(), 0.0)
            .context("circularising beam")?;
    }

    info!("Common beam: {common}");
    println!("Common beam: {common}");

    if cli.dryrun {
        println!("Dry run — no files written.");
        return Ok(());
    }

    // Process files in parallel.
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let results: Vec<BeamLogEntry> = files
        .par_iter()
        .zip(all_beams.par_iter())
        .map(|(file, old_beam)| {
            let entry = process_one(
                file,
                old_beam,
                &common,
                cli.suffix.as_str(),
                cli.prefix.as_deref(),
                cli.outdir.as_deref(),
                cli.cutoff,
            );
            pb.inc(1);
            entry
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_with_message("done");

    if let Some(log_path) = &cli.log {
        write_beamlog(&results, log_path)?;
        info!("Beamlog written to {}", log_path.display());
    }

    Ok(())
}

fn process_one(
    file: &Path,
    old_beam: &Beam,
    new_beam: &Beam,
    suffix: &str,
    prefix: Option<&str>,
    outdir: Option<&Path>,
    cutoff: Option<f64>,
) -> Result<BeamLogEntry> {
    let data = read_fits(file).with_context(|| format!("reading {}", file.display()))?;
    let out = output_path(file, Some(suffix), prefix, outdir);

    let conv_beam = new_beam.deconvolve_or_zero(old_beam);

    let smoothed = smooth(&data.image, old_beam, new_beam, data.dx_deg, data.dy_deg, cutoff)
        .with_context(|| format!("smoothing {}", file.display()))?;

    write_fits(&smoothed, &out, file, new_beam, data.is_4d)
        .with_context(|| format!("writing {}", out.display()))?;

    info!("{} → {}", file.display(), out.display());

    Ok(BeamLogEntry {
        filename: out,
        old_beam: *old_beam,
        new_beam: *new_beam,
        conv_beam,
    })
}

struct BeamLogEntry {
    filename: PathBuf,
    old_beam: Beam,
    new_beam: Beam,
    conv_beam: Beam,
}

fn write_beamlog(entries: &[BeamLogEntry], path: &Path) -> Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(out, "# FileName OldBMAJ[deg] OldBMIN[deg] OldBPA[deg] TargetBMAJ[deg] TargetBMIN[deg] TargetBPA[deg] ConvBMAJ[deg] ConvBMIN[deg] ConvBPA[deg]")?;
    for e in entries {
        writeln!(out,
            "{} {} {} {} {} {} {} {} {} {}",
            e.filename.display(),
            e.old_beam.major_deg, e.old_beam.minor_deg, e.old_beam.pa_deg,
            e.new_beam.major_deg, e.new_beam.minor_deg, e.new_beam.pa_deg,
            e.conv_beam.major_deg, e.conv_beam.minor_deg, e.conv_beam.pa_deg,
        )?;
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Ceiling to `precision` decimal places (matches Python my_ceil).
fn ceil_to(x: f64, precision: i32) -> f64 {
    let factor = 10_f64.powi(precision);
    (x * factor).ceil() / factor
}

/// Round up to `decimals` decimal places.
fn round_up(x: f64, decimals: i32) -> f64 {
    let factor = 10_f64.powi(decimals);
    (x * factor).ceil() / factor
}
