//! Smooth FITS images and cubes on disk to a common beam: the work behind
//! `convolvers 2d` and `convolvers 3d`, for use from Rust and Python.
//!
//! Each call runs on its own thread pool when `ncores` is set, so callers
//! with their own threads are not affected.
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rayon::prelude::*;
use tracing::{Level, debug, info, warn};

use crate::beam::Beam;
use crate::common_beam::{common_beam, fits_in_beam};
use crate::convolve_uv::{DEFAULT_NAN_FRACTION, FftPlans, padded_size};
use crate::cube_io::{
    self, CubeElem, CubeMeta, CubeMode, CubeReader, CubeWriter, PixelType, RefChan,
};
use crate::fits_io::{FitsImageHeader, output_path, read_fits_header, read_fits_plane, write_fits};
use crate::smooth::smooth_with;

/// Options shared by [`smooth_images`] and [`smooth_cubes`].
#[derive(Debug, Clone)]
pub struct SmoothOptions {
    /// Beam to smooth to. Default: the common beam of the inputs.
    pub target_beam: Option<Beam>,
    /// Make the target circular (BMIN = BMAJ, BPA = 0).
    pub circularise: bool,
    /// Blank inputs whose BMAJ is larger than this, in arcsec.
    pub cutoff: Option<f64>,
    /// Output file name suffix.
    pub suffix: String,
    /// Output file name prefix.
    pub prefix: Option<String>,
    /// Output directory. Default: beside each input.
    pub outdir: Option<PathBuf>,
    /// Tolerance of the minimum-volume-ellipse common-beam fit.
    pub tolerance: f64,
    /// Edge samples per beam in the common-beam fit.
    pub nsamps: usize,
    /// Edge inflation in the common-beam fit.
    pub epsilon: f64,
    /// Which pixels near the input's NaNs are blanked.
    pub nan_fraction: f64,
    /// Find the target beams but write nothing.
    pub dryrun: bool,
    /// Threads to use. Default: rayon's global pool.
    pub ncores: Option<usize>,
    /// Show progress bars on stderr.
    pub progress: bool,
}

impl Default for SmoothOptions {
    fn default() -> Self {
        Self {
            target_beam: None,
            circularise: false,
            cutoff: None,
            suffix: "sm".into(),
            prefix: None,
            outdir: None,
            tolerance: 1e-4,
            nsamps: 200,
            epsilon: 5e-4,
            nan_fraction: DEFAULT_NAN_FRACTION,
            dryrun: false,
            ncores: None,
            progress: false,
        }
    }
}

impl SmoothOptions {
    /// Where the smoothed copy of `file` is written.
    pub fn output_for(&self, file: &Path) -> PathBuf {
        output_path(
            file,
            Some(&self.suffix),
            self.prefix.as_deref(),
            self.outdir.as_deref(),
        )
    }

    /// The rounded common beam of `beams`, retrying with a tenth of the
    /// tolerance if the fit fails, as racs_tools does.
    fn common_beam(&self, beams: &[Beam]) -> Result<Beam> {
        let cb = common_beam(beams, self.tolerance, self.nsamps, self.epsilon).or_else(|e| {
            let tolerance = self.tolerance * 0.1;
            warn!("{e}; retrying the common beam with tolerance={tolerance:.1e}");
            common_beam(beams, tolerance, self.nsamps, self.epsilon)
        })?;
        round_common_beam(cb, beams)
    }
}

/// Options for [`smooth_cubes`].
#[derive(Debug, Clone, Default)]
pub struct CubeOptions {
    pub smooth: SmoothOptions,
    /// A common beam per channel, or one for the whole cube.
    pub mode: CubeMode,
    /// Channel whose beam goes in the output's BMAJ/BMIN/BPA.
    pub ref_chan: RefChan,
    /// Reuse each output's beamlog from a previous run as its target beams.
    pub uselogs: bool,
    /// A target beam per channel (`None` blanks the channel), in place of
    /// solving for them. Excludes `smooth.target_beam` and `uselogs`.
    pub target_beams: Option<Vec<Option<Beam>>>,
}

/// One image smoothed by [`smooth_images`].
#[derive(Debug, Clone)]
pub struct SmoothedImage {
    pub input: PathBuf,
    pub output: PathBuf,
    /// The input's beam.
    pub old_beam: Beam,
    /// The beam it was convolved with, or `None` if it was blanked.
    pub kernel: Option<Beam>,
}

/// The result of [`smooth_images`].
#[derive(Debug, Clone)]
pub struct SmoothedImages {
    /// The beam every usable image was smoothed to.
    pub target: Beam,
    /// One entry per input, in input order.
    pub images: Vec<SmoothedImage>,
}

/// One cube smoothed by [`smooth_cubes`].
#[derive(Debug, Clone)]
pub struct SmoothedCube {
    pub input: PathBuf,
    pub output: PathBuf,
    /// The target beam of each channel; `None` for a channel with no target.
    pub targets: Vec<Option<Beam>>,
}

/// Smooth 2D images to one common beam.
///
/// Images with no usable beam, or one over the cutoff, are written as NaN
/// with a zero beam.
pub fn smooth_images(files: &[PathBuf], opts: &SmoothOptions) -> Result<SmoothedImages> {
    in_pool(opts.ncores, || smooth_images_in_pool(files, opts))
}

fn smooth_images_in_pool(files: &[PathBuf], opts: &SmoothOptions) -> Result<SmoothedImages> {
    ensure!(!files.is_empty(), "no input files");
    let cutoff = opts.cutoff;
    let sp = spinner(
        opts.progress,
        format!("Reading headers of {} file(s)…", files.len()),
    );
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
    // With a target given, images that cannot reach it are blanked, as
    // racs_tools does, even when that is all of them.
    let target = match opts.target_beam {
        Some(b) => b,
        None => {
            ensure!(!valid.is_empty(), "all beams are flagged or invalid");
            let sp = spinner(opts.progress, "Solving for the common beam…");
            let cb = opts.common_beam(&valid)?;
            sp.finish_and_clear();
            cb
        }
    };
    let target = finish_target(target, &valid, opts.circularise)?;
    info!("Common beam: {target}");

    let kernel_of = |h: &FitsImageHeader| {
        usable_beam(&h.beam, cutoff).then(|| target.deconvolve_or_zero(&h.beam))
    };
    if opts.dryrun {
        info!("Dry run: no files written.");
        let images = headers
            .iter()
            .map(|h| SmoothedImage {
                input: h.path.clone(),
                output: opts.output_for(&h.path),
                old_beam: h.beam,
                kernel: kernel_of(h),
            })
            .collect();
        return Ok(SmoothedImages { target, images });
    }

    let pb = progress_bar(opts.progress, files.len() as u64);
    let process = |header: &FitsImageHeader| -> Result<SmoothedImage> {
        let output = opts.output_for(&header.path);
        let kernel = kernel_of(header);
        if let Some(k) = kernel {
            debug!(
                "{}: current {} | target {target} | kernel {k}",
                header.path.display(),
                header.beam
            );
        }
        match header.dtype {
            PixelType::F32 => smooth_image_as::<f32>(
                header,
                &output,
                &target,
                kernel.is_none(),
                opts.nan_fraction,
            ),
            PixelType::F64 => smooth_image_as::<f64>(
                header,
                &output,
                &target,
                kernel.is_none(),
                opts.nan_fraction,
            ),
        }?;
        pb.suspend(|| info!("{} → {}", header.path.display(), output.display()));
        pb.inc(1);
        Ok(SmoothedImage {
            input: header.path.clone(),
            output,
            old_beam: header.beam,
            kernel,
        })
    };
    // With fewer images than threads, smooth one at a time and let each
    // convolution use the whole pool.
    let images: Vec<SmoothedImage> = if headers.len() >= rayon::current_num_threads() {
        headers.par_iter().map(process).collect::<Result<_>>()?
    } else {
        headers.iter().map(process).collect::<Result<_>>()?
    };
    pb.finish_with_message("done");
    Ok(SmoothedImages { target, images })
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

/// Smooth one image to `target`, or write it as NaN with a zero beam.
fn smooth_image_as<T: CubeElem>(
    header: &FitsImageHeader,
    out: &Path,
    target: &Beam,
    blank: bool,
    nan_fraction: f64,
) -> Result<()> {
    let file = &header.path;
    if blank {
        let image = Array2::from_elem((header.ny, header.nx), T::nan());
        return write_fits(&image, out, file, &Beam::zero())
            .with_context(|| format!("writing {}", out.display()));
    }
    let data =
        read_fits_plane::<T>(header).with_context(|| format!("reading {}", file.display()))?;
    let image = smooth_with(
        data,
        &header.beam,
        target,
        header.pixel_scale,
        header.unit,
        nan_fraction,
        None,
    )
    .with_context(|| format!("smoothing {}", file.display()))?;
    write_fits(&image, out, file, target).with_context(|| format!("writing {}", out.display()))
}

/// Smooth spectral cubes, channel by channel, to common beams.
///
/// Every cube must have the same number of channels and at most one Stokes
/// plane. A channel with no target, no usable beam, or a beam over the cutoff
/// is written as NaN and masked in the output's BEAMS table. Each output gets
/// a beamlog of its target beams, which `uselogs` reads back.
pub fn smooth_cubes(files: &[PathBuf], opts: &CubeOptions) -> Result<Vec<SmoothedCube>> {
    in_pool(opts.smooth.ncores, || smooth_cubes_in_pool(files, opts))
}

fn smooth_cubes_in_pool(files: &[PathBuf], opts: &CubeOptions) -> Result<Vec<SmoothedCube>> {
    ensure!(!files.is_empty(), "no input files");
    let shared = &opts.smooth;
    ensure!(
        !(opts.uselogs && (shared.target_beam.is_some() || opts.target_beams.is_some())),
        "uselogs cannot be combined with a target beam"
    );
    ensure!(
        !(shared.target_beam.is_some() && opts.target_beams.is_some()),
        "give either one target beam or one per channel, not both"
    );
    let metas = read_cube_metas(files, shared.progress)?;
    let nfreq = metas[0].nfreq;

    // One set of targets per cube: shared, unless read back from each one's log.
    let per_cube: Vec<Vec<Option<Beam>>> = if opts.uselogs {
        metas
            .iter()
            .map(|m| read_previous_targets(&shared.output_for(&m.path), nfreq))
            .collect::<Result<_>>()?
    } else {
        vec![solve_targets(opts, &metas)?; metas.len()]
    };
    report_targets(&per_cube[0]);

    let cubes: Vec<SmoothedCube> = metas
        .iter()
        .zip(per_cube)
        .map(|(m, targets)| SmoothedCube {
            input: m.path.clone(),
            output: shared.output_for(&m.path),
            targets,
        })
        .collect();
    if shared.dryrun {
        info!("Dry run: no files written.");
        return Ok(cubes);
    }

    let pb = progress_bar(shared.progress, (files.len() * nfreq) as u64);
    for (meta, cube) in metas.iter().zip(&cubes) {
        let out = &cube.output;
        pb.suspend(|| info!("Initialising output cube {} …", out.display()));
        pb.set_message("processing");
        let job = CubeJob {
            out,
            meta,
            targets: &cube.targets,
            mode: opts.mode,
            ref_chan: opts.ref_chan,
            cutoff: shared.cutoff,
            nan_fraction: shared.nan_fraction,
            pb: &pb,
        };
        match meta.dtype {
            PixelType::F32 => job.run::<f32>()?,
            PixelType::F64 => job.run::<f64>()?,
        }
        let beamlog = cube_io::beamlog_path(out);
        cube_io::write_beamlog(&beamlog, &cube.targets)
            .with_context(|| format!("writing beamlog {}", beamlog.display()))?;
        pb.suspend(|| info!("{} → {}", meta.path.display(), out.display()));
    }
    pb.finish_with_message("done");
    Ok(cubes)
}

/// The beam of each channel of a cube, `None` where it is masked.
pub fn cube_beams(path: &Path) -> Result<Vec<Option<Beam>>> {
    let meta = cube_io::read_cube_meta(path)
        .with_context(|| format!("reading metadata from {}", path.display()))?;
    Ok(meta.beams)
}

/// The target beams a previous run wrote to the beamlog of output `out`.
fn read_previous_targets(out: &Path, nfreq: usize) -> Result<Vec<Option<Beam>>> {
    let log = cube_io::beamlog_path(out);
    let beams = cube_io::read_beamlog(&log)
        .with_context(|| format!("reading the previous run's beamlog {}", log.display()))?;
    ensure!(
        beams.len() == nfreq,
        "{}: {} channels, but the cube has {nfreq}",
        log.display(),
        beams.len()
    );
    info!("Target beams read from {}", log.display());
    Ok(beams
        .into_iter()
        .map(|b| (!b.is_zero()).then_some(b))
        .collect())
}

/// The per-channel target beams: those given, or the common beams.
fn solve_targets(opts: &CubeOptions, metas: &[CubeMeta]) -> Result<Vec<Option<Beam>>> {
    let shared = &opts.smooth;
    let nfreq = metas[0].nfreq;
    let usable = |c: usize| {
        metas
            .iter()
            .filter_map(move |m| m.beams[c])
            .filter(|b| usable_beam(b, shared.cutoff))
    };
    if let Some(targets) = &opts.target_beams {
        ensure!(
            targets.len() == nfreq,
            "{} target beams for cubes of {nfreq} channels",
            targets.len()
        );
        return targets
            .iter()
            .enumerate()
            .map(|(c, t)| {
                t.map(|t| finish_target(t, &usable(c).collect::<Vec<_>>(), shared.circularise))
                    .transpose()
                    .with_context(|| format!("channel {c}"))
            })
            .collect();
    }
    if let Some(b) = shared.target_beam {
        let valid: Vec<Beam> = (0..nfreq).flat_map(usable).collect();
        return Ok(vec![
            Some(finish_target(b, &valid, shared.circularise)?);
            nfreq
        ]);
    }

    let sp = spinner(
        shared.progress,
        match opts.mode {
            CubeMode::Natural => "Solving for per-channel common beams…",
            CubeMode::Total => "Solving for the common beam across all channels…",
        },
    );
    let targets = match opts.mode {
        CubeMode::Natural => (0..nfreq)
            .map(|c| {
                let valid: Vec<Beam> = usable(c).collect();
                if valid.is_empty() {
                    return Ok(None);
                }
                let cb = shared
                    .common_beam(&valid)
                    .with_context(|| format!("finding common beam for channel {c}"))?;
                Ok(Some(finish_target(cb, &valid, shared.circularise)?))
            })
            .collect::<Result<_>>()?,
        CubeMode::Total => {
            let valid: Vec<Beam> = (0..nfreq).flat_map(usable).collect();
            ensure!(
                !valid.is_empty(),
                "no valid beams found across all cubes/channels"
            );
            let cb = shared
                .common_beam(&valid)
                .context("finding total common beam")?;
            vec![Some(finish_target(cb, &valid, shared.circularise)?); nfreq]
        }
    };
    sp.finish_and_clear();
    Ok(targets)
}

/// Read every cube's metadata and check they can be smoothed together.
fn read_cube_metas(files: &[PathBuf], progress: bool) -> Result<Vec<CubeMeta>> {
    let sp = spinner(
        progress,
        format!("Reading metadata from {} cube(s)…", files.len()),
    );
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
        ensure!(
            m.nfreq == nfreq,
            "{}: expected {nfreq} channels, got {}",
            m.path.display(),
            m.nfreq
        );
        // Only the first Stokes plane is convolved, so the others would be
        // written as zeros.
        ensure!(
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

/// Log the target beams.
fn report_targets(target_beams: &[Option<Beam>]) {
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
}

/// One output cube to write.
struct CubeJob<'a> {
    out: &'a Path,
    meta: &'a CubeMeta,
    targets: &'a [Option<Beam>],
    mode: CubeMode,
    ref_chan: RefChan,
    cutoff: Option<f64>,
    nan_fraction: f64,
    pb: &'a ProgressBar,
}

impl CubeJob<'_> {
    /// Whether channel `c` is convolved rather than blanked.
    fn convolves(&self, c: usize) -> bool {
        match (self.meta.beams[c], self.targets[c]) {
            (Some(old), Some(target)) => {
                usable_beam(&old, self.cutoff) && usable_beam(&target, None)
            }
            _ => false,
        }
    }

    /// Smooth every channel at pixel precision `T`.
    ///
    /// rayon workers convolve channels and send them to one writer thread,
    /// which owns the output (cfitsio handles are not `Send`). The bounded
    /// channel caps memory at the planes in flight.
    fn run<T: CubeElem>(&self) -> Result<()> {
        let (meta, out) = (self.meta, self.out);
        // Blanked channels are masked in the BEAMS table, not given the target.
        let written: Vec<Option<Beam>> = (0..meta.nfreq)
            .map(|c| self.targets[c].filter(|_| self.convolves(c)))
            .collect();
        // One set of plans, large enough for the widest kernel of any channel.
        let (prows, pcols) = (0..meta.nfreq)
            .filter(|&c| self.convolves(c))
            .filter_map(|c| Some((meta.beams[c]?, self.targets[c]?)))
            .map(|(old, new)| padded_size((meta.ny, meta.nx), &old, &new, meta.pixel_scale))
            .fold((1, 1), |(r, c), (pr, pc)| (r.max(pr), c.max(pc)));
        let plans = FftPlans::<T>::new(prows, pcols);
        let cap = (rayon::current_num_threads() * 2).max(4);
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Array2<T>)>(cap);

        std::thread::scope(|s| {
            let written = &written;
            let writer = s.spawn(move || -> Result<()> {
                let mut writer =
                    CubeWriter::create(&meta.path, out, written, self.mode, meta, self.ref_chan)
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
                    let plane = self.smooth_channel::<T>(reader, c, &plans)?;
                    // A send error means the writer stopped; its error is reported below.
                    tx.send((c, plane))
                        .map_err(|_| anyhow::anyhow!("writer thread stopped before channel {c}"))?;
                    self.pb.inc(1);
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

    /// Channel `c` smoothed to its target, or NaN.
    fn smooth_channel<T: CubeElem>(
        &self,
        reader: &mut CubeReader,
        c: usize,
        plans: &FftPlans<T>,
    ) -> Result<Array2<T>> {
        let meta = self.meta;
        if !self.convolves(c) {
            if let Some(old) = meta.beams[c]
                && usable_beam(&old, None)
                && !usable_beam(&old, self.cutoff)
            {
                self.pb.suspend(|| {
                    warn!(
                        "Channel {c}: BMAJ={:.1}\" > cutoff; blanking",
                        old.major_arcsec()
                    )
                });
            }
            return Ok(Array2::from_elem((meta.ny, meta.nx), T::nan()));
        }
        let (Some(old_beam), Some(target)) = (meta.beams[c], self.targets[c]) else {
            unreachable!("convolves() checked both beams");
        };
        if tracing::enabled!(Level::DEBUG) {
            let kernel = target.deconvolve_or_zero(&old_beam);
            self.pb.suspend(|| {
                debug!("Channel {c}: current {old_beam} | target {target} | kernel {kernel}")
            });
        }
        let raw = reader
            .read_channel::<T>(c, meta)
            .with_context(|| format!("reading channel {c} from {}", meta.path.display()))?;
        smooth_with(
            raw,
            &old_beam,
            &target,
            meta.pixel_scale,
            meta.unit,
            self.nan_fraction,
            Some(plans),
        )
        .with_context(|| format!("smoothing channel {c}"))
    }
}

/// True for a beam that can be convolved and is within the cutoff (arcsec).
pub fn usable_beam(b: &Beam, cutoff: Option<f64>) -> bool {
    b.has_area() && cutoff.is_none_or(|c| b.major_arcsec() <= c)
}

/// Round a common beam up to 0.1″ axes and 0.01° PA, as racs_tools does. If
/// the rounded PA leaves an input outside, add another 0.1″.
pub fn round_common_beam(b: Beam, inputs: &[Beam]) -> Result<Beam> {
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

/// Run `f` on a pool of `ncores` threads, or on rayon's global pool.
fn in_pool<R: Send>(ncores: Option<usize>, f: impl FnOnce() -> Result<R> + Send) -> Result<R> {
    let Some(n) = ncores else {
        return f();
    };
    ensure!(n > 0, "ncores must be at least 1");
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .context("setting up the thread pool")?
        .install(f)
}

fn progress_bar(show: bool, total: u64) -> ProgressBar {
    if !show {
        return ProgressBar::hidden();
    }
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
fn spinner(show: bool, msg: impl Into<String>) -> ProgressBar {
    if !show {
        return ProgressBar::hidden();
    }
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

#[cfg(test)]
mod tests {
    /// A channel of flint's test cubes that the fit only solves at a tenth of
    /// the default tolerance.
    #[test]
    fn test_common_beam_retries_with_a_smaller_tolerance() {
        let beams: Vec<Beam> = [
            (14.4205, 12.0973, 75.199),
            (14.4108, 12.1006, 75.011),
            (14.4212, 12.1285, 74.9),
            (13.741, 12.035, 72.886),
            (14.3404, 12.1605, 75.903),
            (14.4298, 12.1451, 74.711),
            (14.408, 12.0558, 75.231),
            (14.4111, 11.9223, 75.674),
            (14.1325, 12.0036, 73.702),
            (14.0073, 11.9958, 74.489),
        ]
        .iter()
        .map(|&(a, b, pa)| Beam::from_arcsec(a, b, pa).unwrap())
        .collect();
        let opts = SmoothOptions::default();
        assert!(common_beam(&beams, opts.tolerance, opts.nsamps, opts.epsilon).is_err());
        let cb = opts.common_beam(&beams).unwrap();
        assert!(beams.iter().all(|b| cb.contains(b)));
    }

    use super::*;

    #[test]
    fn ceil_to_ignores_round_off() {
        assert_eq!(ceil_to(30.500000000000004, 1), 30.5);
        assert_eq!(ceil_to(30.51, 1), 30.6);
    }

    #[test]
    fn rounded_beam_still_holds_inputs() {
        let inputs = [
            Beam::from_arcsec(10.03, 8.0, 30.0).unwrap(),
            Beam::from_arcsec(9.0, 8.5, 75.0).unwrap(),
        ];
        let cb = common_beam(&inputs, 1e-4, 200, 5e-4).unwrap();
        let rounded = round_common_beam(cb, &inputs).unwrap();
        assert!(fits_in_beam(&inputs, &rounded));
        assert!(rounded.major_arcsec() >= cb.major_arcsec());
    }

    #[test]
    fn zero_ncores_is_an_error() {
        assert!(in_pool(Some(0), || Ok(())).is_err());
        assert_eq!(
            in_pool(Some(2), || Ok(rayon::current_num_threads())).unwrap(),
            2
        );
    }
}
