//! FFT-based UV-plane convolution of an image from one Gaussian beam to
//! another, following the "robust" mode of `racs_tools.convolve_uv`.
//!
//! The filter is evaluated analytically at each UV point instead of being
//! stored, and the transforms run in the image's own precision ([`FftFloat`]).
//! Peak memory is about one image on top of the input, and none for an owned,
//! NaN-free plane.
use std::sync::Arc;

use ndarray::{Array2, ArrayView2, CowArray, Ix2};
use num_traits::{Float, cast};
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftNum, FftPlanner, num_complex::Complex};
use thiserror::Error;

use crate::beam::Beam;

#[derive(Debug, Error)]
pub enum ConvolveError {
    #[error("input beam {0} has zero size")]
    ZeroSizeBeam(Beam),
    #[error("target beam {new} cannot be reached by convolving {old}")]
    Unreachable { old: Beam, new: Beam },
    #[error("FFT plans are for {plans:?} images but the image is {image:?}")]
    PlanSize {
        plans: (usize, usize),
        image: (usize, usize),
    },
    #[error("FFT failed: {0}")]
    Fft(String),
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Element type a convolution runs in: `f32` or `f64`.
pub trait FftFloat: FftNum + Float + sealed::Sealed {}
impl FftFloat for f32 {}
impl FftFloat for f64 {}

/// Cast to `T`, saturating to ±∞ outside its range, so an f64 value too large
/// for f32 gives an infinite pixel rather than a panic.
pub(crate) fn cast_saturating<T: FftFloat>(value: f64) -> T {
    cast::<f64, T>(value).unwrap_or_else(|| {
        if value.is_sign_negative() {
            T::neg_infinity()
        } else {
            T::infinity()
        }
    })
}

pub struct ConvolutionResult<T = f32> {
    /// Convolved image, with NaNs where the input's NaNs spread to.
    pub image: Array2<T>,
    /// `√(Ω_new/Ω_old)`, the DC gain of the filter, already applied to `image`.
    pub scaling_factor: f64,
}

/// FFT plans for one image size, shareable across threads.
///
/// Planning is the expensive part of a small transform, so build these once
/// per cube and pass them to every channel.
pub struct FftPlans<T: FftNum = f32> {
    nrows: usize,
    ncols: usize,
    nhalf: usize,
    r2c: Arc<dyn RealToComplex<T>>,
    c2r: Arc<dyn ComplexToReal<T>>,
    col_fwd: Arc<dyn Fft<T>>,
    col_inv: Arc<dyn Fft<T>>,
}

impl<T: FftNum> FftPlans<T> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let mut rplanner = RealFftPlanner::<T>::new();
        let mut cplanner = FftPlanner::<T>::new();
        Self {
            nrows,
            ncols,
            nhalf: ncols / 2 + 1,
            r2c: rplanner.plan_fft_forward(ncols),
            c2r: rplanner.plan_fft_inverse(ncols),
            col_fwd: cplanner.plan_fft_forward(nrows),
            col_inv: cplanner.plan_fft_inverse(nrows),
        }
    }

    /// `(nrows, ncols)` these plans were built for.
    pub fn dim(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// How image pixels map to the sky.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelScale(Scale);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scale {
    Cdelt(f64, f64),
    Matrix([[f64; 2]; 2]),
}

impl PixelScale {
    /// Pixel sizes along x and y in degrees (CDELT1, CDELT2). Their signs
    /// are ignored, as racs_tools does, which suits the usual grid with east
    /// to the left.
    pub fn cdelt(dx_deg: f64, dy_deg: f64) -> Self {
        Self(Scale::Cdelt(dx_deg, dy_deg))
    }

    /// The pixel-to-sky matrix in degrees, as astropy's
    /// `WCS.pixel_scale_matrix` gives it: `m[i][j]` is the step along world
    /// axis `i` (east, north) per pixel along axis `j` (x, y). Signs,
    /// rotation and skew are kept. `None` if it is not finite or singular.
    ///
    /// ```
    /// use convolve_rs::PixelScale;
    ///
    /// let d = 2.5 / 3600.0;
    /// // The usual grid, east to the left, is the same as its CDELTs.
    /// assert_eq!(PixelScale::matrix([[-d, 0.0], [0.0, d]]), Some(PixelScale::cdelt(-d, d)));
    /// assert_eq!(PixelScale::matrix([[d, d], [d, d]]), None);
    /// ```
    pub fn matrix(m: [[f64; 2]; 2]) -> Option<Self> {
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if !(m.iter().flatten().all(|x| x.is_finite()) && det.is_normal()) {
            return None;
        }
        // Kept on the CDELT path so its output is unchanged to the bit.
        if m[0][1] == 0.0 && m[1][0] == 0.0 && m[0][0] < 0.0 && m[1][1] > 0.0 {
            return Some(Self::cdelt(m[0][0], m[1][1]));
        }
        Some(Self(Scale::Matrix(m)))
    }
}

/// The default `nan_fraction`: a convolved pixel is blanked when more than
/// half of its kernel's weight falls on missing data.
pub const DEFAULT_NAN_FRACTION: f64 = 0.5;

/// Convolve `image` from `old_beam` to `new_beam` in the UV plane.
///
/// `dx_deg` and `dy_deg` are the pixel sizes (CDELT1, CDELT2); their sign is
/// ignored. The convolution is linear, not circular: the image is padded, and
/// the sky beyond its edges, like its NaN pixels, is missing data. Each pixel
/// is divided by the share of its kernel's weight that fell on data, and
/// blanked where more than [`DEFAULT_NAN_FRACTION`] fell on missing data.
///
/// Pass `plans` to reuse FFT plans across images; they are used when exactly
/// [`padded_size`], and new ones are made otherwise, so the result never
/// depends on the plans passed.
///
/// # Errors
///
/// If `old_beam` has zero size or `new_beam` cannot be reached from it.
///
/// ```
/// use convolve_rs::{Beam, convolve_uv};
/// use ndarray::Array2;
///
/// let old = Beam::from_arcsec(10.0, 10.0, 0.0)?;
/// let new = Beam::from_arcsec(20.0, 20.0, 0.0)?;
/// let image = Array2::<f32>::from_elem((64, 64), 1.0);
/// let dx = 2.5 / 3600.0;
///
/// let result = convolve_uv(&image, &old, &new, dx, dx, None)?;
/// assert!((result.scaling_factor - 2.0).abs() < 1e-9);
/// assert_eq!(result.image.dim(), (64, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn convolve_uv<'a, T: FftFloat>(
    image: impl Into<CowArray<'a, T, Ix2>>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    plans: Option<&FftPlans<T>>,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    convolve_uv_with(
        image,
        old_beam,
        new_beam,
        PixelScale::cdelt(dx_deg, dy_deg),
        DEFAULT_NAN_FRACTION,
        plans,
    )
}

/// [`convolve_uv`] on any pixel grid, blanking pixels where more than
/// `nan_fraction` of the kernel's weight fell on missing data.
///
/// # Errors
///
/// As for [`convolve_uv`].
pub fn convolve_uv_with<'a, T: FftFloat>(
    image: impl Into<CowArray<'a, T, Ix2>>,
    old_beam: &Beam,
    new_beam: &Beam,
    scale: PixelScale,
    nan_fraction: f64,
    plans: Option<&FftPlans<T>>,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    convolve_with_gain(
        image.into(),
        old_beam,
        new_beam,
        scale,
        nan_fraction,
        plans,
        |ratio| ratio,
    )
}

/// The transform size [`convolve_uv`] pads an image of `dim` to, for
/// convolving `old_beam` to `new_beam`: each axis grown by the kernel's reach
/// and rounded up to a fast FFT length. Plans of this size are reused.
///
/// ```
/// use convolve_rs::{Beam, PixelScale, padded_size};
///
/// let old = Beam::from_arcsec(10.0, 10.0, 0.0)?;
/// let new = Beam::from_arcsec(20.0, 20.0, 0.0)?;
/// let (ny, nx) = padded_size((100, 120), &old, &new, PixelScale::cdelt(-2.5 / 3600.0, 2.5 / 3600.0));
/// assert!(ny > 100 && nx > 120);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn padded_size(
    (nrows, ncols): (usize, usize),
    old_beam: &Beam,
    new_beam: &Beam,
    scale: PixelScale,
) -> (usize, usize) {
    let (taper, steps) = taper_for(old_beam, new_beam, scale);
    let reach = kernel_reach(&taper, steps);
    (fast_len(nrows + reach), fast_len(ncols + reach))
}

/// The filter, and the row and column steps its frequencies are in.
fn taper_for(old_beam: &Beam, new_beam: &Beam, scale: PixelScale) -> (UvTaper, (f64, f64)) {
    match scale.0 {
        // Rows run along y (CDELT2), columns along x (CDELT1). Only the
        // magnitudes matter: passing a negative CDELT1 to fftfreq would mirror the PA.
        Scale::Cdelt(dx, dy) => (
            UvTaper::new(old_beam, new_beam),
            (dy.abs().to_radians(), dx.abs().to_radians()),
        ),
        Scale::Matrix(m) => (UvTaper::in_pixels(old_beam, new_beam, m), (1.0, 1.0)),
    }
}

/// The kernel's covariance in pixels, `(rows, columns, cross)`: its
/// transform is exp(-2π² fᵀ S f) at f cycles per pixel.
fn pixel_covariance(taper: &UvTaper, (row_step, col_step): (f64, f64)) -> (f64, f64, f64) {
    let k = -1.0 / (2.0 * std::f64::consts::PI.powi(2));
    (
        k * taper.a / row_step.powi(2),
        k * taper.c / col_step.powi(2),
        k * taper.b / 2.0 / (row_step * col_step),
    )
}

/// The kernel's reach in pixels: 6.1 standard deviations along its major
/// axis, plus one. Beyond it the kernel holds under 1e-8 of its sum.
fn kernel_reach(taper: &UvTaper, steps: (f64, f64)) -> usize {
    let (srr, scc, src) = pixel_covariance(taper, steps);
    let largest = (srr + scc) / 2.0 + (((srr - scc) / 2.0).powi(2) + src * src).sqrt();
    (6.1 * largest.max(0.0).sqrt()).ceil() as usize + 1
}

/// Whether the kernel's transform is above 1e-5 somewhere at the Nyquist
/// frequency. Such a kernel, under about 1.5 pixels of standard deviation,
/// rings beyond its reach, and differently in transforms of different sizes,
/// so its missing shares found in tiles would be off by more than that.
fn rings(taper: &UvTaper, steps: (f64, f64)) -> bool {
    let (srr, scc, src) = pixel_covariance(taper, steps);
    // The least of fᵀ S f along an edge of the band, one frequency at ±1/2.
    let edge = |s_fixed: f64, s_free: f64| {
        let free = if s_free > 0.0 {
            (-src / (2.0 * s_free)).clamp(-0.5, 0.5)
        } else {
            0.5
        };
        s_fixed / 4.0 + src * free + s_free * free * free
    };
    let least = edge(srr, scc).min(edge(scc, srr));
    (-2.0 * std::f64::consts::PI.powi(2) * least).exp() >= 1e-5
}

/// [`convolve_uv`] with the filter's DC gain set to `gain(g_ratio)`, where
/// `g_ratio = √(Ω_new/Ω_old)`. Folding the flux scaling into the filter saves
/// a pass over the image.
pub(crate) fn convolve_with_gain<T: FftFloat>(
    image: CowArray<'_, T, Ix2>,
    old_beam: &Beam,
    new_beam: &Beam,
    scale: PixelScale,
    nan_fraction: f64,
    plans: Option<&FftPlans<T>>,
    gain: impl Fn(f64) -> f64,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    if old_beam.approx_eq(new_beam) {
        return Ok(ConvolutionResult {
            image: image.into_owned(),
            scaling_factor: 1.0,
        });
    }
    if !old_beam.has_area() {
        return Err(ConvolveError::ZeroSizeBeam(*old_beam));
    }
    if !new_beam.contains(old_beam) {
        return Err(ConvolveError::Unreachable {
            old: *old_beam,
            new: *new_beam,
        });
    }

    let (taper, steps) = taper_for(old_beam, new_beam, scale);
    let nan_count = image.iter().filter(|x| x.is_nan()).count();
    if nan_count == image.len() {
        return Ok(ConvolutionResult {
            image: image.into_owned(),
            scaling_factor: taper.ratio,
        });
    }

    let (nrows, ncols) = image.dim();
    let reach = kernel_reach(&taper, steps);
    let needed = (fast_len(nrows + reach), fast_len(ncols + reach));
    let local_plans;
    let plans = match plans {
        // Padding more than needed changes the result slightly for a kernel
        // narrower than a pixel, so only plans of exactly this size are used.
        Some(p) if p.dim() == needed => p,
        _ => {
            local_plans = FftPlans::new(needed.0, needed.1);
            &local_plans
        }
    };
    let padded = plans.dim();
    let grid = UvGrid::new(padded, steps.0, steps.1, &taper);
    let missing = missing_fractions(
        image.view(),
        &taper,
        steps,
        reach,
        nan_count > 0,
        (&grid, plans),
    )?;

    let mut buf = padded_zero_filled(image.view(), plans);
    let npix = (padded.0 * padded.1) as f64;
    convolve_in_place(&mut buf, &taper, gain(taper.ratio) / npix, &grid, plans)?;
    for i in 1..nrows {
        buf.copy_within(i * padded.1..i * padded.1 + ncols, i * ncols);
    }
    buf.truncate(nrows * ncols);
    apply_missing(&mut buf, ncols, &missing, nan_fraction);

    Ok(ConvolutionResult {
        image: Array2::from_shape_vec((nrows, ncols), buf)
            .map_err(|e| ConvolveError::Fft(e.to_string()))?,
        scaling_factor: taper.ratio,
    })
}

/// Fractions of the kernel's weight below this are ignored: the kernel is cut at
/// the same level by its reach.
const MISSING_ROUND_OFF: f64 = 1e-8;

/// Pixels of `row`, columns `cols`, within the kernel's reach of missing
/// data, with the fraction of the kernel's weight that falls on it. No
/// `values` means all of it.
struct MissingSpan<T> {
    row: usize,
    cols: std::ops::Range<usize>,
    values: Vec<T>,
}

/// Where the kernel reaches missing data: NaN pixels and the sky beyond the
/// image. Away from it a pixel is left as it is, so only the part near it is
/// computed, tile by tile ([`Tiling`]), unless the kernel is large next to
/// the image or [`rings`].
fn missing_fractions<T: FftFloat>(
    image: ArrayView2<'_, T>,
    taper: &UvTaper,
    steps: (f64, f64),
    reach: usize,
    has_nans: bool,
    (grid, plans): (&UvGrid, &FftPlans<T>),
) -> Result<Vec<MissingSpan<T>>, ConvolveError> {
    match Tiling::new(image.dim(), reach).filter(|_| !rings(taper, steps)) {
        Some(tiling) => missing_tiled(image, taper, steps, &tiling, has_nans),
        None => missing_whole(image, taper, grid, plans),
    }
}

fn missing_whole<T: FftFloat>(
    image: ArrayView2<'_, T>,
    taper: &UvTaper,
    grid: &UvGrid,
    plans: &FftPlans<T>,
) -> Result<Vec<MissingSpan<T>>, ConvolveError> {
    let (nrows, ncols) = image.dim();
    let (prows, pcols) = plans.dim();
    let mut mask = Vec::with_capacity(spectrum_len(plans));
    for row in image.rows() {
        mask.extend(
            row.iter()
                .map(|x| if x.is_nan() { T::one() } else { T::zero() }),
        );
        mask.resize(mask.len() + pcols - ncols, T::one());
    }
    mask.resize(prows * pcols, T::one());
    convolve_in_place(&mut mask, taper, 1.0 / (prows * pcols) as f64, grid, plans)?;
    let round_off = cast_saturating::<T>(MISSING_ROUND_OFF);
    Ok((0..nrows)
        .filter_map(|r| {
            let values = &mask[r * pcols..r * pcols + ncols];
            values
                .iter()
                .any(|&m| m.abs() > round_off)
                .then(|| MissingSpan {
                    row: r,
                    cols: 0..ncols,
                    values: values.to_vec(),
                })
        })
        .collect())
}

/// Square tiles of `inner` pixels, each convolved with a halo of `halo` pixels
/// on every side in a `size`-pixel transform.
struct Tiling {
    inner: usize,
    halo: usize,
    size: usize,
}

impl Tiling {
    /// Tiles with a halo of the kernel's `reach`, or `None` if they would not
    /// be much smaller than the image.
    fn new((nrows, ncols): (usize, usize), reach: usize) -> Option<Self> {
        let size = fast_len((6 * reach).max(128));
        (8 * size <= nrows.min(ncols) * 2).then_some(Self {
            inner: size - 2 * reach,
            halo: reach,
            size,
        })
    }
}

/// The smallest `n' >= n` whose only prime factors are 2, 3 and 5.
fn fast_len(n: usize) -> usize {
    (n..)
        .find(|&m| {
            let mut m = m;
            for p in [2, 3, 5] {
                while m % p == 0 {
                    m /= p;
                }
            }
            m == 1
        })
        .unwrap_or(n)
}

/// [`missing_whole`], tile by tile. A tile with no missing data within the
/// halo is left alone, one with nothing else is all missing, and only those
/// in between are convolved.
fn missing_tiled<T: FftFloat>(
    image: ArrayView2<'_, T>,
    taper: &UvTaper,
    (row_step, col_step): (f64, f64),
    tiling: &Tiling,
    has_nans: bool,
) -> Result<Vec<MissingSpan<T>>, ConvolveError> {
    let (nrows, ncols) = image.dim();
    let Tiling { inner, halo, size } = *tiling;
    let plans = FftPlans::<T>::new(size, size);
    let grid = UvGrid::new((size, size), row_step, col_step, taper);
    let gain = 1.0 / (size * size) as f64;
    let round_off = cast_saturating::<T>(MISSING_ROUND_OFF);
    // Rows and columns of the image a tile's transform covers, from its
    // corner; None beyond the image.
    let span = |start: usize, n: usize| {
        (0..size).map(move |k| (start + k).checked_sub(halo).filter(|&i| i < n))
    };

    let tiles: Vec<(usize, usize)> = (0..nrows.div_ceil(inner))
        .flat_map(|ti| (0..ncols.div_ceil(inner)).map(move |tj| (ti * inner, tj * inner)))
        .collect();
    let spans = tiles
        .par_iter()
        .map(|&(r0, c0)| -> Result<Vec<MissingSpan<T>>, ConvolveError> {
            let (r1, c1) = ((r0 + inner).min(nrows), (c0 + inner).min(ncols));
            let rows: Vec<Option<usize>> = span(r0, nrows).collect();
            let cols: Vec<Option<usize>> = span(c0, ncols).collect();
            let inside = rows.iter().chain(&cols).all(Option::is_some);
            if inside && !has_nans {
                return Ok(Vec::new());
            }
            let is_missing = |r: Option<usize>, c: Option<usize>| match (r, c) {
                (Some(r), Some(c)) => image[(r, c)].is_nan(),
                _ => true,
            };
            let missing_count: usize = rows
                .iter()
                .map(|&r| cols.iter().filter(|&&c| is_missing(r, c)).count())
                .sum();
            if missing_count == 0 {
                return Ok(Vec::new());
            }
            if missing_count == size * size {
                return Ok((r0..r1)
                    .map(|row| MissingSpan {
                        row,
                        cols: c0..c1,
                        values: Vec::new(),
                    })
                    .collect());
            }
            let mut mask = Vec::with_capacity(size * 2 * plans.nhalf);
            for &r in &rows {
                mask.extend(cols.iter().map(|&c| {
                    if is_missing(r, c) {
                        T::one()
                    } else {
                        T::zero()
                    }
                }));
            }
            convolve_in_place(&mut mask, taper, gain, &grid, &plans)?;
            Ok((r0..r1)
                .filter_map(|r| {
                    let start = (r - r0 + halo) * size + halo;
                    let values = &mask[start..start + (c1 - c0)];
                    values
                        .iter()
                        .any(|&m| m.abs() > round_off)
                        .then(|| MissingSpan {
                            row: r,
                            cols: c0..c1,
                            values: values.to_vec(),
                        })
                })
                .collect())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(spans.into_iter().flatten().collect())
}

/// Divide each pixel of the `ncols`-wide image in `buf` by the share of its
/// kernel's weight that fell on data, or blank it where more than
/// `nan_fraction` fell on missing data.
fn apply_missing<T: FftFloat>(
    buf: &mut [T],
    ncols: usize,
    missing: &[MissingSpan<T>],
    nan_fraction: f64,
) {
    let round_off = cast_saturating::<T>(MISSING_ROUND_OFF);
    let limit = cast_saturating::<T>(nan_fraction.max(MISSING_ROUND_OFF));
    for span in missing {
        let row = &mut buf[span.row * ncols..(span.row + 1) * ncols];
        if span.values.is_empty() {
            row[span.cols.clone()].fill(T::nan());
            continue;
        }
        for (x, &m) in row[span.cols.clone()].iter_mut().zip(&span.values) {
            if m > limit {
                *x = T::nan();
            } else if m.abs() > round_off {
                // A kernel sampled coarsely rings, so near missing data its
                // share there can dip just below zero.
                *x = *x / (T::one() - m);
            }
        }
    }
}

fn spectrum_len<T: FftNum>(plans: &FftPlans<T>) -> usize {
    plans.nrows * 2 * plans.nhalf
}

/// The image row-major with NaNs set to zero, in the top-left corner of a
/// zeroed buffer the size of `plans`.
fn padded_zero_filled<T: FftFloat>(image: ArrayView2<'_, T>, plans: &FftPlans<T>) -> Vec<T> {
    let (prows, pcols) = plans.dim();
    let ncols = image.ncols();
    let mut buf = Vec::with_capacity(spectrum_len(plans));
    for row in image.rows() {
        buf.extend(row.iter().map(|&x| if x.is_nan() { T::zero() } else { x }));
        buf.resize(buf.len() + pcols - ncols, T::zero());
    }
    buf.resize(prows * pcols, T::zero());
    buf
}

/// The UV filter `racs_tools.gaussft` evaluates, `g_ratio · exp(g_arg − dg_arg)`,
/// reduced to one quadratic form `a u² + b uv + c v²` in the exponent so it
/// is cheap enough to evaluate on the fly.
#[derive(Clone, Copy, Debug)]
struct UvTaper {
    a: f64,
    b: f64,
    c: f64,
    /// DC gain, `√(Ω_new/Ω_old)`.
    ratio: f64,
}

impl UvTaper {
    fn new(old_beam: &Beam, new_beam: &Beam) -> Self {
        use std::f64::consts::PI;
        let fwhm_to_sigma = 2.0 * (2.0 * 2_f64.ln()).sqrt();
        let sigma = |fwhm_deg: f64| fwhm_deg.to_radians() / fwhm_to_sigma;
        let (sx, sy) = (sigma(new_beam.major_deg()), sigma(new_beam.minor_deg()));
        let (sx_in, sy_in) = (sigma(old_beam.major_deg()), sigma(old_beam.minor_deg()));

        // (σx u_r)² + (σy v_r)² with u_r = u cos θ − v sin θ, v_r = u sin θ + v cos θ.
        let form = |sx: f64, sy: f64, pa_deg: f64| {
            let (s, c) = pa_deg.to_radians().sin_cos();
            let (sx2, sy2) = (sx * sx, sy * sy);
            (
                sx2 * c * c + sy2 * s * s,
                2.0 * c * s * (sy2 - sx2),
                sx2 * s * s + sy2 * c * c,
            )
        };
        let (a, b, c) = form(sx, sy, new_beam.pa_deg());
        let (a_in, b_in, c_in) = form(sx_in, sy_in, old_beam.pa_deg());
        let k = -2.0 * PI * PI;
        Self {
            a: k * (a - a_in),
            b: k * (b - b_in),
            c: k * (c - c_in),
            ratio: (2.0 * PI * sx * sy).sqrt() / (2.0 * PI * sx_in * sy_in).sqrt(),
        }
    }

    /// The taper in cycles per pixel, for a grid whose pixels step by the
    /// columns of `m_deg` on the sky (see [`PixelScale::matrix`]). The kernel's
    /// sky covariance, `Σ_new − Σ_old`, becomes `M⁻¹ (Σ_new − Σ_old) M⁻ᵀ` in pixels.
    fn in_pixels(old_beam: &Beam, new_beam: &Beam, m_deg: [[f64; 2]; 2]) -> Self {
        use std::f64::consts::PI;
        let fwhm_to_sigma = 2.0 * (2.0 * 2_f64.ln()).sqrt();
        // Covariance along (east, north), in radians²; the major axis points
        // along (sin PA, cos PA).
        let cov = |b: &Beam| {
            let major = (b.major_deg().to_radians() / fwhm_to_sigma).powi(2);
            let minor = (b.minor_deg().to_radians() / fwhm_to_sigma).powi(2);
            let (s, c) = b.pa_deg().to_radians().sin_cos();
            let en = (major - minor) * s * c;
            [
                [major * s * s + minor * c * c, en],
                [en, major * c * c + minor * s * s],
            ]
        };
        let (new, old) = (cov(new_beam), cov(old_beam));
        let d = [
            [new[0][0] - old[0][0], new[0][1] - old[0][1]],
            [new[1][0] - old[1][0], new[1][1] - old[1][1]],
        ];
        let m = m_deg.map(|row| row.map(f64::to_radians));
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let inv = [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ];
        // p = inv · d · invᵀ, symmetric; index 0 is x (columns), 1 is y (rows).
        let p = |i: usize, j: usize| {
            (0..2)
                .flat_map(|k| (0..2).map(move |l| (k, l)))
                .map(|(k, l)| inv[i][k] * d[k][l] * inv[j][l])
                .sum::<f64>()
        };
        let k = -2.0 * PI * PI;
        Self {
            a: k * p(1, 1),
            b: k * 2.0 * p(0, 1),
            c: k * p(0, 0),
            ratio: Self::new(old_beam, new_beam).ratio,
        }
    }

    fn at(&self, u: f64, v: f64) -> f64 {
        self.ratio * (self.a * u * u + self.b * u * v + self.c * v * v).exp()
    }
}

/// The UV filter from `old_beam` to `new_beam` at every `(u, v)`, row-major,
/// and its DC gain, as `racs_tools.gaussft` returns them. For inspection and
/// testing; the convolution evaluates the filter on the fly.
pub fn gaussft(
    old_beam: &Beam,
    new_beam: &Beam,
    u_freqs: &[f64],
    v_freqs: &[f64],
) -> (Vec<f64>, f64) {
    let taper = UvTaper::new(old_beam, new_beam);
    let g_final = u_freqs
        .iter()
        .flat_map(|&u| v_freqs.iter().map(move |&v| taper.at(u, v)))
        .collect();
    (g_final, taper.ratio)
}

/// numpy's `fftfreq(n, d)`, including its negative Nyquist bin for even `n`.
///
/// ```
/// use convolve_rs::fftfreq;
///
/// assert_eq!(fftfreq(4, 1.0), vec![0.0, 0.25, -0.5, -0.25]);
/// assert_eq!(fftfreq(5, 1.0), vec![0.0, 0.2, 0.4, -0.4, -0.2]);
/// ```
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    let m = n.div_ceil(2);
    (0..n)
        .map(|i| {
            if i < m {
                i as f64 * val
            } else {
                (i as f64 - n as f64) * val
            }
        })
        .collect()
}

/// Columns transformed together in the column pass, so each gather reads
/// whole cache lines of the row-major spectrum.
const COL_BLOCK: usize = 16;

/// UV coordinates of the half spectrum, with `a·u²` precomputed per row.
///
/// On an even axis the Nyquist bin holds +1/2 and -1/2 cycles per pixel at
/// once. Convolving the image's band-limited interpolant exactly and
/// resampling averages the filter over both, which turns the cross term
/// `exp(b u v)` into `cosh(b u v)` there and leaves the rest unchanged.
struct UvGrid {
    u: Vec<f64>,
    au2: Vec<f64>,
    /// `v` for the `ncols/2 + 1` columns of the half spectrum.
    v: Vec<f64>,
    /// The Nyquist row and column, on even axes.
    nyquist_row: Option<usize>,
    nyquist_col: Option<usize>,
}

impl UvGrid {
    fn new(
        (nrows, ncols): (usize, usize),
        row_step_rad: f64,
        col_step_rad: f64,
        taper: &UvTaper,
    ) -> Self {
        let u = fftfreq(nrows, row_step_rad);
        let au2 = u.iter().map(|&u| taper.a * u * u).collect();
        let mut v = fftfreq(ncols, col_step_rad);
        v.truncate(ncols / 2 + 1);
        let nyquist = |n: usize| n.is_multiple_of(2).then_some(n / 2);
        Self {
            u,
            au2,
            v,
            nyquist_row: nyquist(nrows),
            nyquist_col: nyquist(ncols),
        }
    }
}

fn as_complex<T: FftFloat>(re: &[T]) -> &[Complex<T>] {
    debug_assert!(re.len().is_multiple_of(2));
    // SAFETY: `Complex<T>` is `#[repr(C)] { re: T, im: T }`: two `T`s with the
    // alignment of one. The view covers the first `2 * (len / 2)` values.
    unsafe { std::slice::from_raw_parts(re.as_ptr().cast(), re.len() / 2) }
}

fn as_complex_mut<T: FftFloat>(re: &mut [T]) -> &mut [Complex<T>] {
    debug_assert!(re.len().is_multiple_of(2));
    // SAFETY: as for `as_complex`, carrying over the exclusive borrow.
    unsafe { std::slice::from_raw_parts_mut(re.as_mut_ptr().cast(), re.len() / 2) }
}

fn fft_error(e: impl std::fmt::Display) -> ConvolveError {
    ConvolveError::Fft(e.to_string())
}

/// Convolve the `nrows × ncols` row-major image in `buf` with `gain ·` the
/// unit-gain filter, in place. `gain` must include the inverse FFT's `1/N`.
///
/// The buffer grows by two values per row to hold the half spectrum. On Linux
/// that remaps rather than copies (0.9 ms against 180 ms for a copy at 8192²,
/// measured).
fn convolve_in_place<T: FftFloat>(
    buf: &mut Vec<T>,
    taper: &UvTaper,
    gain: f64,
    grid: &UvGrid,
    plans: &FftPlans<T>,
) -> Result<(), ConvolveError> {
    let (nrows, ncols) = (plans.nrows, plans.ncols);
    let stride = 2 * plans.nhalf;
    buf.resize(nrows * stride, T::zero());
    // Called from a rayon worker, the caller is already parallel over planes;
    // the in-place row passes then avoid two copies of the image.
    let parallel_rows = rayon::current_thread_index().is_none();

    if parallel_rows {
        // Space rows out to the half-spectrum stride so each transforms
        // independently.
        for i in (1..nrows).rev() {
            buf.copy_within(i * ncols..(i + 1) * ncols, i * stride);
        }
        forward_rows(plans, buf)?;
    } else {
        forward_rows_in_place(plans, buf)?;
    }
    filter_columns(plans, as_complex_mut(buf), taper, gain, grid)?;
    if parallel_rows {
        inverse_rows(plans, buf)?;
        for i in 1..nrows {
            buf.copy_within(i * stride..i * stride + ncols, i * ncols);
        }
    } else {
        inverse_rows_in_place(plans, buf)?;
    }
    buf.truncate(nrows * ncols);
    Ok(())
}

/// Sequential [`forward_rows`] on the packed image. Spectrum row `i` starts at
/// `2·nhalf·i ≥ ncols·i`, so it can only overwrite image rows `≥ i`; going from
/// the last row up, those are already transformed.
fn forward_rows_in_place<T: FftFloat>(
    plans: &FftPlans<T>,
    buf: &mut [T],
) -> Result<(), ConvolveError> {
    let (ncols, stride) = (plans.ncols, 2 * plans.nhalf);
    let mut scratch = plans.r2c.make_scratch_vec();
    let mut inrow = plans.r2c.make_input_vec();
    for i in (0..plans.nrows).rev() {
        inrow.copy_from_slice(&buf[i * ncols..(i + 1) * ncols]);
        let out = as_complex_mut(&mut buf[stride * i..stride * (i + 1)]);
        plans
            .r2c
            .process_with_scratch(&mut inrow, out, &mut scratch)
            .map_err(fft_error)?;
    }
    Ok(())
}

/// Sequential [`inverse_rows`] that packs the image as it goes. Image row `i`
/// ends at `ncols·(i + 1) ≤ 2·nhalf·(i + 1)`, so it can only overwrite
/// spectrum rows `≤ i`, which are already transformed.
fn inverse_rows_in_place<T: FftFloat>(
    plans: &FftPlans<T>,
    buf: &mut [T],
) -> Result<(), ConvolveError> {
    let (ncols, nhalf) = (plans.ncols, plans.nhalf);
    let stride = 2 * nhalf;
    let mut scratch = plans.c2r.make_scratch_vec();
    let mut inrow = plans.c2r.make_input_vec();
    for i in 0..plans.nrows {
        inrow.copy_from_slice(as_complex(&buf[stride * i..stride * (i + 1)]));
        inrow[0].im = T::zero();
        if ncols.is_multiple_of(2) {
            inrow[nhalf - 1].im = T::zero();
        }
        plans
            .c2r
            .process_with_scratch(
                &mut inrow,
                &mut buf[i * ncols..(i + 1) * ncols],
                &mut scratch,
            )
            .map_err(fft_error)?;
    }
    Ok(())
}

/// Real-to-complex FFT of every row; row `i` starts at `i * 2 * nhalf`.
fn forward_rows<T: FftFloat>(plans: &FftPlans<T>, buf: &mut [T]) -> Result<(), ConvolveError> {
    let ncols = plans.ncols;
    buf.par_chunks_mut(2 * plans.nhalf).try_for_each_init(
        || (plans.r2c.make_input_vec(), plans.r2c.make_scratch_vec()),
        |(inrow, scratch), row| {
            inrow.copy_from_slice(&row[..ncols]);
            plans
                .r2c
                .process_with_scratch(inrow, as_complex_mut(row), scratch)
                .map_err(fft_error)
        },
    )
}

/// Complex-to-real FFT of every half-spectrum row, leaving each image row at
/// the start of its slot.
fn inverse_rows<T: FftFloat>(plans: &FftPlans<T>, buf: &mut [T]) -> Result<(), ConvolveError> {
    let (ncols, nhalf) = (plans.ncols, plans.nhalf);
    let even = ncols.is_multiple_of(2);
    buf.par_chunks_mut(2 * nhalf).try_for_each_init(
        || {
            (
                plans.c2r.make_input_vec(),
                plans.c2r.make_output_vec(),
                plans.c2r.make_scratch_vec(),
            )
        },
        |(inrow, outrow, scratch), row| {
            inrow.copy_from_slice(as_complex(row));
            // c2r wants the DC and (even ncols) Nyquist bins exactly real.
            inrow[0].im = T::zero();
            if even {
                inrow[nhalf - 1].im = T::zero();
            }
            plans
                .c2r
                .process_with_scratch(inrow, outrow, scratch)
                .map_err(fft_error)?;
            row[..ncols].copy_from_slice(outrow);
            Ok(())
        },
    )
}

/// Raw pointer to the spectrum, shared by the column-block tasks.
#[derive(Clone, Copy)]
struct SpectrumPtr<T>(*mut Complex<T>);
// SAFETY: each task only touches the columns of its own block (see
// `filter_columns`), so no element is accessed by two threads.
unsafe impl<T: Send> Send for SpectrumPtr<T> {}
unsafe impl<T: Send> Sync for SpectrumPtr<T> {}

impl<T> SpectrumPtr<T> {
    /// A method rather than field access, so closures capture the whole
    /// (`Sync`) wrapper instead of the raw pointer.
    fn get(self) -> *mut Complex<T> {
        self.0
    }
}

/// Column FFT, filter and inverse column FFT of the `nrows × nhalf` half
/// spectrum, [`COL_BLOCK`] columns per task.
fn filter_columns<T: FftFloat>(
    plans: &FftPlans<T>,
    spectrum: &mut [Complex<T>],
    taper: &UvTaper,
    gain: f64,
    grid: &UvGrid,
) -> Result<(), ConvolveError> {
    let (nrows, nhalf) = (plans.nrows, plans.nhalf);
    if spectrum.len() != nrows * nhalf {
        return Err(ConvolveError::Fft("spectrum has the wrong size".into()));
    }
    let zero = Complex::new(T::zero(), T::zero());
    let scratch_len = plans
        .col_fwd
        .get_inplace_scratch_len()
        .max(plans.col_inv.get_inplace_scratch_len());
    let ptr = SpectrumPtr(spectrum.as_mut_ptr());

    (0..nhalf.div_ceil(COL_BLOCK))
        .into_par_iter()
        .for_each_init(
            || (vec![zero; COL_BLOCK * nrows], vec![zero; scratch_len]),
            |(tile, scratch), block| {
                let j0 = block * COL_BLOCK;
                let width = COL_BLOCK.min(nhalf - j0);
                let tile = &mut tile[..width * nrows];
                // SAFETY: `i * nhalf + j0 + k` is in bounds for `i < nrows` and
                // `j0 + k < nhalf`, and this block is the only one using columns
                // `j0..j0 + width`. The `&mut` borrow of `spectrum` outlives the loop.
                let at = |i: usize, k: usize| unsafe { ptr.get().add(i * nhalf + j0 + k) };

                for i in 0..nrows {
                    for k in 0..width {
                        tile[k * nrows + i] = unsafe { at(i, k).read() };
                    }
                }
                plans.col_fwd.process_with_scratch(tile, scratch);
                for (k, col) in tile.chunks_exact_mut(nrows).enumerate() {
                    let j = j0 + k;
                    let v = grid.v[j];
                    let (bv, cv2) = (taper.b * v, taper.c * v * v);
                    let nyquist_col = grid.nyquist_col == Some(j);
                    for (i, ((s, &u), &au2)) in
                        col.iter_mut().zip(&grid.u).zip(&grid.au2).enumerate()
                    {
                        let g = if nyquist_col || grid.nyquist_row == Some(i) {
                            gain * (au2 + cv2).exp() * (bv * u).cosh()
                        } else {
                            gain * (au2 + bv * u + cv2).exp()
                        };
                        *s = s.scale(cast_saturating::<T>(g));
                    }
                }
                plans.col_inv.process_with_scratch(tile, scratch);
                for i in 0..nrows {
                    for k in 0..width {
                        unsafe { at(i, k).write(tile[k * nrows + i]) };
                    }
                }
            },
        );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Mirroring the image and both PAs mirrors the result. Taking one sign of
    /// the Nyquist bin, rather than averaging both, broke this on even axes.
    #[test]
    fn test_mirror_symmetric_on_even_and_odd_grids() {
        let old = Beam::from_arcsec(10.0, 8.0, 30.0).unwrap();
        let new = Beam::from_arcsec(12.3, 9.1, 65.8).unwrap();
        let flip_pa = |b: &Beam| Beam::new(b.major_deg(), b.minor_deg(), -b.pa_deg()).unwrap();
        let dx = 2.0 / 3600.0;
        for (ny, nx) in [(32, 32), (32, 33), (33, 32), (33, 33)] {
            let image = Array2::from_shape_fn((ny, nx), |(i, j)| {
                (((i * 131 + j * 71) % 97) as f64 / 97.0 - 0.5) * 2.0
            });
            let mirror =
                |a: &Array2<f64>| Array2::from_shape_fn((ny, nx), |(i, j)| a[(i, nx - 1 - j)]);
            let direct = convolve_uv(&image, &old, &new, dx, dx, None).unwrap().image;
            let flipped = convolve_uv(mirror(&image), &flip_pa(&old), &flip_pa(&new), dx, dx, None)
                .unwrap()
                .image;
            let worst = mirror(&flipped)
                .iter()
                .zip(direct.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-12, "{ny}x{nx}: mirrored result off by {worst}");
        }
    }

    #[test]
    fn test_fftfreq() {
        let f = fftfreq(4, 1.0);
        let expected = [0.0, 0.25, -0.5, -0.25];
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "got {a}, want {b}");
        }
    }

    /// A flat filter returns the image, for even and odd widths (the Nyquist
    /// bin differs) and both precisions.
    fn roundtrip<T: FftFloat + std::fmt::Debug>(nrows: usize, ncols: usize, tol: f64) {
        let data: Vec<T> = (0..nrows * ncols)
            .map(|k| cast_saturating::<T>(((k * 7) % 13) as f64 - 6.0))
            .collect();
        let plans = FftPlans::<T>::new(nrows, ncols);
        let mut buf = data.clone();
        let flat = UvTaper {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            ratio: 1.0,
        };
        let grid = UvGrid::new((nrows, ncols), 1.0, 1.0, &flat);
        let n = (nrows * ncols) as f64;
        convolve_in_place(&mut buf, &flat, 1.0 / n, &grid, &plans).unwrap();
        for (a, b) in data.iter().zip(buf.iter()) {
            let (a, b) = (a.to_f64().unwrap(), b.to_f64().unwrap());
            assert!((a - b).abs() < tol, "roundtrip {nrows}x{ncols}: {a} vs {b}");
        }
    }

    /// The sequential row passes, used on rayon workers, give the same result
    /// as the parallel ones.
    #[test]
    fn test_row_paths_agree() {
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        for (r, c) in [(37, 36), (40, 71)] {
            let img = test_image(r, c, true);
            let outside = convolve_uv(&img, &old, &new, dx, dx, None).unwrap().image;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let inside = pool
                .install(|| convolve_uv(&img, &old, &new, dx, dx, None))
                .unwrap()
                .image;
            for (a, b) in outside.iter().zip(inside.iter()) {
                assert_eq!(a.is_nan(), b.is_nan());
                if !a.is_nan() {
                    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn test_in_place_roundtrip() {
        for &(r, c) in &[(4, 4), (5, 7), (6, 9), (7, 6), (1, 8), (33, 40)] {
            roundtrip::<f64>(r, c, 1e-10);
            roundtrip::<f32>(r, c, 1e-4);
        }
    }

    /// Reference: the image and the missing-data mask (NaNs, and the padding
    /// beyond the image) padded as `convolve_uv` pads them, each through full
    /// complex 2D FFTs and the filter on the whole grid, then normalised by
    /// the share of the kernel on data and blanked where too little was.
    fn reference(
        image: &Array2<f64>,
        old: &Beam,
        new: &Beam,
        dx: f64,
        nan_fraction: f64,
    ) -> Array2<f64> {
        let (nrows, ncols) = image.dim();
        let (prows, pcols) = padded_size((nrows, ncols), old, new, PixelScale::cdelt(dx, dx));
        let taper = UvTaper::new(old, new);
        let (u, v) = (
            fftfreq(prows, dx.to_radians()),
            fftfreq(pcols, dx.to_radians()),
        );
        let filter: Vec<f64> = (0..prows)
            .flat_map(|i| (0..pcols).map(move |j| (i, j)))
            .map(|(i, j)| {
                let (u, v) = (u[i], v[j]);
                let even = taper.a * u * u + taper.c * v * v;
                let nyquist =
                    (prows % 2 == 0 && i == prows / 2) || (pcols % 2 == 0 && j == pcols / 2);
                if nyquist {
                    even.exp() * (taper.b * u * v).cosh()
                } else {
                    (even + taper.b * u * v).exp()
                }
            })
            .collect();
        let mut planner = FftPlanner::<f64>::new();
        let (row_fwd, col_fwd) = (
            planner.plan_fft_forward(pcols),
            planner.plan_fft_forward(prows),
        );
        let (row_inv, col_inv) = (
            planner.plan_fft_inverse(pcols),
            planner.plan_fft_inverse(prows),
        );
        let fft2 = |z: &mut [Complex<f64>], rows: &dyn Fft<f64>, cols: &dyn Fft<f64>| {
            z.chunks_exact_mut(pcols).for_each(|row| rows.process(row));
            let mut col = vec![Complex::new(0.0, 0.0); prows];
            for j in 0..pcols {
                (0..prows).for_each(|i| col[i] = z[i * pcols + j]);
                cols.process(&mut col);
                (0..prows).for_each(|i| z[i * pcols + j] = col[i]);
            }
        };
        let conv = |value: &dyn Fn(Option<f64>) -> f64| -> Vec<f64> {
            let mut z: Vec<Complex<f64>> = (0..prows * pcols)
                .map(|k| {
                    let (i, j) = (k / pcols, k % pcols);
                    let x = (i < nrows && j < ncols).then(|| image[(i, j)]);
                    Complex::new(value(x), 0.0)
                })
                .collect();
            fft2(&mut z, &*row_fwd, &*col_fwd);
            z.iter_mut().zip(&filter).for_each(|(s, &g)| *s *= g);
            fft2(&mut z, &*row_inv, &*col_inv);
            z.iter().map(|s| s.re / (prows * pcols) as f64).collect()
        };
        let data = conv(&|x| x.filter(|x| !x.is_nan()).unwrap_or(0.0));
        let missing = conv(&|x| if x.is_none_or(f64::is_nan) { 1.0 } else { 0.0 });
        Array2::from_shape_fn((nrows, ncols), |(i, j)| {
            let (x, m) = (taper.ratio * data[i * pcols + j], missing[i * pcols + j]);
            if m > nan_fraction.max(MISSING_ROUND_OFF) {
                f64::NAN
            } else if m > MISSING_ROUND_OFF {
                x / (1.0 - m)
            } else {
                x
            }
        })
    }

    fn test_image(nrows: usize, ncols: usize, nans: bool) -> Array2<f64> {
        let mut img = Array2::from_shape_fn((nrows, ncols), |(i, j)| {
            (((i * 31 + j * 17) % 23) as f64 - 11.0) / 7.0 + if i > nrows / 2 { 3.0 } else { 0.0 }
        });
        if nans {
            for i in 0..nrows / 3 {
                for j in 0..ncols / 2 {
                    img[(i, j)] = f64::NAN;
                }
            }
            img[(nrows - 2, ncols - 3)] = f64::NAN; // isolated: interpolated over
        }
        img
    }

    /// Matches the reference for odd sizes, rotated beams, NaNs, blanking
    /// fractions, widths that leave a partial column block, and an image
    /// large enough to be done in tiles.
    #[test]
    fn test_convolve_uv_matches_padded_reference() {
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(12.0, 10.0, 20.0).unwrap();
        let new = Beam::from_arcsec(18.0, 15.0, 35.0).unwrap();
        for &(r, c) in &[(64, 48), (65, 49), (40, 71), (37, 36), (300, 280)] {
            for nans in [false, true] {
                for nan_fraction in [0.5, 0.1] {
                    let img = test_image(r, c, nans);
                    let expected = reference(&img, &old, &new, dx, nan_fraction);
                    let scale = PixelScale::cdelt(dx, dx);
                    let got = convolve_uv_with(&img, &old, &new, scale, nan_fraction, None)
                        .unwrap()
                        .image;
                    let got32 = convolve_uv_with(
                        img.mapv(|x| x as f32),
                        &old,
                        &new,
                        scale,
                        nan_fraction,
                        None,
                    )
                    .unwrap()
                    .image;
                    let what = format!("{r}x{c} nans={nans} fraction={nan_fraction}");
                    for ((&e, &g), &g32) in expected.iter().zip(got.iter()).zip(got32.iter()) {
                        assert_eq!(e.is_nan(), g.is_nan(), "{what}: NaN mask differs");
                        assert_eq!(e.is_nan(), g32.is_nan(), "{what}: f32 NaN mask");
                        if !e.is_nan() {
                            assert!((e - g).abs() < 1e-9, "{what}: {e} vs {g}");
                            assert!((e - g32 as f64).abs() < 1e-4, "{what}: f32 {e} vs {g32}");
                        }
                    }
                }
            }
        }
    }

    /// A flat image stays flat up to its edges and NaNs, rather than dimming
    /// towards them, and nothing wraps from one edge to the other.
    #[test]
    fn test_flat_image_stays_flat_and_nothing_wraps() {
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(12.0, 10.0, 20.0).unwrap();
        let new = Beam::from_arcsec(30.0, 25.0, 35.0).unwrap();
        let mut img = Array2::<f64>::ones((120, 100));
        for i in 50..70 {
            for j in 40..60 {
                img[(i, j)] = f64::NAN;
            }
        }
        // A well-sampled kernel, and a coarsely sampled one that rings.
        for (o, n) in [
            (old, new),
            (
                Beam::from_arcsec(9.0, 9.0, 0.0).unwrap(),
                Beam::from_arcsec(12.3, 9.1, 30.0).unwrap(),
            ),
        ] {
            let res = convolve_uv(&img, &o, &n, dx, dx, None).unwrap();
            for &x in res.image.iter().filter(|x| !x.is_nan()) {
                assert!((x - res.scaling_factor).abs() < 1e-7, "{x}");
            }
        }
        let mut point = Array2::<f64>::zeros((120, 100));
        point[(60, 0)] = 1.0;
        let out = convolve_uv(&point, &old, &new, dx, dx, None).unwrap().image;
        assert!(
            out[(60, 99)].abs() < 1e-12,
            "flux wrapped round: {}",
            out[(60, 99)]
        );
    }

    /// Owned and borrowed inputs give identical output.
    #[test]
    fn test_owned_matches_borrowed() {
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        for nans in [false, true] {
            let img = test_image(45, 38, nans).mapv(|x| x as f32);
            let plans = FftPlans::<f32>::new(45, 38);
            let borrowed = convolve_uv(&img, &old, &new, dx, dx, Some(&plans)).unwrap();
            let owned = convolve_uv(img.clone(), &old, &new, dx, dx, Some(&plans)).unwrap();
            assert_eq!(borrowed.scaling_factor, owned.scaling_factor);
            for (a, b) in borrowed.image.iter().zip(owned.image.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "owned path changed output");
            }
        }
    }

    /// A transposed input is convolved as the logical image.
    #[test]
    fn test_owned_non_standard_layout() {
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        let img = test_image(30, 41, true).reversed_axes(); // 41 x 30, column-major
        let plans = FftPlans::<f64>::new(41, 30);
        let std_layout = img.as_standard_layout().to_owned();
        let a = convolve_uv(&std_layout, &old, &new, dx, dx, Some(&plans)).unwrap();
        let b = convolve_uv(img, &old, &new, dx, dx, Some(&plans)).unwrap();
        for (x, y) in a.image.iter().zip(b.image.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    /// A row-sliced owned array keeps its parent's buffer at an offset; the
    /// owned path must convolve the slice, not the parent's leading rows.
    #[test]
    fn test_owned_sliced_rows() {
        use ndarray::s;
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        for nans in [false, true] {
            let parent = test_image(48, 30, nans);
            for sl in [s![5..45, ..], s![..40, ..]] {
                let owned = parent.clone().slice_move(sl);
                assert!(owned.is_standard_layout());
                let plans = FftPlans::<f64>::new(40, 30);
                let want = convolve_uv(owned.view(), &old, &new, dx, dx, Some(&plans)).unwrap();
                let got = convolve_uv(owned, &old, &new, dx, dx, Some(&plans)).unwrap();
                for (a, b) in want.image.iter().zip(got.image.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }

    /// The sign of a pixel size must not change the result (CDELT1 is
    /// normally negative).
    #[test]
    fn test_negative_pixel_size_keeps_pa() {
        let old = Beam::from_arcsec(8.0, 5.0, 30.0).unwrap();
        let new = Beam::from_arcsec(16.0, 9.0, 70.0).unwrap();
        let img = test_image(40, 36, false);
        let d = 2.0 / 3600.0;
        let pos = convolve_uv(&img, &old, &new, d, d, None).unwrap().image;
        for (dx, dy) in [(-d, d), (d, -d), (-d, -d)] {
            let got = convolve_uv(&img, &old, &new, dx, dy, None).unwrap().image;
            for (a, b) in pos.iter().zip(got.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "dx={dx} dy={dy}");
            }
        }
    }

    /// The missing fractions found in tiles are the whole-image ones, NaN
    /// regions on the edges and lone NaN pixels included, in both precisions.
    fn tiled_matches_whole<T: FftFloat + std::fmt::Debug>() {
        let (nrows, ncols) = (600, 560);
        let d = 2.0 / 3600.0;
        let old = Beam::from_arcsec(10.0, 8.0, 30.0).unwrap();
        let new = Beam::from_arcsec(16.0, 11.0, 75.0).unwrap();
        let mut image = Array2::<T>::from_elem((nrows, ncols), T::one());
        for i in 0..nrows {
            for j in 0..ncols {
                let corner = i + j < 150 || (nrows - i) + (ncols - j) < 120;
                let block = (300..340).contains(&i) && (200..330).contains(&j);
                let left_edge = (420..470).contains(&i) && j < 6;
                if corner || block || left_edge {
                    image[(i, j)] = T::nan();
                }
            }
        }
        image[(100, 400)] = T::nan();
        image[(0, 300)] = T::nan();
        let scale = PixelScale::cdelt(d, d);
        let (taper, steps) = taper_for(&old, &new, scale);
        let reach = kernel_reach(&taper, steps);
        let tiling = Tiling::new((nrows, ncols), reach).expect("tiles fit");
        assert!(
            tiling.size * 3 < nrows,
            "tiles of {} for {nrows}",
            tiling.size
        );
        let padded = padded_size((nrows, ncols), &old, &new, scale);
        let plans = FftPlans::<T>::new(padded.0, padded.1);
        let grid = UvGrid::new(padded, steps.0, steps.1, &taper);
        let dense = |spans: Vec<MissingSpan<T>>| {
            let mut out = Array2::<f64>::zeros((nrows, ncols));
            for span in spans {
                for (k, c) in span.cols.enumerate() {
                    out[(span.row, c)] = span.values.get(k).map_or(1.0, |v| v.to_f64().unwrap());
                }
            }
            out
        };
        let whole = dense(missing_whole(image.view(), &taper, &grid, &plans).unwrap());
        let tiled = dense(missing_tiled(image.view(), &taper, steps, &tiling, true).unwrap());
        let worst = whole
            .iter()
            .zip(tiled.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        // This kernel's response at the Nyquist frequency is ~3e-6, and the
        // binary mask has power there, so transforms of different sizes
        // differ at that level.
        assert!(worst < 1e-5, "tiled fractions off by {worst}");
        assert!(whole[(0, 0)] > 0.99 && whole[(nrows / 2, ncols - 1)] > 0.3);
    }

    #[test]
    fn test_tiled_missing_matches_whole() {
        tiled_matches_whole::<f64>();
        tiled_matches_whole::<f32>();
    }

    /// Kernels under about two pixels wide ring, in any orientation; wider
    /// ones do not.
    #[test]
    fn test_narrow_kernels_ring() {
        let step = (2.0f64 / 3600.0).to_radians();
        let ring = |old: (f64, f64, f64), new: (f64, f64, f64)| {
            let beam = |(a, b, pa)| Beam::from_arcsec(a, b, pa).unwrap();
            rings(&UvTaper::new(&beam(old), &beam(new)), (step, step))
        };
        assert!(ring((10.0, 10.0, 0.0), (10.5, 10.3, 0.0)));
        // Wide along one axis, but under a pixel along the other
        assert!(ring((10.0, 10.0, 0.0), (30.0, 10.3, 0.0)));
        assert!(ring((10.0, 10.0, 0.0), (30.0, 10.3, 37.0)));
        // About 1.4 pixels of standard deviation along the narrow axis
        assert!(ring((10.0, 8.0, 0.0), (20.0, 10.0, 0.0)));
        // About 1.8 pixels, where the transform at Nyquist is about 2e-7
        assert!(!ring((10.0, 10.0, 0.0), (13.0, 13.0, 0.0)));
        // 2.5 pixels of standard deviation along every axis
        assert!(!ring((10.0, 10.0, 0.0), (15.3, 15.3, 0.0)));
        assert!(!ring((10.0, 8.0, 30.0), (25.0, 20.0, 75.0)));
    }

    /// A kernel large next to the image is done whole.
    #[test]
    fn test_no_tiles_for_large_kernels() {
        let taper = UvTaper::new(
            &Beam::from_arcsec(10.0, 10.0, 0.0).unwrap(),
            &Beam::from_arcsec(200.0, 200.0, 0.0).unwrap(),
        );
        let step = (2.0f64 / 3600.0).to_radians();
        let reach = kernel_reach(&taper, (step, step));
        assert!(Tiling::new((600, 600), reach).is_none());
        assert!(Tiling::new((20000, 20000), reach).is_some());
    }

    /// The usual diagonal matrix takes the CDELT path; others are kept.
    #[test]
    fn test_pixel_scale_matrix_validation() {
        let d = 2.0 / 3600.0;
        assert_eq!(
            PixelScale::matrix([[-d, 0.0], [0.0, d]]),
            Some(PixelScale::cdelt(-d, d))
        );
        assert!(matches!(
            PixelScale::matrix([[d, 0.0], [0.0, d]]),
            Some(PixelScale(Scale::Matrix(_)))
        ));
        assert_eq!(PixelScale::matrix([[d, 2.0 * d], [d, 2.0 * d]]), None);
        assert_eq!(PixelScale::matrix([[f64::NAN, 0.0], [0.0, d]]), None);
    }

    /// The matrix path gives the same image as the CDELT path on a diagonal
    /// grid, to round-off.
    #[test]
    fn test_matrix_path_matches_cdelt_path() {
        let d = 2.0 / 3600.0;
        let old = Beam::from_arcsec(8.0, 5.0, 30.0).unwrap();
        let new = Beam::from_arcsec(16.0, 9.0, 70.0).unwrap();
        for nans in [false, true] {
            let img = test_image(40, 36, nans);
            let cdelt = convolve_uv(&img, &old, &new, -d, d, None).unwrap().image;
            let scale = PixelScale(Scale::Matrix([[-d, 0.0], [0.0, d]]));
            let matrix = convolve_uv_with(&img, &old, &new, scale, DEFAULT_NAN_FRACTION, None)
                .unwrap()
                .image;
            for (a, b) in cdelt.iter().zip(matrix.iter()) {
                assert_eq!(a.is_nan(), b.is_nan());
                if !a.is_nan() {
                    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
                }
            }
        }
    }

    /// On rotated, skewed and east-right grids, a smoothed point source has
    /// the pixel covariance `M⁻¹ (Σ_new − Σ_old) M⁻ᵀ`.
    #[test]
    fn test_kernel_covariance_on_any_grid() {
        let old = Beam::from_arcsec(6.0, 4.0, 20.0).unwrap();
        let new = Beam::from_arcsec(14.0, 9.0, 65.0).unwrap();
        let fwhm_to_sigma = 1.0 / (8.0 * 2f64.ln()).sqrt();
        let cov = |b: &Beam| {
            let (maj, min) = (
                (b.major_arcsec() * fwhm_to_sigma).powi(2),
                (b.minor_arcsec() * fwhm_to_sigma).powi(2),
            );
            let (s, c) = b.pa_deg().to_radians().sin_cos();
            let en = (maj - min) * s * c;
            [
                [maj * s * s + min * c * c, en],
                [en, maj * c * c + min * s * s],
            ]
        };
        let (cn, co) = (cov(&new), cov(&old));
        let (t, a) = (0.4f64, 2.0);
        for m in [
            [[-a * t.cos(), a * t.sin()], [a * t.sin(), a * t.cos()]],
            [[-a, 0.6], [0.3, 1.5 * a]],
            [[a, 0.0], [0.0, a]],
        ] {
            let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            let inv = [
                [m[1][1] / det, -m[0][1] / det],
                [-m[1][0] / det, m[0][0] / det],
            ];
            let want = |i: usize, j: usize| {
                let mut sum = 0.0;
                for k in 0..2 {
                    for l in 0..2 {
                        sum += inv[i][k] * (cn[k][l] - co[k][l]) * inv[j][l];
                    }
                }
                sum
            };
            let n = 96;
            let mut img = Array2::<f64>::zeros((n, n));
            img[(n / 2, n / 2)] = 1.0;
            let m_deg = m.map(|row| row.map(|x| x / 3600.0));
            let scale = PixelScale::matrix(m_deg).unwrap();
            let res = convolve_uv_with(&img, &old, &new, scale, DEFAULT_NAN_FRACTION, None)
                .unwrap()
                .image;
            let total: f64 = res.iter().filter(|v| !v.is_nan()).sum();
            let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
            for ((i, j), &v) in res.indexed_iter().filter(|(_, v)| !v.is_nan()) {
                let (x, y) = (j as f64 - (n / 2) as f64, i as f64 - (n / 2) as f64);
                xx += v * x * x;
                xy += v * x * y;
                yy += v * y * y;
            }
            let got = [xx / total, xy / total, yy / total];
            let expected = [want(0, 0), want(0, 1), want(1, 1)];
            let scale = expected[0].max(expected[2]);
            for (g, e) in got.iter().zip(expected) {
                assert!(
                    (g - e).abs() < 1e-3 * scale,
                    "{m:?}: {got:?} vs {expected:?}"
                );
            }
        }
    }

    /// Second moments (arcsec²) along columns (x) and rows (y) of `img`.
    fn moments(img: &Array2<f64>, dx_as: f64, dy_as: f64) -> (f64, f64) {
        // The blanked corners hold none of the flux.
        let img = img.mapv(|v| if v.is_nan() { 0.0 } else { v });
        let total: f64 = img.sum();
        let (mut cy, mut cx) = (0.0, 0.0);
        for ((i, j), &v) in img.indexed_iter() {
            cy += v * i as f64;
            cx += v * j as f64;
        }
        let (cy, cx) = (cy / total, cx / total);
        let (mut vy, mut vx) = (0.0, 0.0);
        for ((i, j), &v) in img.indexed_iter() {
            vy += v * (i as f64 - cy).powi(2);
            vx += v * (j as f64 - cx).powi(2);
        }
        (vx / total * dx_as * dx_as, vy / total * dy_as * dy_as)
    }

    /// With rectangular pixels the kernel has the same angular size along x
    /// and y; rows use CDELT2 and columns CDELT1.
    #[test]
    fn test_non_square_pixels() {
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(12.0, 12.0, 0.0).unwrap();
        let (dx_as, dy_as) = (1.5, 3.0);
        let mut img = Array2::<f64>::zeros((64, 96));
        img[(32, 48)] = 1.0;
        let res = convolve_uv(&img, &old, &new, dx_as / 3600.0, dy_as / 3600.0, None).unwrap();
        let (vx, vy) = moments(&res.image, dx_as, dy_as);
        let fwhm_to_sigma = 1.0 / (8.0 * 2f64.ln()).sqrt();
        let want = (144.0 - 36.0) * fwhm_to_sigma * fwhm_to_sigma;
        assert!((vx / want - 1.0).abs() < 1e-3, "x variance {vx} vs {want}");
        assert!((vy / want - 1.0).abs() < 1e-3, "y variance {vy} vs {want}");
    }

    #[test]
    fn test_unreachable_and_zero_beams_are_errors() {
        let img = test_image(16, 16, false);
        let d = 2.0 / 3600.0;
        let small = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
        let big = Beam::from_arcsec(20.0, 10.0, 0.0).unwrap();
        assert!(matches!(
            convolve_uv(&img, &big, &small, d, d, None),
            Err(ConvolveError::Unreachable { .. })
        ));
        assert!(matches!(
            convolve_uv(&img, &Beam::zero(), &small, d, d, None),
            Err(ConvolveError::ZeroSizeBeam(_))
        ));
        // A one-dimensional kernel is fine.
        let round = Beam::from_arcsec(20.0, 20.0, 0.0).unwrap();
        assert!(convolve_uv(&img, &big, &round, d, d, None).is_ok());
    }

    /// An all-NaN image comes back as is, with the same scaling factor a
    /// finite image would get.
    #[test]
    fn test_all_nan_scaling_factor() {
        let d = 2.5 / 3600.0;
        let old = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
        let new = Beam::from_arcsec(20.0, 20.0, 0.0).unwrap();
        let nan = Array2::from_elem((16, 16), f32::NAN);
        let res = convolve_uv(&nan, &old, &new, d, d, None).unwrap();
        assert!(res.image.iter().all(|x| x.is_nan()));
        assert!((res.scaling_factor - 2.0).abs() < 1e-12);
    }

    /// With nearly equal beams the kernel is narrower than a pixel, so every
    /// NaN input pixel must still be NaN in the output, including the edge of
    /// a blanked block, where the mask falls just short of 1.
    #[test]
    fn test_near_equal_beams_keep_nan_block() {
        let d = 2.5 / 3600.0;
        let old = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
        let new = Beam::from_arcsec(10.000001, 10.0, 0.0).unwrap();
        assert!(!old.approx_eq(&new));
        let n = 256;
        let mut img = test_image(n, n, false).mapv(|x| x as f32);
        for i in 64..192 {
            for j in 64..192 {
                img[(i, j)] = f32::NAN;
            }
        }
        let out = convolve_uv(&img, &old, &new, d, d, None).unwrap().image;
        for (x, y) in img.iter().zip(out.iter()) {
            assert_eq!(x.is_nan(), y.is_nan());
        }
    }

    /// Plans of the wrong size are replaced, so the image is the same to the
    /// bit whatever plans are passed. Padding more would change it slightly
    /// for a kernel narrower than a pixel.
    #[test]
    fn test_plans_of_any_size_give_the_same_image() {
        let img = test_image(40, 36, true);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        let d = 2.0 / 3600.0;
        let want = convolve_uv(&img, &old, &new, d, d, None).unwrap().image;
        let exact = padded_size((40, 36), &old, &new, PixelScale::cdelt(d, d));
        for (r, c) in [(16, 17), (80, 75), exact] {
            let plans = FftPlans::<f64>::new(r, c);
            let got = convolve_uv(&img, &old, &new, d, d, Some(&plans))
                .unwrap()
                .image;
            for (a, b) in want.iter().zip(got.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "plans {r}x{c}");
            }
        }
    }

    /// `gaussft`'s collapsed quadratic form agrees with the per-beam rotated
    /// expression `racs_tools.gaussft` evaluates.
    #[test]
    fn test_gaussft_matches_rotated_form() {
        let old = Beam::from_arcsec(12.0, 10.0, 20.0).unwrap();
        let new = Beam::from_arcsec(18.0, 15.0, 35.0).unwrap();
        let u = fftfreq(33, 1.2e-5);
        let v = fftfreq(20, 1.2e-5);
        let (g, ratio) = gaussft(&old, &new, &u, &v);
        let f = 2.0 * (2.0 * 2f64.ln()).sqrt();
        let arg = |b: &Beam, u: f64, v: f64| {
            let (sx, sy) = (
                b.major_deg().to_radians() / f,
                b.minor_deg().to_radians() / f,
            );
            let (s, c) = b.pa_deg().to_radians().sin_cos();
            let (ur, vr) = (u * c - v * s, u * s + v * c);
            -2.0 * std::f64::consts::PI.powi(2) * ((sx * ur).powi(2) + (sy * vr).powi(2))
        };
        for (i, &uu) in u.iter().enumerate() {
            for (j, &vv) in v.iter().enumerate() {
                let want = ratio * (arg(&new, uu, vv) - arg(&old, uu, vv)).exp();
                let got = g[i * v.len() + j];
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1e-300),
                    "{got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn test_convolve_uv_no_change_when_beams_equal() {
        let beam = Beam::new(10.0 / 3600.0, 10.0 / 3600.0, 0.0).unwrap();
        let img = Array2::from_elem((16, 16), 1.0_f32);
        let result = convolve_uv(&img, &beam, &beam, 2.5 / 3600.0, 2.5 / 3600.0, None).unwrap();
        assert!((result.scaling_factor - 1.0).abs() < 1e-10);
    }

    /// A point source convolves to a Gaussian peaked on the source whose sum
    /// is the filter's DC gain.
    #[test]
    fn test_convolve_uv_point_source_flux_and_peak() {
        let (n, dx) = (64usize, 2.0 / 3600.0);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(12.0, 12.0, 0.0).unwrap();

        let mut img = Array2::<f64>::zeros((n, n));
        img[(n / 2, n / 2)] = 1.0;

        let res = convolve_uv(&img, &old, &new, dx, dx, None).unwrap();
        // The corners, mostly beyond the image, are blank.
        let total: f64 = res.image.iter().filter(|v| !v.is_nan()).sum();

        assert!(
            (total - res.scaling_factor).abs() < 1e-6,
            "integral {total} != DC gain {}",
            res.scaling_factor
        );

        let peak = res.image[(n / 2, n / 2)];
        assert!(peak > 0.0);
        for &v in res.image.iter().filter(|v| !v.is_nan()) {
            assert!(v <= peak + 1e-9, "pixel {v} exceeds peak {peak}");
        }
    }

    /// f32 and f64 agree to f32 precision.
    #[test]
    fn test_convolve_uv_f32_matches_f64() {
        let (n, dx) = (48usize, 2.5 / 3600.0);
        let old = Beam::from_arcsec(8.0, 6.0, 20.0).unwrap();
        let new = Beam::from_arcsec(15.0, 12.0, 20.0).unwrap();

        let img64 =
            Array2::<f64>::from_shape_fn((n, n), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 11.0);
        let img32 = img64.mapv(|x| x as f32);

        let r64 = convolve_uv(&img64, &old, &new, dx, dx, None).unwrap();
        let r32 = convolve_uv(&img32, &old, &new, dx, dx, None).unwrap();

        for (a, b) in r64.image.iter().zip(r32.image.iter()) {
            assert_eq!(a.is_nan(), b.is_nan());
            if !a.is_nan() {
                assert!(
                    (*a - *b as f64).abs() < 1e-4,
                    "f32/f64 mismatch: {a} vs {b}"
                );
            }
        }
    }

    /// A NaN block wider than the kernel stays blank inside; far pixels stay
    /// finite. (Isolated NaNs are interpolated over.)
    #[test]
    fn test_convolve_uv_propagates_nans() {
        let (n, dx) = (48usize, 2.5 / 3600.0);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(12.0, 12.0, 0.0).unwrap();

        let mut img = Array2::<f32>::from_elem((n, n), 1.0);
        for i in 0..12 {
            for j in 0..12 {
                img[(i, j)] = f32::NAN;
            }
        }

        let res = convolve_uv(&img, &old, &new, dx, dx, None).unwrap();
        assert!(res.image[(3, 3)].is_nan(), "block interior should stay NaN");
        assert!(res.image[(n - 1, n / 2)].is_finite());
    }

    /// Shared plans give the same output as per-call plans.
    #[test]
    fn test_with_plans_matches_per_call() {
        let (n, dx) = (32usize, 2.5 / 3600.0);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        let img = Array2::<f32>::from_shape_fn((n, n), |(i, j)| (i + 2 * j) as f32);

        let per_call = convolve_uv(&img, &old, &new, dx, dx, None).unwrap();

        let (pr, pc) = padded_size((n, n), &old, &new, PixelScale::cdelt(dx, dx));
        let plans = FftPlans::<f32>::new(pr, pc);
        let reused = convolve_uv(&img, &old, &new, dx, dx, Some(&plans)).unwrap();

        for (a, b) in per_call.image.iter().zip(reused.image.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "plan reuse changed output");
        }
    }

    /// `gaussft` at DC (u=v=0) equals the amplitude ratio g_ratio.
    #[test]
    fn test_gaussft_dc_equals_ratio() {
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(12.0, 10.0, 30.0).unwrap();
        let (g, ratio) = gaussft(&old, &new, &[0.0], &[0.0]);
        assert!(
            (g[0] - ratio).abs() < 1e-12,
            "DC {} != ratio {}",
            g[0],
            ratio
        );
        assert!(ratio > 1.0, "larger target beam should have ratio > 1");
    }
}
