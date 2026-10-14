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

/// Convolve `image` from `old_beam` to `new_beam` in the UV plane.
///
/// `dx_deg` and `dy_deg` are the pixel sizes (CDELT1, CDELT2); their sign is
/// ignored. Pass `plans` to reuse FFT plans across images of the same size.
/// An owned image is transformed in its own buffer.
///
/// # Errors
///
/// If `old_beam` has zero size, `new_beam` cannot be reached from it, or
/// `plans` were built for another image size.
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
    convolve_with_gain(
        image.into(),
        old_beam,
        new_beam,
        (dx_deg, dy_deg),
        plans,
        |ratio| ratio,
    )
}

/// [`convolve_uv`] with the filter's DC gain set to `gain(g_ratio)`, where
/// `g_ratio = √(Ω_new/Ω_old)`. Folding the flux scaling into the filter saves
/// a pass over the image.
pub(crate) fn convolve_with_gain<T: FftFloat>(
    image: CowArray<'_, T, Ix2>,
    old_beam: &Beam,
    new_beam: &Beam,
    (dx_deg, dy_deg): (f64, f64),
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

    let dim = image.dim();
    let local_plans;
    let plans = match plans {
        Some(p) => p,
        None => {
            local_plans = FftPlans::new(dim.0, dim.1);
            &local_plans
        }
    };
    if plans.dim() != dim {
        return Err(ConvolveError::PlanSize {
            plans: plans.dim(),
            image: dim,
        });
    }

    let taper = UvTaper::new(old_beam, new_beam);
    let nan_count = image.iter().filter(|x| x.is_nan()).count();
    if nan_count == image.len() {
        return Ok(ConvolutionResult {
            image: image.into_owned(),
            scaling_factor: taper.ratio,
        });
    }

    // Rows run along y (CDELT2), columns along x (CDELT1). Only the magnitudes
    // matter: passing a negative CDELT1 to fftfreq would mirror the PA.
    let grid = UvGrid::new(
        dim,
        dy_deg.abs().to_radians(),
        dx_deg.abs().to_radians(),
        &taper,
    );
    // Done before the image is unpacked, so the mask's buffer is freed first.
    let blank = if nan_count > 0 {
        Some(nan_spread(image.view(), &taper, &grid, plans)?)
    } else {
        None
    };

    let mut buf = zero_filled(image, plans);
    let npix = (dim.0 * dim.1) as f64;
    convolve_in_place(&mut buf, &taper, gain(taper.ratio) / npix, &grid, plans)?;
    if let Some(blank) = blank {
        blank.for_each_set(|k| buf[k] = T::nan());
    }

    Ok(ConvolutionResult {
        image: Array2::from_shape_vec(dim, buf).map_err(|e| ConvolveError::Fft(e.to_string()))?,
        scaling_factor: taper.ratio,
    })
}

/// Pixels to blank: where the NaN mask, convolved like the image, reaches
/// 0.99. racs_tools uses 1; with nearly equal beams (g_ratio ≈ 1) a solid NaN
/// region sits at 1 minus round-off and the NaN pixels on its edge just under
/// 1, and both must stay blank.
fn nan_spread<T: FftFloat>(
    image: ArrayView2<'_, T>,
    taper: &UvTaper,
    grid: &UvGrid,
    plans: &FftPlans<T>,
) -> Result<Bitset, ConvolveError> {
    let mut mask = Vec::with_capacity(spectrum_len(plans));
    mask.extend(
        image
            .iter()
            .map(|x| if x.is_nan() { T::one() } else { T::zero() }),
    );
    let npix = (plans.nrows * plans.ncols) as f64;
    convolve_in_place(&mut mask, taper, taper.ratio / npix, grid, plans)?;
    let threshold = T::one() - cast_saturating::<T>(1e-2);
    Ok(Bitset::from_predicate(&mask, |m| m >= threshold))
}

fn spectrum_len<T: FftNum>(plans: &FftPlans<T>) -> usize {
    plans.nrows * 2 * plans.nhalf
}

/// The image row-major with NaNs set to zero. An owned standard-layout image
/// is reused in place.
fn zero_filled<T: FftFloat>(image: CowArray<'_, T, Ix2>, plans: &FftPlans<T>) -> Vec<T> {
    let zero_nan = |x: T| if x.is_nan() { T::zero() } else { x };
    if image.is_view() || !image.is_standard_layout() {
        let mut buf = Vec::with_capacity(spectrum_len(plans));
        buf.extend(image.iter().map(|&x| zero_nan(x)));
        return buf;
    }
    let image = image.into_owned();
    let len = image.len();
    let (mut buf, offset) = image.into_raw_vec_and_offset();
    // A sliced array keeps its parent's buffer; in standard layout its
    // elements are the `len` values starting at `offset`.
    buf.drain(..offset.unwrap_or(0));
    buf.truncate(len);
    for x in buf.iter_mut() {
        *x = zero_nan(*x);
    }
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
struct UvGrid {
    u: Vec<f64>,
    au2: Vec<f64>,
    /// `v` for the `ncols/2 + 1` columns of the half spectrum. Taken from the
    /// full `fftfreq` rather than `rfftfreq`, so the Nyquist bin keeps the
    /// sign the full transform gives it.
    v: Vec<f64>,
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
        Self { u, au2, v }
    }
}

/// One bit per pixel.
struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    fn from_predicate<T: Copy>(values: &[T], pred: impl Fn(T) -> bool) -> Self {
        let words = values
            .chunks(64)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u64, |w, (b, &x)| w | ((pred(x) as u64) << b))
            })
            .collect();
        Self { words }
    }

    fn for_each_set(&self, mut f: impl FnMut(usize)) {
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                f(w * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
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
                    let v = grid.v[j0 + k];
                    let (bv, cv2) = (taper.b * v, taper.c * v * v);
                    for ((s, &u), &au2) in col.iter_mut().zip(&grid.u).zip(&grid.au2) {
                        let g = gain * (au2 + bv * u + cv2).exp();
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

    #[test]
    fn test_fftfreq() {
        // Match numpy: fftfreq(4, 1) = [0, 0.25, -0.5, -0.25]
        let f = fftfreq(4, 1.0);
        let expected = [0.0, 0.25, -0.5, -0.25];
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "got {a}, want {b}");
        }
    }

    /// Forward then inverse in-place transform (no filter) recovers the image,
    /// for even and odd widths (the Nyquist handling differs) and both precisions.
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

    /// Straightforward reference: full complex 2-D FFT of the NaN-filled image,
    /// multiply by `gaussft` on the full grid, inverse, take the real part, and
    /// blank where the convolved NaN mask reaches the threshold. This is the
    /// algorithm as `racs_tools` first wrote it, before any memory saving.
    fn reference(image: &Array2<f64>, old: &Beam, new: &Beam, dx: f64) -> Array2<f64> {
        let (nrows, ncols) = image.dim();
        let u = fftfreq(nrows, dx.to_radians());
        let v = fftfreq(ncols, dx.to_radians());
        let (g, _) = gaussft(old, new, &u, &v);
        let mut planner = FftPlanner::<f64>::new();
        let (row_fwd, col_fwd) = (
            planner.plan_fft_forward(ncols),
            planner.plan_fft_forward(nrows),
        );
        let (row_inv, col_inv) = (
            planner.plan_fft_inverse(ncols),
            planner.plan_fft_inverse(nrows),
        );
        let fft2 = |z: &mut [Complex<f64>], rows: &dyn Fft<f64>, cols: &dyn Fft<f64>| {
            rows.process(z);
            let mut col = vec![Complex::new(0.0, 0.0); nrows];
            for j in 0..ncols {
                (0..nrows).for_each(|i| col[i] = z[i * ncols + j]);
                cols.process(&mut col);
                (0..nrows).for_each(|i| z[i * ncols + j] = col[i]);
            }
        };
        let conv = |data: &dyn Fn(f64) -> f64| -> Vec<f64> {
            let mut z: Vec<Complex<f64>> =
                image.iter().map(|&x| Complex::new(data(x), 0.0)).collect();
            fft2(&mut z, &*row_fwd, &*col_fwd);
            z.iter_mut().zip(&g).for_each(|(s, &g)| *s *= g);
            fft2(&mut z, &*row_inv, &*col_inv);
            z.iter().map(|s| s.re / (nrows * ncols) as f64).collect()
        };
        let img = conv(&|x| if x.is_nan() { 0.0 } else { x });
        let mask = conv(&|x| if x.is_nan() { 1.0 } else { 0.0 });
        let out = img
            .iter()
            .zip(&mask)
            .map(|(&x, &m)| if m >= 1.0 - 1e-2 { f64::NAN } else { x })
            .collect();
        Array2::from_shape_vec((nrows, ncols), out).unwrap()
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

    /// The in-place, on-the-fly-filter, blocked-column convolution matches the
    /// naive full-FFT reference, including odd dimensions, rotated beams, NaN
    /// blanking, and widths that leave a partial final column block.
    #[test]
    fn test_convolve_uv_matches_full_fft_reference() {
        let dx = 2.5 / 3600.0;
        let old = Beam::from_arcsec(12.0, 10.0, 20.0).unwrap();
        let new = Beam::from_arcsec(18.0, 15.0, 35.0).unwrap();
        for &(r, c) in &[(64, 48), (65, 49), (40, 71), (37, 36)] {
            for nans in [false, true] {
                let img = test_image(r, c, nans);
                let expected = reference(&img, &old, &new, dx);
                let got = convolve_uv(&img, &old, &new, dx, dx, None).unwrap().image;
                let got32 = convolve_uv(img.mapv(|x| x as f32), &old, &new, dx, dx, None)
                    .unwrap()
                    .image;
                for ((&e, &g), &g32) in expected.iter().zip(got.iter()).zip(got32.iter()) {
                    assert_eq!(
                        e.is_nan(),
                        g.is_nan(),
                        "{r}x{c} nans={nans}: NaN mask differs"
                    );
                    assert_eq!(
                        e.is_nan(),
                        g32.is_nan(),
                        "{r}x{c} nans={nans}: f32 NaN mask"
                    );
                    if !e.is_nan() {
                        assert!((e - g).abs() < 1e-9, "{r}x{c} nans={nans}: {e} vs {g}");
                        assert!((e - g32 as f64).abs() < 1e-4, "{r}x{c}: f32 {e} vs {g32}");
                    }
                }
            }
        }
    }

    /// Convolving an owned plane in its own buffer gives bit-identical output to
    /// convolving a borrowed one, with and without NaNs.
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

    /// A non-standard-layout (transposed) input is convolved as the logical
    /// image, not as its memory order.
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

    /// Second moments (arcsec²) along columns (x) and rows (y) of `img`.
    fn moments(img: &Array2<f64>, dx_as: f64, dy_as: f64) -> (f64, f64) {
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

    #[test]
    fn test_wrong_plan_size_is_an_error() {
        let img = test_image(16, 16, false);
        let plans = FftPlans::<f64>::new(16, 17);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        let d = 2.0 / 3600.0;
        assert!(matches!(
            convolve_uv(&img, &old, &new, d, d, Some(&plans)),
            Err(ConvolveError::PlanSize { .. })
        ));
    }

    #[test]
    fn test_bitset() {
        let vals: Vec<u8> = (0..200).map(|k| (k % 3 == 0 || k == 199) as u8).collect();
        let bits = Bitset::from_predicate(&vals, |v| v == 1);
        let mut got = vec![];
        bits.for_each_set(|k| got.push(k));
        let want: Vec<usize> = (0..200).filter(|&k| vals[k] == 1).collect();
        assert_eq!(got, want);
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

    /// Convolving a point source yields a Gaussian whose integral equals the
    /// filter's DC gain (`scaling_factor` = g_ratio, which `convolve_uv` bakes
    /// into the image) and whose peak sits at the source pixel. Anchors the FFT
    /// path to a known answer.
    #[test]
    fn test_convolve_uv_point_source_flux_and_peak() {
        let (n, dx) = (64usize, 2.0 / 3600.0);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(12.0, 12.0, 0.0).unwrap();

        let mut img = Array2::<f64>::zeros((n, n));
        img[(n / 2, n / 2)] = 1.0;

        let res = convolve_uv(&img, &old, &new, dx, dx, None).unwrap();
        let total: f64 = res.image.iter().sum();

        // The UV filter has DC gain g_ratio (= scaling_factor), so a unit point
        // source convolves to a Gaussian whose pixels sum to that gain.
        assert!(
            (total - res.scaling_factor).abs() < 1e-6,
            "integral {total} != DC gain {}",
            res.scaling_factor
        );

        // Peak stays at the source pixel and is the image maximum.
        let peak = res.image[(n / 2, n / 2)];
        assert!(peak > 0.0);
        for &v in res.image.iter() {
            assert!(v <= peak + 1e-9, "pixel {v} exceeds peak {peak}");
        }
    }

    /// f32 and f64 convolutions of the same data must agree to f32 precision —
    /// confirms the precision-generic path is consistent.
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
            assert!(
                (*a - *b as f64).abs() < 1e-4,
                "f32/f64 mismatch: {a} vs {b}"
            );
        }
    }

    /// A solid NaN region larger than the kernel must stay blanked in the
    /// output (the convolved mask reaches the filter's DC gain ≥ 1 there), while
    /// data far from it stays finite. Isolated single NaNs are intentionally
    /// interpolated over, so the test uses a block.
    #[test]
    fn test_convolve_uv_propagates_nans() {
        let (n, dx) = (48usize, 2.5 / 3600.0);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(12.0, 12.0, 0.0).unwrap();

        let mut img = Array2::<f32>::from_elem((n, n), 1.0);
        // Blank a solid block in one corner, several kernel-widths across.
        for i in 0..12 {
            for j in 0..12 {
                img[(i, j)] = f32::NAN;
            }
        }

        let res = convolve_uv(&img, &old, &new, dx, dx, None).unwrap();
        // The interior of the blanked block stays NaN…
        assert!(res.image[(3, 3)].is_nan(), "block interior should stay NaN");
        // …while a pixel far from the block stays finite.
        assert!(res.image[(n - 1, n - 1)].is_finite());
    }

    /// Reusing one `FftPlans` across calls must give bit-identical output to the
    /// per-call planning path. Guards the Tier-0 plan-cache optimisation.
    #[test]
    fn test_with_plans_matches_per_call() {
        let (n, dx) = (32usize, 2.5 / 3600.0);
        let old = Beam::from_arcsec(6.0, 6.0, 0.0).unwrap();
        let new = Beam::from_arcsec(11.0, 9.0, 15.0).unwrap();
        let img = Array2::<f32>::from_shape_fn((n, n), |(i, j)| (i + 2 * j) as f32);

        let per_call = convolve_uv(&img, &old, &new, dx, dx, None).unwrap();

        let plans = FftPlans::<f32>::new(n, n);
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
