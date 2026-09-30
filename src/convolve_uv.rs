//! FFT-based UV-plane beam convolution.
//!
//! This is a port of `racs_tools.convolve_uv.convolve` and `racs_tools.gaussft.gaussft`.
//! The "robust" mode is implemented: the FT of the convolving Gaussian is computed
//! analytically at each UV point (no kernel image needed), which handles NaNs gracefully.
//!
//! The convolution is generic over the floating-point element type [`FftFloat`]
//! (`f32` or `f64`): the transforms run in the **same precision as the input
//! image**, so f32 data (the common radio-astronomy case) is transformed in f32
//! — roughly half the memory traffic and compute of an f64 transform — while
//! genuine f64 data is honoured exactly. Plans and scratch buffers are reused
//! across calls via [`FftPlans`], which matters when convolving every channel of
//! a cube at the same image size.
//!
//! Peak memory is about one image on top of the input (none at all for an owned,
//! NaN-free plane — see [`convolve_uv_owned_with_plans`]): the real FFT runs in
//! place, the filter is evaluated on the fly rather than stored, and the NaN mask
//! is reduced to a bitset before the image is transformed.
use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, cast};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftNum, FftPlanner, num_complex::Complex};
use thiserror::Error;

use crate::beam::{Beam, gauss_factor};

#[derive(Debug, Error)]
pub enum ConvolveError {
    #[error("image is entirely NaN")]
    AllNaN,
    #[error("beam larger than cutoff — image blanked")]
    AboveCutoff,
}

/// Floating-point element type a convolution can run in: `f32` or `f64`.
///
/// `FftNum` makes it usable by `rustfft`/`realfft`; `Float` provides the NaN
/// handling and numeric casts the convolution needs. Sealed in practice to the
/// two IEEE types `rustfft` supports.
pub trait FftFloat: FftNum + Float {}
impl FftFloat for f32 {}
impl FftFloat for f64 {}

/// Cast an `f64` to `T`, saturating to ±∞ when the value is outside `T`'s finite
/// range instead of panicking.
///
/// [`num_traits::cast`] returns `None` for out-of-range values (e.g. an `f64`
/// filter coefficient or flux factor exceeding `f32::MAX`); saturating to
/// infinity matches the old `value as f32` behaviour and lets the convolution
/// produce an inf/NaN pixel rather than aborting the whole run mid-cube.
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
    /// Convolved image (NaNs propagated from input).
    pub image: Array2<T>,
    /// Flux scaling factor for Jy/beam.
    pub scaling_factor: f64,
}

/// Cached FFT plans for a fixed `(nrows, ncols)` image size.
///
/// `rustfft`/`realfft` plans are `Arc<dyn …>` and `Send + Sync`, so a single
/// `FftPlans` can be built once per cube and **shared by reference** across the
/// rayon workers that convolve channels in parallel — each call brings its own
/// scratch, so only the (immutable) plans are shared. Building plans is the
/// expensive part of a transform; reusing them avoids re-planning on every
/// channel.
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
    /// Plan the forward/inverse real (row) and complex (column) FFTs for an
    /// `nrows × ncols` image. Reuse this across all channels of a cube.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let mut rplanner = RealFftPlanner::<T>::new();
        let r2c = rplanner.plan_fft_forward(ncols);
        let c2r = rplanner.plan_fft_inverse(ncols);

        let mut cplanner = FftPlanner::<T>::new();
        let col_fwd = cplanner.plan_fft_forward(nrows);
        let col_inv = cplanner.plan_fft_inverse(nrows);

        Self {
            nrows,
            ncols,
            nhalf: ncols / 2 + 1,
            r2c,
            c2r,
            col_fwd,
            col_inv,
        }
    }

    /// Image dimensions `(nrows, ncols)` these plans were built for.
    pub fn dim(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// The image handed to the convolution: either borrowed, in which case one
/// working buffer is allocated for the output, or owned, in which case the
/// transform runs in the caller's own buffer and nothing image-sized is
/// allocated at all (for an image without NaNs).
pub(crate) enum PlaneInput<'a, T> {
    Borrowed(ArrayView2<'a, T>),
    Owned(Array2<T>),
}

impl<T: FftFloat> PlaneInput<'_, T> {
    fn view(&self) -> ArrayView2<'_, T> {
        match self {
            PlaneInput::Borrowed(v) => v.view(),
            PlaneInput::Owned(a) => a.view(),
        }
    }

    fn into_owned(self) -> Array2<T> {
        match self {
            PlaneInput::Borrowed(v) => v.to_owned(),
            PlaneInput::Owned(a) => a,
        }
    }

    /// Row-major working buffer holding the image with NaNs zero-filled. A new
    /// buffer is allocated with room for `capacity` values, so growing it to
    /// hold the spectrum never reallocates.
    fn into_zero_filled_vec(self, capacity: usize) -> Vec<T> {
        let zero_nan = |x: T| if x.is_nan() { T::zero() } else { x };
        match self {
            PlaneInput::Owned(a) if a.is_standard_layout() => {
                let (mut buf, offset) = a.into_raw_vec_and_offset();
                debug_assert_eq!(offset.unwrap_or(0), 0);
                for x in buf.iter_mut() {
                    *x = zero_nan(*x);
                }
                buf
            }
            other => {
                let mut buf = Vec::with_capacity(capacity);
                buf.extend(other.view().iter().map(|&x| zero_nan(x)));
                buf
            }
        }
    }
}

/// Extra flux scaling applied to the output on top of the filter's own DC gain
/// (`g_ratio`), folded into the filter so it costs no extra pass over the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputGain {
    /// Leave the output as the filter produced it (what [`convolve_uv`] returns).
    Unit,
    /// Multiply by `g_ratio` once more (Jy/beam).
    Ratio,
    /// Divide the filter's `g_ratio` back out (Kelvin).
    InverseRatio,
}

/// Convolve `image` from `old_beam` to `new_beam` in the UV plane.
///
/// `dx_deg` / `dy_deg` are the pixel sizes in degrees (FITS |CDELT1|, |CDELT2|).
/// `cutoff_arcsec` blanks images whose current beam exceeds this size.
///
/// The returned [`ConvolutionResult::scaling_factor`] is `√(Ω_new/Ω_old)`; see
/// [`crate::smooth::smooth`] for how this becomes the Jy/beam or Kelvin factor.
///
/// This builds FFT plans for the image size on each call. To convolve many
/// images of the same size (e.g. cube channels), build an [`FftPlans`] once and
/// call [`convolve_uv_with_plans`] to reuse it.
///
/// # Examples
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
/// // √(Ω_new/Ω_old) = √4 = 2 for a doubling of both axes.
/// assert!((result.scaling_factor - 2.0).abs() < 1e-9);
/// assert_eq!(result.image.dim(), (64, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn convolve_uv<T: FftFloat>(
    image: &Array2<T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    let (nrows, ncols) = image.dim();
    let plans = FftPlans::<T>::new(nrows, ncols);
    convolve_uv_with_plans(
        image,
        old_beam,
        new_beam,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
        &plans,
    )
}

/// Like [`convolve_uv`], but reuses pre-built [`FftPlans`] instead of planning
/// per call. `plans` must have been built for `image`'s dimensions.
pub fn convolve_uv_with_plans<T: FftFloat>(
    image: &Array2<T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    plans: &FftPlans<T>,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    convolve_plane(
        PlaneInput::Borrowed(image.view()),
        old_beam,
        new_beam,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
        OutputGain::Unit,
        plans,
    )
}

/// Like [`convolve_uv_with_plans`], but takes ownership of `image` and
/// transforms in its buffer, so the only image-sized allocation is the one the
/// NaN mask needs (none at all for an image without NaNs). Use this when the
/// input plane is not needed afterwards, e.g. a cube channel just read from disk.
pub fn convolve_uv_owned_with_plans<T: FftFloat>(
    image: Array2<T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    plans: &FftPlans<T>,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    convolve_plane(
        PlaneInput::Owned(image),
        old_beam,
        new_beam,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
        OutputGain::Unit,
        plans,
    )
}

/// The convolution behind every public entry point.
///
/// Peak memory is kept to roughly one image on top of the input: the real FFT
/// runs in place in a single buffer (the half spectrum of an `nrows × ncols`
/// real image is `nrows × (ncols/2 + 1)` complex values, which fits in that
/// buffer with only two extra values per row), the filter is evaluated on the
/// fly rather than stored, and the NaN mask is convolved *first*, in its own
/// buffer that is freed before the image's is needed, leaving only a bitset of
/// the pixels to blank.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve_plane<T: FftFloat>(
    image: PlaneInput<'_, T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    gain: OutputGain,
    plans: &FftPlans<T>,
) -> Result<ConvolutionResult<T>, ConvolveError> {
    // Cutoff check.
    if let Some(cutoff) = cutoff_arcsec
        && old_beam.major_arcsec() > cutoff
    {
        return Err(ConvolveError::AboveCutoff);
    }

    // Beams identical → no-op with unit scaling.
    if old_beam.approx_eq(new_beam) {
        return Ok(ConvolutionResult {
            image: image.into_owned(),
            scaling_factor: 1.0,
        });
    }

    let (nrows, ncols) = image.view().dim();
    assert_eq!(
        plans.dim(),
        (nrows, ncols),
        "FftPlans built for {:?} but image is {:?}",
        plans.dim(),
        (nrows, ncols)
    );

    let nan_count = image.view().iter().filter(|x| x.is_nan()).count();

    // All-NaN fast path.
    if nan_count == nrows * ncols {
        // Compute the convolving beam (new² - old² in quadrature) and flux scaling.
        let conv_beam = new_beam.deconvolve_or_zero(old_beam);
        let (fac, ..) = gauss_factor(
            &conv_beam,
            old_beam,
            dx_deg.abs() * 3600.0,
            dy_deg.abs() * 3600.0,
        );
        return Ok(ConvolutionResult {
            image: image.into_owned(),
            scaling_factor: fac,
        });
    }

    // UV coordinates: fftfreq(n, d_rad) where d_rad = pixel_size_in_radians.
    // The data is real, so we use a real-input FFT: the column (ncols) axis only
    // needs its non-negative half, `nhalf = ncols/2 + 1` bins. We slice the full
    // `fftfreq` rather than using `rfftfreq` so the filter is evaluated at exactly
    // the frequencies the equivalent full FFT assigns to bins 0..nhalf (incl. the
    // signed Nyquist), keeping results aligned with the full-FFT port.
    let taper = UvTaper::new(old_beam, new_beam);
    let grid = UvGrid::new(
        nrows,
        ncols,
        dx_deg.to_radians(),
        dy_deg.to_radians(),
        &taper,
    );
    let npix = (nrows * ncols) as f64;
    let spectrum_len = nrows * 2 * (ncols / 2 + 1);

    // Convolve the NaN mask with the same filter to find where the blanking
    // spreads to. Done before the image so its buffer is gone by the time the
    // image's is needed.
    let blank = (nan_count > 0).then(|| {
        let mut mask: Vec<T> = Vec::with_capacity(spectrum_len);
        mask.extend(
            image
                .view()
                .iter()
                .map(|x| if x.is_nan() { T::one() } else { T::zero() }),
        );
        convolve_in_place(&mut mask, &taper, taper.ratio / npix, &grid, plans);
        // A fully-NaN-covered pixel reaches the filter's DC gain (g_ratio ≥ 1) in
        // the convolved mask; isolated NaNs stay well below 1 and are interpolated
        // over. The threshold sits just under 1 so that f32 round-off — which can
        // leave a solid-NaN interior at ~0.999 when the beams are nearly equal
        // (g_ratio ≈ 1) — does not silently un-blank a masked region.
        let blank_threshold = T::one() - cast_saturating::<T>(1e-2);
        Bitset::from_predicate(&mask, |m| m >= blank_threshold)
    });

    let out_gain = match gain {
        OutputGain::Unit => taper.ratio,
        OutputGain::Ratio => taper.ratio * taper.ratio,
        OutputGain::InverseRatio => 1.0,
    };
    let mut buf = image.into_zero_filled_vec(spectrum_len);
    convolve_in_place(&mut buf, &taper, out_gain / npix, &grid, plans);

    if let Some(blank) = blank {
        blank.for_each_set(|k| buf[k] = T::nan());
    }

    let out =
        Array2::from_shape_vec((nrows, ncols), buf).expect("shape mismatch in convolve_uv output");

    Ok(ConvolutionResult {
        image: out,
        scaling_factor: taper.ratio,
    })
}

// ── gaussft ───────────────────────────────────────────────────────────────────

/// The UV-plane filter that deconvolves one Gaussian beam and re-convolves with
/// another, reduced to the coefficients of a single quadratic form.
///
/// `racs_tools.gaussft` evaluates `g_ratio · exp(g_arg − dg_arg)`, where each
/// argument is `−2π² ((σx u_r)² + (σy v_r)²)` in coordinates rotated by that
/// beam's position angle. Expanding the rotations, each argument is a quadratic
/// form `a u² + b uv + c v²`, so their difference is too. Collapsing both into
/// one set of coefficients makes each filter value a few multiply-adds and one
/// `exp`, cheap enough to evaluate on the fly instead of storing the filter.
#[derive(Clone, Copy, Debug)]
struct UvTaper {
    a: f64,
    b: f64,
    c: f64,
    /// Amplitude ratio (= flux scaling factor = DC gain of the filter).
    ratio: f64,
}

impl UvTaper {
    fn new(old_beam: &Beam, new_beam: &Beam) -> Self {
        let deg2rad = std::f64::consts::PI / 180.0;
        let fwhm_to_sigma = 2.0 * (2.0 * 2_f64.ln()).sqrt(); // = 2*sqrt(2*ln2)

        // New beam (target) and old beam (input PSF) as Gaussian sigmas.
        let sx = new_beam.major_deg * deg2rad / fwhm_to_sigma;
        let sy = new_beam.minor_deg * deg2rad / fwhm_to_sigma;
        let sx_in = old_beam.major_deg * deg2rad / fwhm_to_sigma;
        let sy_in = old_beam.minor_deg * deg2rad / fwhm_to_sigma;

        // Amplitude ratio (= flux scaling factor).
        let g_amp = (2.0 * std::f64::consts::PI * sx * sy).sqrt();
        let dg_amp = (2.0 * std::f64::consts::PI * sx_in * sy_in).sqrt();

        // (σx u_r)² + (σy v_r)² with u_r = u cos θ − v sin θ, v_r = u sin θ + v cos θ.
        let form = |sx: f64, sy: f64, pa_deg: f64| {
            let (s, c) = (pa_deg * deg2rad).sin_cos();
            let (sx2, sy2) = (sx * sx, sy * sy);
            (
                sx2 * c * c + sy2 * s * s,
                2.0 * c * s * (sy2 - sx2),
                sx2 * s * s + sy2 * c * c,
            )
        };
        let (a, b, c) = form(sx, sy, new_beam.pa_deg);
        let (a_in, b_in, c_in) = form(sx_in, sy_in, old_beam.pa_deg);
        let k = -2.0 * std::f64::consts::PI * std::f64::consts::PI;

        Self {
            a: k * (a - a_in),
            b: k * (b - b_in),
            c: k * (c - c_in),
            ratio: g_amp / dg_amp,
        }
    }

    /// Filter value at `(u, v)`, including the DC gain `ratio`.
    fn at(&self, u: f64, v: f64) -> f64 {
        self.ratio * (self.a * u * u + self.b * u * v + self.c * v * v).exp()
    }
}

/// Compute the UV-plane filter that deconvolves `old_beam` and re-convolves with
/// `new_beam`. Direct port of `racs_tools.gaussft.gaussft`.
///
/// `u_freqs` has length `nrows`, `v_freqs` has length `ncols` (or `nhalf` for a
/// half-spectrum / real-FFT layout). The filter is real-valued, so it is returned
/// as `Vec<f64>` of length `nrows * v_freqs.len()` in row-major order.
///
/// The convolution itself never materialises this array — it evaluates the
/// same filter on the fly — so this is for inspection and testing.
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

// ── FFT helpers ───────────────────────────────────────────────────────────────

/// numpy-compatible `fftfreq(n, d)`.
///
/// For even n the Nyquist bin (index n/2) is listed as negative, matching numpy.
///
/// # Examples
///
/// ```
/// use convolve_rs::fftfreq;
///
/// assert_eq!(fftfreq(4, 1.0), vec![0.0, 0.25, -0.5, -0.25]);
/// assert_eq!(fftfreq(5, 1.0), vec![0.0, 0.2, 0.4, -0.4, -0.2]);
/// ```
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    let m = n.div_ceil(2); // ceiling(n/2): positive-frequency count
    let mut freqs = vec![0.0_f64; n];
    for (i, freq) in freqs.iter_mut().enumerate().take(m) {
        *freq = i as f64 * val;
    }
    for (i, freq) in freqs.iter_mut().enumerate().take(n).skip(m) {
        *freq = (i as f64 - n as f64) * val;
    }
    freqs
}

/// Columns transformed together in the column pass. The half spectrum is stored
/// row-major, so a single column is `nrows` values each a full row apart; gathering
/// a block of adjacent columns at once reads whole cache lines instead of one
/// value per line, which is what makes the column pass cheap on wide images.
const COL_BLOCK: usize = 16;

/// The UV coordinates of the half spectrum, with the parts of the filter's
/// exponent that depend on only one of them precomputed.
struct UvGrid {
    /// `u` for each of the `nrows` rows.
    u: Vec<f64>,
    /// `a·u²` for each row.
    au2: Vec<f64>,
    /// `v` for each of the `nhalf` columns of the half spectrum.
    v: Vec<f64>,
}

impl UvGrid {
    fn new(nrows: usize, ncols: usize, dx_rad: f64, dy_rad: f64, taper: &UvTaper) -> Self {
        let u = fftfreq(nrows, dx_rad);
        let au2 = u.iter().map(|&u| taper.a * u * u).collect();
        let mut v = fftfreq(ncols, dy_rad);
        v.truncate(ncols / 2 + 1);
        Self { u, au2, v }
    }
}

/// A packed set of pixel indices (one bit per pixel).
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

/// View a buffer of reals as the complex values it stores in pairs.
fn as_complex<T: FftFloat>(re: &[T]) -> &[Complex<T>] {
    assert!(re.len().is_multiple_of(2));
    // SAFETY: `Complex<T>` is `#[repr(C)] { re: T, im: T }`, so it has the size of
    // two `T`s and the alignment of one, and any pair of initialised `T`s is a
    // valid value. The length is halved, so the view covers exactly `re`.
    unsafe { std::slice::from_raw_parts(re.as_ptr().cast(), re.len() / 2) }
}

/// Mutable form of [`as_complex`].
fn as_complex_mut<T: FftFloat>(re: &mut [T]) -> &mut [Complex<T>] {
    assert!(re.len().is_multiple_of(2));
    // SAFETY: as for `as_complex`; the exclusive borrow of `re` is carried over.
    unsafe { std::slice::from_raw_parts_mut(re.as_mut_ptr().cast(), re.len() / 2) }
}

/// Convolve the real `nrows × ncols` image held row-major in `buf` with the
/// UV-plane filter, in place. The filter applied is `gain · exp(…)`: the shape of
/// [`UvTaper`] with unit DC gain, times `gain`.
///
/// The buffer is grown by two values per row to hold the half spectrum, the
/// forward real FFT runs in place, the filter is applied during the column
/// pass, and the inverse writes the image back over the spectrum before the
/// buffer is shrunk to the image again. `gain` must include the `1/N`
/// normalisation of the inverse FFT.
fn convolve_in_place<T: FftFloat>(
    buf: &mut Vec<T>,
    taper: &UvTaper,
    gain: f64,
    grid: &UvGrid,
    plans: &FftPlans<T>,
) {
    let (nrows, ncols, nhalf) = (plans.nrows, plans.ncols, plans.nhalf);
    debug_assert_eq!(buf.len(), nrows * ncols);
    // A large buffer is its own mapping, so growing it this little is a remap
    // rather than a copy.
    buf.resize(nrows * 2 * nhalf, T::zero());

    forward_rows_in_place(plans, buf);
    filter_columns(plans, as_complex_mut(buf), taper, gain, grid);
    inverse_rows_in_place(plans, buf);

    buf.truncate(nrows * ncols);
}

/// Row-wise real→complex FFT, in place.
///
/// On entry the first `nrows · ncols` values of `buf` are the image, row-major;
/// on exit `buf` holds the `nrows × nhalf` half spectrum, row-major. Spectrum row
/// `i` starts at real offset `2·nhalf·i ≥ ncols·i`, so writing it can only
/// overwrite image rows `≥ i`; going from the last row to the first, those have
/// all been transformed already (row `i` itself is copied out first).
fn forward_rows_in_place<T: FftFloat>(plans: &FftPlans<T>, buf: &mut [T]) {
    let (nrows, ncols, nhalf) = (plans.nrows, plans.ncols, plans.nhalf);
    let mut scratch = plans.r2c.make_scratch_vec();
    let mut inrow = plans.r2c.make_input_vec();
    for i in (0..nrows).rev() {
        inrow.copy_from_slice(&buf[i * ncols..(i + 1) * ncols]);
        let out = as_complex_mut(&mut buf[2 * nhalf * i..2 * nhalf * (i + 1)]);
        plans
            .r2c
            .process_with_scratch(&mut inrow, out, &mut scratch)
            .expect("r2c FFT");
    }
}

/// Row-wise complex→real FFT, in place: the inverse of [`forward_rows_in_place`].
///
/// Image row `i` ends at real offset `ncols·(i + 1) ≤ 2·nhalf·(i + 1)`, so writing
/// it can only overwrite spectrum rows `≤ i`; going from the first row to the
/// last, those have all been transformed already.
fn inverse_rows_in_place<T: FftFloat>(plans: &FftPlans<T>, buf: &mut [T]) {
    let (nrows, ncols, nhalf) = (plans.nrows, plans.ncols, plans.nhalf);
    let mut scratch = plans.c2r.make_scratch_vec();
    let mut inrow = plans.c2r.make_input_vec();
    let even = ncols.is_multiple_of(2);
    for i in 0..nrows {
        inrow.copy_from_slice(as_complex(&buf[2 * nhalf * i..2 * nhalf * (i + 1)]));
        // c2r requires the DC (and, for even ncols, Nyquist) bins to be purely
        // real; they are up to rounding, so zero the imaginary parts explicitly.
        inrow[0].im = T::zero();
        if even {
            inrow[nhalf - 1].im = T::zero();
        }
        plans
            .c2r
            .process_with_scratch(
                &mut inrow,
                &mut buf[i * ncols..(i + 1) * ncols],
                &mut scratch,
            )
            .expect("c2r FFT");
    }
}

/// Column-wise forward FFT, filter, and inverse FFT of the `nrows × nhalf` half
/// spectrum, a block of [`COL_BLOCK`] columns at a time.
///
/// Each block is gathered into a contiguous tile once, transformed, multiplied
/// by `gain ·` the filter (evaluated on the fly, never stored), transformed back
/// and scattered, so the spectrum is traversed once rather than three times.
fn filter_columns<T: FftFloat>(
    plans: &FftPlans<T>,
    spectrum: &mut [Complex<T>],
    taper: &UvTaper,
    gain: f64,
    grid: &UvGrid,
) {
    let (nrows, nhalf) = (plans.nrows, plans.nhalf);
    let zero = Complex::new(T::zero(), T::zero());
    let scratch_len = plans
        .col_fwd
        .get_inplace_scratch_len()
        .max(plans.col_inv.get_inplace_scratch_len());
    let mut scratch = vec![zero; scratch_len];
    let mut tile = vec![zero; COL_BLOCK * nrows];

    for j0 in (0..nhalf).step_by(COL_BLOCK) {
        let width = COL_BLOCK.min(nhalf - j0);
        let tile = &mut tile[..width * nrows];

        // Gather: tile row k is spectrum column j0 + k.
        for (i, row) in spectrum.chunks_exact(nhalf).enumerate() {
            for (k, &s) in row[j0..j0 + width].iter().enumerate() {
                tile[k * nrows + i] = s;
            }
        }

        plans.col_fwd.process_with_scratch(tile, &mut scratch);

        for (k, col) in tile.chunks_exact_mut(nrows).enumerate() {
            let v = grid.v[j0 + k];
            let (bv, cv2) = (taper.b * v, taper.c * v * v);
            for ((s, &u), &au2) in col.iter_mut().zip(&grid.u).zip(&grid.au2) {
                let g = gain * (au2 + bv * u + cv2).exp();
                *s = s.scale(cast_saturating::<T>(g));
            }
        }

        plans.col_inv.process_with_scratch(tile, &mut scratch);

        // Scatter back.
        for (i, row) in spectrum.chunks_exact_mut(nhalf).enumerate() {
            for (k, s) in row[j0..j0 + width].iter_mut().enumerate() {
                *s = tile[k * nrows + i];
            }
        }
    }
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
        buf.resize(nrows * 2 * plans.nhalf, T::zero());
        forward_rows_in_place(&plans, &mut buf);
        // Column pass with a flat unit filter: a == b == c == 0.
        let flat = UvTaper {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            ratio: 1.0,
        };
        let grid = UvGrid::new(nrows, ncols, 1.0, 1.0, &flat);
        let n = (nrows * ncols) as f64;
        filter_columns(&plans, as_complex_mut(&mut buf), &flat, 1.0 / n, &grid);
        inverse_rows_in_place(&plans, &mut buf);
        buf.truncate(nrows * ncols);
        for (a, b) in data.iter().zip(buf.iter()) {
            let (a, b) = (a.to_f64().unwrap(), b.to_f64().unwrap());
            assert!((a - b).abs() < tol, "roundtrip {nrows}x{ncols}: {a} vs {b}");
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
                let got32 = convolve_uv(&img.mapv(|x| x as f32), &old, &new, dx, dx, None)
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
            let borrowed = convolve_uv_with_plans(&img, &old, &new, dx, dx, None, &plans).unwrap();
            let owned = convolve_uv_owned_with_plans(img.clone(), &old, &new, dx, dx, None, &plans)
                .unwrap();
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
        let a = convolve_uv_with_plans(&std_layout, &old, &new, dx, dx, None, &plans).unwrap();
        let b = convolve_uv_owned_with_plans(img, &old, &new, dx, dx, None, &plans).unwrap();
        for (x, y) in a.image.iter().zip(b.image.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
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
            let (sx, sy) = (b.major_deg.to_radians() / f, b.minor_deg.to_radians() / f);
            let (s, c) = b.pa_deg.to_radians().sin_cos();
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
        let reused = convolve_uv_with_plans(&img, &old, &new, dx, dx, None, &plans).unwrap();

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
