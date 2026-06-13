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
use std::sync::Arc;

use ndarray::Array2;
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
    // Cutoff check.
    if let Some(cutoff) = cutoff_arcsec
        && old_beam.major_arcsec() > cutoff
    {
        return Err(ConvolveError::AboveCutoff);
    }

    // Beams identical → no-op with unit scaling.
    if old_beam.approx_eq(new_beam) {
        return Ok(ConvolutionResult {
            image: image.clone(),
            scaling_factor: 1.0,
        });
    }

    // Compute the convolving beam (new² - old² in quadrature) and flux scaling.
    let conv_beam = new_beam.deconvolve_or_zero(old_beam);
    let (fac, ..) = gauss_factor(
        &conv_beam,
        old_beam,
        dx_deg.abs() * 3600.0,
        dy_deg.abs() * 3600.0,
    );

    let (nrows, ncols) = image.dim();
    assert_eq!(
        plans.dim(),
        (nrows, ncols),
        "FftPlans built for {:?} but image is {:?}",
        plans.dim(),
        (nrows, ncols)
    );

    // Single pass over the pixels: zero-fill NaNs into the working buffer, build
    // a NaN mask only if any NaNs are present, and detect the all-NaN case.
    let mut clean_image: Vec<T> = Vec::with_capacity(nrows * ncols);
    let mut nan_count = 0usize;
    for &x in image.iter() {
        if x.is_nan() {
            nan_count += 1;
            clean_image.push(T::zero());
        } else {
            clean_image.push(x);
        }
    }

    // All-NaN fast path.
    if nan_count == nrows * ncols {
        return Ok(ConvolutionResult {
            image: image.clone(),
            scaling_factor: fac,
        });
    }

    let nan_mask: Option<Vec<T>> = if nan_count > 0 {
        Some(
            image
                .iter()
                .map(|&x| if x.is_nan() { T::one() } else { T::zero() })
                .collect(),
        )
    } else {
        None
    };

    // UV coordinates: fftfreq(n, d_rad) where d_rad = pixel_size_in_radians.
    // The data is real, so we use a real-input FFT: the column (ncols) axis only
    // needs its non-negative half, `nhalf = ncols/2 + 1` bins. We slice the full
    // `fftfreq` rather than using `rfftfreq` so the filter is evaluated at exactly
    // the frequencies the equivalent full FFT assigns to bins 0..nhalf (incl. the
    // signed Nyquist), keeping results bit-for-bit aligned with the full-FFT port.
    let nhalf = ncols / 2 + 1;
    let dx_rad = dx_deg.to_radians();
    let dy_rad = dy_deg.to_radians();
    let u_freqs = fftfreq(nrows, dx_rad); // shape (nrows,)
    let v_freqs_full = fftfreq(ncols, dy_rad);
    let v_freqs = &v_freqs_full[..nhalf]; // half spectrum

    // UV-plane filter on the half spectrum (shape nrows × nhalf), real-valued.
    // `gaussft` works in f64 (analytic, cheap); cast to the image precision once
    // for the elementwise multiply against the spectrum.
    let (g_final, g_ratio) = gaussft(old_beam, new_beam, &u_freqs, v_freqs);
    let g_t: Vec<T> = g_final
        .iter()
        .map(|&g| cast::<f64, T>(g).expect("filter value out of range"))
        .collect();

    // Forward real FFT, apply the filter in place, inverse real FFT.
    let mut im_f = rfft2(plans, &clean_image);
    for (s, &g) in im_f.iter_mut().zip(g_t.iter()) {
        *s = s.scale(g);
    }
    let im_conv_flat = irfft2(plans, im_f);

    // NaN propagation.
    let out_flat: Vec<T> = if let Some(mask) = nan_mask {
        let mut mask_f = rfft2(plans, &mask);
        for (s, &g) in mask_f.iter_mut().zip(g_t.iter()) {
            *s = s.scale(g);
        }
        let mask_conv = irfft2(plans, mask_f);
        im_conv_flat
            .iter()
            .zip(mask_conv.iter())
            .map(|(&v, &m)| if m >= T::one() { T::nan() } else { v })
            .collect()
    } else {
        im_conv_flat
    };

    let out = Array2::from_shape_vec((nrows, ncols), out_flat)
        .expect("shape mismatch in convolve_uv output");

    Ok(ConvolutionResult {
        image: out,
        scaling_factor: g_ratio,
    })
}

// ── gaussft ───────────────────────────────────────────────────────────────────

/// Compute the UV-plane filter that deconvolves `old_beam` and re-convolves with
/// `new_beam`. Direct port of `racs_tools.gaussft.gaussft`.
///
/// `u_freqs` has length `nrows`, `v_freqs` has length `ncols` (or `nhalf` for a
/// half-spectrum / real-FFT layout). The filter is real-valued, so it is returned
/// as `Vec<f64>` of length `nrows * v_freqs.len()` in row-major order.
pub fn gaussft(
    old_beam: &Beam,
    new_beam: &Beam,
    u_freqs: &[f64],
    v_freqs: &[f64],
) -> (Vec<f64>, f64) {
    let deg2rad = std::f64::consts::PI / 180.0;
    let two_ln2 = 2.0 * 2_f64.ln();
    let fwhm_to_sigma = 2.0 * two_ln2.sqrt(); // = 2*sqrt(2*ln2)

    // New beam (target).
    let bmaj_rad = new_beam.major_deg * deg2rad;
    let bmin_rad = new_beam.minor_deg * deg2rad;
    let bpa_rad = new_beam.pa_deg * deg2rad;
    let sx = bmaj_rad / fwhm_to_sigma;
    let sy = bmin_rad / fwhm_to_sigma;

    // Old beam (input PSF).
    let bmaj_in_rad = old_beam.major_deg * deg2rad;
    let bmin_in_rad = old_beam.minor_deg * deg2rad;
    let bpa_in_rad = old_beam.pa_deg * deg2rad;
    let sx_in = bmaj_in_rad / fwhm_to_sigma;
    let sy_in = bmin_in_rad / fwhm_to_sigma;

    // Amplitude ratio (= flux scaling factor).
    let g_amp = (2.0 * std::f64::consts::PI * sx * sy).sqrt();
    let dg_amp = (2.0 * std::f64::consts::PI * sx_in * sy_in).sqrt();
    let g_ratio = g_amp / dg_amp;

    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    let nrows = u_freqs.len();
    let ncols = v_freqs.len();
    let mut g_final = vec![0.0_f64; nrows * ncols];

    // Pre-rotate u and v for new beam.
    let u_cos = u_freqs
        .iter()
        .map(|&u| u * bpa_rad.cos())
        .collect::<Vec<_>>();
    let u_sin = u_freqs
        .iter()
        .map(|&u| u * bpa_rad.sin())
        .collect::<Vec<_>>();
    let v_cos = v_freqs
        .iter()
        .map(|&v| v * bpa_rad.cos())
        .collect::<Vec<_>>();
    let v_sin = v_freqs
        .iter()
        .map(|&v| v * bpa_rad.sin())
        .collect::<Vec<_>>();

    // Pre-rotate u and v for old beam.
    let u_cos_in = u_freqs
        .iter()
        .map(|&u| u * bpa_in_rad.cos())
        .collect::<Vec<_>>();
    let u_sin_in = u_freqs
        .iter()
        .map(|&u| u * bpa_in_rad.sin())
        .collect::<Vec<_>>();
    let v_cos_in = v_freqs
        .iter()
        .map(|&v| v * bpa_in_rad.cos())
        .collect::<Vec<_>>();
    let v_sin_in = v_freqs
        .iter()
        .map(|&v| v * bpa_in_rad.sin())
        .collect::<Vec<_>>();

    for i in 0..nrows {
        for j in 0..ncols {
            // Rotated UV coordinates for new beam.
            let ur = u_cos[i] - v_sin[j];
            let vr = u_sin[i] + v_cos[j];
            // Rotated UV coordinates for old beam.
            let ur_in = u_cos_in[i] - v_sin_in[j];
            let vr_in = u_sin_in[i] + v_cos_in[j];

            let g_arg = -2.0 * pi2 * ((sx * ur).powi(2) + (sy * vr).powi(2));
            let dg_arg = -2.0 * pi2 * ((sx_in * ur_in).powi(2) + (sy_in * vr_in).powi(2));

            g_final[i * ncols + j] = g_ratio * (g_arg - dg_arg).exp();
        }
    }

    (g_final, g_ratio)
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

/// 2D forward FFT of real-valued data stored row-major in `data` (shape nrows×ncols).
///
/// Uses a real-input FFT along the contiguous (ncols) axis, so only the
/// non-negative half of that axis is kept: the returned spectrum is
/// `nrows × nhalf` (`nhalf = ncols/2 + 1`) complex values, row-major. This roughly
/// halves the spectrum memory versus a full complex FFT — the dominant cost at
/// large image sizes. Scratch buffers are allocated once here, not per row/column.
fn rfft2<T: FftFloat>(plans: &FftPlans<T>, data: &[T]) -> Vec<Complex<T>> {
    let (nrows, ncols, nhalf) = (plans.nrows, plans.ncols, plans.nhalf);
    let zero = Complex::new(T::zero(), T::zero());

    // Row-wise real→complex FFT.
    let mut scratch = plans.r2c.make_scratch_vec();
    let mut inrow = plans.r2c.make_input_vec();
    let mut spectrum = vec![zero; nrows * nhalf];
    for (i, chunk) in data.chunks(ncols).enumerate() {
        inrow.copy_from_slice(chunk);
        plans
            .r2c
            .process_with_scratch(
                &mut inrow,
                &mut spectrum[i * nhalf..(i + 1) * nhalf],
                &mut scratch,
            )
            .expect("r2c FFT");
    }

    // Column-wise complex FFT over the `nhalf` columns (gather, process, scatter).
    // A single scratch buffer is reused across all columns.
    let mut col_scratch = vec![zero; plans.col_fwd.get_inplace_scratch_len()];
    let mut col_buf = vec![zero; nrows];
    for j in 0..nhalf {
        for i in 0..nrows {
            col_buf[i] = spectrum[i * nhalf + j];
        }
        plans
            .col_fwd
            .process_with_scratch(&mut col_buf, &mut col_scratch);
        for i in 0..nrows {
            spectrum[i * nhalf + j] = col_buf[i];
        }
    }

    spectrum
}

/// 2D inverse of [`rfft2`] (un-normalised → divide by N = nrows*ncols).
/// Consumes the half `nrows × nhalf` spectrum and returns the real nrows×ncols image.
fn irfft2<T: FftFloat>(plans: &FftPlans<T>, mut spectrum: Vec<Complex<T>>) -> Vec<T> {
    let (nrows, ncols, nhalf) = (plans.nrows, plans.ncols, plans.nhalf);
    let zero = Complex::new(T::zero(), T::zero());

    // Column-wise inverse complex FFT over the `nhalf` columns.
    let mut col_scratch = vec![zero; plans.col_inv.get_inplace_scratch_len()];
    let mut col_buf = vec![zero; nrows];
    for j in 0..nhalf {
        for i in 0..nrows {
            col_buf[i] = spectrum[i * nhalf + j];
        }
        plans
            .col_inv
            .process_with_scratch(&mut col_buf, &mut col_scratch);
        for i in 0..nrows {
            spectrum[i * nhalf + j] = col_buf[i];
        }
    }

    // Row-wise complex→real FFT.
    let mut scratch = plans.c2r.make_scratch_vec();
    let mut inrow = plans.c2r.make_input_vec();
    let mut out = vec![T::zero(); nrows * ncols];
    let even = ncols.is_multiple_of(2);
    for i in 0..nrows {
        inrow.copy_from_slice(&spectrum[i * nhalf..(i + 1) * nhalf]);
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
                &mut out[i * ncols..(i + 1) * ncols],
                &mut scratch,
            )
            .expect("c2r FFT");
    }

    let norm = cast::<usize, T>(nrows * ncols).expect("size out of range");
    for v in out.iter_mut() {
        *v = *v / norm;
    }
    out
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

    #[test]
    fn test_rfft2_irfft2_roundtrip() {
        // Use even dimensions to exercise the Nyquist handling in irfft2.
        let data = vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let (nrows, ncols) = (4, 4);
        let plans = FftPlans::<f64>::new(nrows, ncols);
        let spectrum = rfft2(&plans, &data);
        let recovered = irfft2(&plans, spectrum);
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_rfft2_irfft2_roundtrip_f32() {
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let (nrows, ncols) = (4, 4);
        let plans = FftPlans::<f32>::new(nrows, ncols);
        let spectrum = rfft2(&plans, &data);
        let recovered = irfft2(&plans, spectrum);
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-3, "f32 roundtrip failed: {a} vs {b}");
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
