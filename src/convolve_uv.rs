/// FFT-based UV-plane beam convolution.
///
/// This is a port of `racs_tools.convolve_uv.convolve` and `racs_tools.gaussft.gaussft`.
/// The "robust" mode is implemented: the FT of the convolving Gaussian is computed
/// analytically at each UV point (no kernel image needed), which handles NaNs gracefully.
use ndarray::Array2;
use realfft::RealFftPlanner;
use rustfft::{FftPlanner, num_complex::Complex};
use thiserror::Error;

use crate::beam::{Beam, gauss_factor};

#[derive(Debug, Error)]
pub enum ConvolveError {
    #[error("image is entirely NaN")]
    AllNaN,
    #[error("beam larger than cutoff — image blanked")]
    AboveCutoff,
}

pub struct ConvolutionResult {
    /// Convolved image (NaNs propagated from input).
    pub image: Array2<f32>,
    /// Flux scaling factor for Jy/beam.
    pub scaling_factor: f64,
}

/// Convolve `image` from `old_beam` to `new_beam` in the UV plane.
///
/// `dx_deg` / `dy_deg` are the pixel sizes in degrees (FITS |CDELT1|, |CDELT2|).
/// `cutoff_arcsec` blanks images whose current beam exceeds this size.
pub fn convolve_uv(
    image: &Array2<f32>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
) -> Result<ConvolutionResult, ConvolveError> {
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

    // All-NaN fast path.
    if image.iter().all(|x| x.is_nan()) {
        return Ok(ConvolutionResult {
            image: image.clone(),
            scaling_factor: fac,
        });
    }

    let (nrows, ncols) = image.dim();

    // Handle NaNs: zero-fill and track a mask.
    let has_nan = image.iter().any(|x| x.is_nan());
    let (clean_image, nan_mask): (Vec<f64>, Option<Vec<f64>>) = if has_nan {
        let vals: Vec<f64> = image
            .iter()
            .map(|&x| if x.is_nan() { 0.0 } else { x as f64 })
            .collect();
        let mask: Vec<f64> = image
            .iter()
            .map(|&x| if x.is_nan() { 1.0 } else { 0.0 })
            .collect();
        (vals, Some(mask))
    } else {
        let vals: Vec<f64> = image.iter().map(|&x| x as f64).collect();
        (vals, None)
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
    let (g_final, g_ratio) = gaussft(old_beam, new_beam, &u_freqs, v_freqs);

    // Forward real FFT, apply the filter in place, inverse real FFT.
    let mut im_f = rfft2(&clean_image, nrows, ncols);
    for (s, &g) in im_f.iter_mut().zip(g_final.iter()) {
        *s *= g;
    }
    let im_conv_flat = irfft2(im_f, nrows, ncols);

    // NaN propagation.
    let out_flat: Vec<f32> = if let Some(mask) = nan_mask {
        let mut mask_f = rfft2(&mask, nrows, ncols);
        for (s, &g) in mask_f.iter_mut().zip(g_final.iter()) {
            *s *= g;
        }
        let mask_conv = irfft2(mask_f, nrows, ncols);
        im_conv_flat
            .iter()
            .zip(mask_conv.iter())
            .map(|(&v, &m)| if m >= 1.0 { f32::NAN } else { v as f32 })
            .collect()
    } else {
        im_conv_flat.iter().map(|&v| v as f32).collect()
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
/// large image sizes.
fn rfft2(data: &[f64], nrows: usize, ncols: usize) -> Vec<Complex<f64>> {
    let nhalf = ncols / 2 + 1;

    // Row-wise real→complex FFT.
    let mut rplanner = RealFftPlanner::<f64>::new();
    let r2c = rplanner.plan_fft_forward(ncols);
    let mut scratch = r2c.make_scratch_vec();
    let mut inrow = r2c.make_input_vec();
    let mut spectrum = vec![Complex::new(0.0, 0.0); nrows * nhalf];
    for (i, chunk) in data.chunks(ncols).enumerate() {
        inrow.copy_from_slice(chunk);
        r2c.process_with_scratch(
            &mut inrow,
            &mut spectrum[i * nhalf..(i + 1) * nhalf],
            &mut scratch,
        )
        .expect("r2c FFT");
    }

    // Column-wise complex FFT over the `nhalf` columns (gather, process, scatter).
    let col_fft = FftPlanner::new().plan_fft_forward(nrows);
    let mut col_buf = vec![Complex::new(0.0, 0.0); nrows];
    for j in 0..nhalf {
        for i in 0..nrows {
            col_buf[i] = spectrum[i * nhalf + j];
        }
        col_fft.process(&mut col_buf);
        for i in 0..nrows {
            spectrum[i * nhalf + j] = col_buf[i];
        }
    }

    spectrum
}

/// 2D inverse of [`rfft2`] (un-normalised → divide by N = nrows*ncols).
/// Consumes the half `nrows × nhalf` spectrum and returns the real nrows×ncols image.
fn irfft2(mut spectrum: Vec<Complex<f64>>, nrows: usize, ncols: usize) -> Vec<f64> {
    let nhalf = ncols / 2 + 1;

    // Column-wise inverse complex FFT over the `nhalf` columns.
    let col_ifft = FftPlanner::new().plan_fft_inverse(nrows);
    let mut col_buf = vec![Complex::new(0.0, 0.0); nrows];
    for j in 0..nhalf {
        for i in 0..nrows {
            col_buf[i] = spectrum[i * nhalf + j];
        }
        col_ifft.process(&mut col_buf);
        for i in 0..nrows {
            spectrum[i * nhalf + j] = col_buf[i];
        }
    }

    // Row-wise complex→real FFT.
    let mut rplanner = RealFftPlanner::<f64>::new();
    let c2r = rplanner.plan_fft_inverse(ncols);
    let mut scratch = c2r.make_scratch_vec();
    let mut inrow = c2r.make_input_vec();
    let mut out = vec![0.0_f64; nrows * ncols];
    let even = ncols % 2 == 0;
    for i in 0..nrows {
        inrow.copy_from_slice(&spectrum[i * nhalf..(i + 1) * nhalf]);
        // c2r requires the DC (and, for even ncols, Nyquist) bins to be purely
        // real; they are up to rounding, so zero the imaginary parts explicitly.
        inrow[0].im = 0.0;
        if even {
            inrow[nhalf - 1].im = 0.0;
        }
        c2r.process_with_scratch(
            &mut inrow,
            &mut out[i * ncols..(i + 1) * ncols],
            &mut scratch,
        )
        .expect("c2r FFT");
    }

    let norm = (nrows * ncols) as f64;
    for v in out.iter_mut() {
        *v /= norm;
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
        let spectrum = rfft2(&data, nrows, ncols);
        let recovered = irfft2(spectrum, nrows, ncols);
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_convolve_uv_no_change_when_beams_equal() {
        let beam = Beam::new(10.0 / 3600.0, 10.0 / 3600.0, 0.0).unwrap();
        let img = Array2::from_elem((16, 16), 1.0_f32);
        let result = convolve_uv(&img, &beam, &beam, 2.5 / 3600.0, 2.5 / 3600.0, None).unwrap();
        assert!((result.scaling_factor - 1.0).abs() < 1e-10);
    }
}
