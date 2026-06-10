/// FFT-based UV-plane beam convolution.
///
/// This is a port of `racs_tools.convolve_uv.convolve` and `racs_tools.gaussft.gaussft`.
/// The "robust" mode is implemented: the FT of the convolving Gaussian is computed
/// analytically at each UV point (no kernel image needed), which handles NaNs gracefully.
use ndarray::Array2;
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
    let dx_rad = dx_deg.to_radians();
    let dy_rad = dy_deg.to_radians();
    let u_freqs = fftfreq(nrows, dx_rad); // shape (nrows,)
    let v_freqs = fftfreq(ncols, dy_rad); // shape (ncols,)

    // Compute the UV-plane filter g_final[i, j] (shape nrows × ncols).
    let (g_final, g_ratio) = gaussft(old_beam, new_beam, &u_freqs, &v_freqs);

    // FFT the image.
    let im_f = fft2(&clean_image, nrows, ncols);

    // Multiply element-wise.
    let convolved_f: Vec<Complex<f64>> = im_f
        .iter()
        .zip(g_final.iter())
        .map(|(imf, gf)| imf * gf)
        .collect();

    // Inverse FFT and take real part.
    let im_conv_flat = ifft2(&convolved_f, nrows, ncols);

    // NaN propagation.
    let out_flat: Vec<f32> = if let Some(mask) = nan_mask {
        let mask_f = fft2(&mask, nrows, ncols);
        let mask_conv_f: Vec<Complex<f64>> = mask_f
            .iter()
            .zip(g_final.iter())
            .map(|(mf, gf)| mf * gf)
            .collect();
        let mask_conv = ifft2(&mask_conv_f, nrows, ncols);
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
/// `u_freqs` has length `nrows`, `v_freqs` has length `ncols`.
/// Returns `(g_final, g_ratio)` where `g_final` has length `nrows * ncols`
/// stored in row-major order.
pub fn gaussft(
    old_beam: &Beam,
    new_beam: &Beam,
    u_freqs: &[f64],
    v_freqs: &[f64],
) -> (Vec<Complex<f64>>, f64) {
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
    let mut g_final = vec![Complex::<f64>::new(0.0, 0.0); nrows * ncols];

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

            let val = g_ratio * (g_arg - dg_arg).exp();
            g_final[i * ncols + j] = Complex::new(val, 0.0);
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
/// Returns complex spectrum in the same layout.
fn fft2(data: &[f64], nrows: usize, ncols: usize) -> Vec<Complex<f64>> {
    let mut buf: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    let mut planner = FftPlanner::new();

    // Row-wise FFT.
    let row_fft = planner.plan_fft_forward(ncols);
    for row in buf.chunks_mut(ncols) {
        row_fft.process(row);
    }

    // Column-wise FFT (gather, process, scatter).
    let col_fft = planner.plan_fft_forward(nrows);
    let mut col_buf = vec![Complex::new(0.0, 0.0); nrows];
    for j in 0..ncols {
        for i in 0..nrows {
            col_buf[i] = buf[i * ncols + j];
        }
        col_fft.process(&mut col_buf);
        for i in 0..nrows {
            buf[i * ncols + j] = col_buf[i];
        }
    }

    buf
}

/// 2D inverse FFT (un-normalised → divide by N = nrows*ncols).
/// Returns the real part of the result.
fn ifft2(spectrum: &[Complex<f64>], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut buf = spectrum.to_vec();

    let mut planner = FftPlanner::new();

    // Row-wise inverse FFT.
    let row_ifft = planner.plan_fft_inverse(ncols);
    for row in buf.chunks_mut(ncols) {
        row_ifft.process(row);
    }

    // Column-wise inverse FFT.
    let col_ifft = planner.plan_fft_inverse(nrows);
    let mut col_buf = vec![Complex::new(0.0, 0.0); nrows];
    for j in 0..ncols {
        for i in 0..nrows {
            col_buf[i] = buf[i * ncols + j];
        }
        col_ifft.process(&mut col_buf);
        for i in 0..nrows {
            buf[i * ncols + j] = col_buf[i];
        }
    }

    let norm = (nrows * ncols) as f64;
    buf.iter().map(|c| c.re / norm).collect()
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
    fn test_fft2_ifft2_roundtrip() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (nrows, ncols) = (3, 3);
        let spectrum = fft2(&data, nrows, ncols);
        let recovered = ifft2(&spectrum, nrows, ncols);
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
