/// Radio astronomy beam (PSF) represented as a 2D Gaussian.
///
/// All stored values use FITS conventions: major/minor FWHM in degrees, PA in degrees.
/// The MIRIAD gaupar.for algorithms are used for deconvolution and convolution.
use std::fmt;

use thiserror::Error;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Clone, Copy)]
pub struct Beam {
    /// FWHM major axis in degrees (FITS BMAJ)
    pub major_deg: f64,
    /// FWHM minor axis in degrees (FITS BMIN)
    pub minor_deg: f64,
    /// Position angle in degrees East of North (FITS BPA)
    pub pa_deg: f64,
}

#[derive(Debug, Error)]
pub enum BeamError {
    #[error("beam could not be deconvolved: source beam is smaller than the PSF")]
    DeconvolveFailed,
    #[error("invalid beam: minor axis ({minor}) > major axis ({major})")]
    InvalidAxes { major: f64, minor: f64 },
    #[error("beam is not finite (NaN or infinite values)")]
    NotFinite,
}

impl Beam {
    pub fn new(major_deg: f64, minor_deg: f64, pa_deg: f64) -> Result<Self, BeamError> {
        if !major_deg.is_finite() || !minor_deg.is_finite() || !pa_deg.is_finite() {
            return Err(BeamError::NotFinite);
        }
        if minor_deg > major_deg {
            return Err(BeamError::InvalidAxes { major: major_deg, minor: minor_deg });
        }
        Ok(Self { major_deg, minor_deg, pa_deg })
    }

    pub fn from_arcsec(major_arcsec: f64, minor_arcsec: f64, pa_deg: f64) -> Result<Self, BeamError> {
        Self::new(major_arcsec / 3600.0, minor_arcsec / 3600.0, pa_deg)
    }

    pub fn zero() -> Self {
        Self { major_deg: 0.0, minor_deg: 0.0, pa_deg: 0.0 }
    }

    pub fn major_arcsec(&self) -> f64 { self.major_deg * 3600.0 }
    pub fn minor_arcsec(&self) -> f64 { self.minor_deg * 3600.0 }

    pub fn is_finite(&self) -> bool {
        self.major_deg.is_finite() && self.minor_deg.is_finite() && self.pa_deg.is_finite()
            && self.major_deg > 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.major_deg == 0.0 && self.minor_deg == 0.0
    }

    pub fn is_circular(&self, rtol: f64) -> bool {
        if self.major_deg == 0.0 { return true; }
        (self.major_deg - self.minor_deg) / self.major_deg <= rtol
    }

    pub fn area_sr(&self) -> f64 {
        let fwhm_to_area = 2.0 * std::f64::consts::PI / (8.0 * 2_f64.ln());
        self.major_deg.to_radians() * self.minor_deg.to_radians() * fwhm_to_area
    }

    /// Deconvolve `other` from `self` (i.e. `self` = result ⊛ `other`).
    ///
    /// Implements MIRIAD gaupar.for GauDfac by R. Sault.
    /// Inputs/outputs in degrees. PA returned in radians, then converted.
    pub fn deconvolve(&self, other: &Beam) -> Result<Beam, BeamError> {
        let (new_major, new_minor, new_pa_rad) =
            deconvolve_deg(self.major_deg, self.minor_deg, self.pa_deg,
                           other.major_deg, other.minor_deg, other.pa_deg, false)?;
        let pa_deg = new_pa_rad.to_degrees();
        Ok(Beam { major_deg: new_major, minor_deg: new_minor, pa_deg })
    }

    /// Like `deconvolve` but returns a zero beam on failure instead of an error.
    pub fn deconvolve_or_zero(&self, other: &Beam) -> Beam {
        match self.deconvolve(other) {
            Ok(b) => b,
            Err(_) => Beam::zero(),
        }
    }

    /// Convolve `self` with `other`. Implements MIRIAD gaupar.for GauCvl.
    pub fn convolve(&self, other: &Beam) -> Beam {
        let pa1 = self.pa_deg * DEG2RAD;
        let pa2 = other.pa_deg * DEG2RAD;

        let alpha = (self.major_deg * pa1.cos()).powi(2)
            + (self.minor_deg * pa1.sin()).powi(2)
            + (other.major_deg * pa2.cos()).powi(2)
            + (other.minor_deg * pa2.sin()).powi(2);

        let beta = (self.major_deg * pa1.sin()).powi(2)
            + (self.minor_deg * pa1.cos()).powi(2)
            + (other.major_deg * pa2.sin()).powi(2)
            + (other.minor_deg * pa2.cos()).powi(2);

        let gamma = 2.0 * ((self.minor_deg.powi(2) - self.major_deg.powi(2)) * pa1.sin() * pa1.cos()
            + (other.minor_deg.powi(2) - other.major_deg.powi(2)) * pa2.sin() * pa2.cos());

        let s = alpha + beta;
        let t = ((alpha - beta).powi(2) + gamma.powi(2)).sqrt();

        let new_major = (0.5 * (s + t)).sqrt();
        let new_minor = (0.5 * (s - t).max(0.0)).sqrt();

        let pa_rad = if (gamma.abs() + (alpha - beta).abs()).sqrt() < 1e-7 / 3600.0 {
            0.0_f64
        } else {
            0.5 * (-gamma).atan2(alpha - beta)
        };

        Beam {
            major_deg: new_major,
            minor_deg: new_minor,
            pa_deg: pa_rad.to_degrees(),
        }
    }

    /// Approximate equality with a tolerance of ~1e-10 degrees (~0.4 nanoarcsec).
    pub fn approx_eq(&self, other: &Beam) -> bool {
        const ATOL: f64 = 1e-10;
        let pa_self = self.pa_deg.rem_euclid(180.0);
        let pa_other = other.pa_deg.rem_euclid(180.0);
        let pa_eq = if self.is_circular(1e-6) {
            true
        } else {
            (pa_self - pa_other).abs() < ATOL
        };
        (self.major_deg - other.major_deg).abs() < ATOL
            && (self.minor_deg - other.minor_deg).abs() < ATOL
            && pa_eq
    }
}

impl fmt::Display for Beam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BMAJ={:.4}\" BMIN={:.4}\" BPA={:.2}°",
            self.major_arcsec(),
            self.minor_arcsec(),
            self.pa_deg,
        )
    }
}

impl PartialEq for Beam {
    fn eq(&self, other: &Self) -> bool { self.approx_eq(other) }
}

/// MIRIAD GauDfac: deconvolve beam2 from beam1 (all params in degrees, PA returns radians).
///
/// Returns `(new_major_deg, new_minor_deg, new_pa_rad)`.
pub(crate) fn deconvolve_deg(
    maj1: f64, min1: f64, pa1_deg: f64,
    maj2: f64, min2: f64, pa2_deg: f64,
    failure_returns_zero: bool,
) -> Result<(f64, f64, f64), BeamError> {
    let pa1 = pa1_deg * DEG2RAD;
    let pa2 = pa2_deg * DEG2RAD;

    let alpha = (maj1 * pa1.cos()).powi(2) + (min1 * pa1.sin()).powi(2)
        - (maj2 * pa2.cos()).powi(2) - (min2 * pa2.sin()).powi(2);

    let beta = (maj1 * pa1.sin()).powi(2) + (min1 * pa1.cos()).powi(2)
        - (maj2 * pa2.sin()).powi(2) - (min2 * pa2.cos()).powi(2);

    let gamma = 2.0 * ((min1.powi(2) - maj1.powi(2)) * pa1.sin() * pa1.cos()
        - (min2.powi(2) - maj2.powi(2)) * pa2.sin() * pa2.cos());

    let s = alpha + beta;
    let t = ((alpha - beta).powi(2) + gamma.powi(2)).sqrt();

    let atol_t = f64::EPSILON / 3600.0_f64.powi(2);
    let alpha_fail = alpha + f64::EPSILON < 0.0;
    let beta_fail = beta + f64::EPSILON < 0.0;
    let st_fail = s < t + atol_t;

    if alpha_fail || beta_fail || st_fail {
        if failure_returns_zero {
            return Ok((0.0, 0.0, 0.0));
        }
        return Err(BeamError::DeconvolveFailed);
    }

    let new_major = (0.5 * (s + t)).sqrt() + f64::EPSILON;
    let new_minor = (0.5 * (s - t)).sqrt() + f64::EPSILON;

    let atol = 1e-7 / 3600.0;
    let new_pa = if (gamma.abs() + (alpha - beta).abs()).sqrt() < atol {
        0.0_f64
    } else {
        0.5 * (-gamma).atan2(alpha - beta)
    };

    Ok((new_major, new_minor, new_pa))
}

/// MIRIAD gaufac: scaling factor for Jy/beam images after convolution.
///
/// `conv_beam` and `orig_beam` axes in arcsec, PA in degrees.
/// `dx_arcsec`, `dy_arcsec` are pixel sizes in arcsec.
///
/// Returns `(fac, amp, result_bmaj, result_bmin, result_bpa_deg)`.
pub fn gauss_factor(
    conv_beam: &Beam,
    orig_beam: &Beam,
    dx_arcsec: f64,
    dy_arcsec: f64,
) -> (f64, f64, f64, f64, f64) {
    let bmaj2 = conv_beam.major_arcsec();
    let bmin2 = conv_beam.minor_arcsec();
    let bpa2 = conv_beam.pa_deg * DEG2RAD;

    let bmaj1 = orig_beam.major_arcsec();
    let bmin1 = orig_beam.minor_arcsec();
    let bpa1 = orig_beam.pa_deg * DEG2RAD;

    let cospa1 = bpa1.cos();
    let sinpa1 = bpa1.sin();
    let cospa2 = bpa2.cos();
    let sinpa2 = bpa2.sin();

    let alpha = (bmaj1 * cospa1).powi(2) + (bmin1 * sinpa1).powi(2)
        + (bmaj2 * cospa2).powi(2) + (bmin2 * sinpa2).powi(2);

    let beta = (bmaj1 * sinpa1).powi(2) + (bmin1 * cospa1).powi(2)
        + (bmaj2 * sinpa2).powi(2) + (bmin2 * cospa2).powi(2);

    let gamma = 2.0 * ((bmin1.powi(2) - bmaj1.powi(2)) * sinpa1 * cospa1
        + (bmin2.powi(2) - bmaj2.powi(2)) * sinpa2 * cospa2);

    let s = alpha + beta;
    let t = ((alpha - beta).powi(2) + gamma.powi(2)).sqrt();

    let bmaj_out = (0.5 * (s + t)).sqrt();
    let bmin_out = (0.5 * (s - t).max(0.0)).sqrt();

    let bpa_out_rad = if (gamma.abs() + (alpha - beta).abs()) == 0.0 {
        0.0_f64
    } else {
        0.5 * (-gamma).atan2(alpha - beta)
    };

    let denom = (alpha * beta - 0.25 * gamma.powi(2)).sqrt();
    let amp = std::f64::consts::PI / (4.0 * 2_f64.ln()) * bmaj1 * bmin1 * bmaj2 * bmin2 / denom;

    let fac = dx_arcsec.abs() * dy_arcsec.abs() / amp;

    (fac, amp, bmaj_out, bmin_out, bpa_out_rad.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_deconvolve_roundtrip() {
        let beam_a = Beam::new(10.0 / 3600.0, 8.0 / 3600.0, 30.0).unwrap();
        let beam_b = Beam::new(6.0 / 3600.0, 5.0 / 3600.0, 15.0).unwrap();
        let convolved = beam_a.convolve(&beam_b);
        let recovered = convolved.deconvolve(&beam_a).unwrap();
        assert!((recovered.major_deg - beam_b.major_deg).abs() < 1e-9,
            "major mismatch: {} vs {}", recovered.major_deg, beam_b.major_deg);
        assert!((recovered.minor_deg - beam_b.minor_deg).abs() < 1e-9,
            "minor mismatch: {} vs {}", recovered.minor_deg, beam_b.minor_deg);
    }

    #[test]
    fn test_deconvolve_fails_when_psf_larger() {
        let small = Beam::new(5.0 / 3600.0, 5.0 / 3600.0, 0.0).unwrap();
        let large = Beam::new(10.0 / 3600.0, 10.0 / 3600.0, 0.0).unwrap();
        assert!(small.deconvolve(&large).is_err());
    }

    #[test]
    fn test_gauss_factor_circular() {
        let conv = Beam::new(5.0 / 3600.0, 5.0 / 3600.0, 0.0).unwrap();
        let orig = Beam::new(10.0 / 3600.0, 10.0 / 3600.0, 0.0).unwrap();
        let dx = 2.5;
        let (fac, ..) = gauss_factor(&conv, &orig, dx, dx);
        assert!(fac.is_finite() && fac > 0.0);
    }
}
