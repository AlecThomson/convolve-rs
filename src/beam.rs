//! Radio astronomy beam (PSF) represented as a 2D elliptical Gaussian.
//!
//! All stored values use FITS conventions: major/minor FWHM in degrees, PA in
//! degrees East of North.
//!
//! The beam algebra (convolution, deconvolution, and the Jy/beam flux-scaling
//! factor) is implemented here from the standard second-moment / covariance
//! formulation of an elliptical Gaussian:
//!
//!   * An elliptical Gaussian is described by a symmetric 2×2 covariance matrix
//!     `C` in (East, North) axes. Its eigenvalues are the squared axis lengths
//!     and its eigenvectors give the orientation.
//!   * Convolving two Gaussians **adds** their covariance matrices; deconvolving
//!     **subtracts** them (valid only while the residual stays positive-definite).
//!   * The integral of a 2D Gaussian is proportional to `sqrt(det C)`, which
//!     yields the peak-amplitude / flux-rescaling factor as a ratio of
//!     determinants.
//!
//! These are textbook results for combining Gaussian point-spread functions (see
//! Wild 1970, *Aust. J. Phys.* 23, 113, for the radio-astronomy form). The same
//! standard formulae underpin `radio_beam` and RACS-tools; this module is an
//! independent implementation in terms of the covariance matrix and its
//! eigendecomposition.
use std::fmt;

use thiserror::Error;

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

/// Symmetric 2×2 covariance matrix `[[xx, xy], [xy, yy]]` of an elliptical
/// Gaussian, expressed in (East = x, North = y) axes.
///
/// The matrix entries carry units of (axis length)², so a beam given in degrees
/// produces a covariance in deg² and one given in arcsec produces arcsec². Beam
/// operations are linear in this representation: convolution is matrix addition,
/// deconvolution is subtraction, and the axis lengths / position angle are the
/// eigen-pairs.
#[derive(Debug, Clone, Copy)]
struct Cov {
    xx: f64,
    yy: f64,
    xy: f64,
}

impl Cov {
    /// Build the covariance of a Gaussian with the given FWHM axes and position
    /// angle (radians, East of North). The major axis points along
    /// `(sin θ, cos θ)` and the minor along `(cos θ, −sin θ)`.
    fn from_axes(major: f64, minor: f64, pa_rad: f64) -> Self {
        let (sin, cos) = pa_rad.sin_cos();
        let a2 = major * major;
        let b2 = minor * minor;
        Self {
            xx: a2 * sin * sin + b2 * cos * cos,
            yy: a2 * cos * cos + b2 * sin * sin,
            xy: (a2 - b2) * sin * cos,
        }
    }

    fn add(&self, other: &Cov) -> Cov {
        Cov {
            xx: self.xx + other.xx,
            yy: self.yy + other.yy,
            xy: self.xy + other.xy,
        }
    }

    fn sub(&self, other: &Cov) -> Cov {
        Cov {
            xx: self.xx - other.xx,
            yy: self.yy - other.yy,
            xy: self.xy - other.xy,
        }
    }

    fn det(&self) -> f64 {
        self.xx * self.yy - self.xy * self.xy
    }

    /// Eigenvalues `(larger, smaller)` — the squared major and minor axis lengths.
    fn eigenvalues(&self) -> (f64, f64) {
        let mean = 0.5 * (self.xx + self.yy);
        // Half the spread of the eigenvalues about their mean.
        let radius = 0.5 * (self.xx - self.yy).hypot(2.0 * self.xy);
        (mean + radius, mean - radius)
    }

    /// Position angle of the major axis in radians, East of North, folded into
    /// `(−π/2, π/2]`. A circular (degenerate) covariance has no defined
    /// orientation and returns 0.
    fn position_angle(&self) -> f64 {
        use std::f64::consts::{FRAC_PI_2, PI};
        let off_diag = self.xx - self.yy;
        let two_xy = 2.0 * self.xy;
        // Circular to machine precision → orientation undefined.
        if off_diag.hypot(two_xy) <= f64::EPSILON * (self.xx + self.yy).abs() {
            return 0.0;
        }
        // Principal-axis angle. The eigenvector of the larger eigenvalue makes
        // angle ½·atan2(2·xy, xx−yy) with the East (x) axis; converting to a
        // North-referenced PA gives π/2 minus that.
        let mut pa = FRAC_PI_2 - 0.5 * two_xy.atan2(off_diag);
        if pa > FRAC_PI_2 {
            pa -= PI;
        }
        pa
    }
}

impl Beam {
    pub fn new(major_deg: f64, minor_deg: f64, pa_deg: f64) -> Result<Self, BeamError> {
        if !major_deg.is_finite() || !minor_deg.is_finite() || !pa_deg.is_finite() {
            return Err(BeamError::NotFinite);
        }
        if minor_deg > major_deg {
            return Err(BeamError::InvalidAxes {
                major: major_deg,
                minor: minor_deg,
            });
        }
        Ok(Self {
            major_deg,
            minor_deg,
            pa_deg,
        })
    }

    pub fn from_arcsec(
        major_arcsec: f64,
        minor_arcsec: f64,
        pa_deg: f64,
    ) -> Result<Self, BeamError> {
        Self::new(major_arcsec / 3600.0, minor_arcsec / 3600.0, pa_deg)
    }

    pub fn zero() -> Self {
        Self {
            major_deg: 0.0,
            minor_deg: 0.0,
            pa_deg: 0.0,
        }
    }

    pub fn major_arcsec(&self) -> f64 {
        self.major_deg * 3600.0
    }
    pub fn minor_arcsec(&self) -> f64 {
        self.minor_deg * 3600.0
    }

    pub fn is_finite(&self) -> bool {
        self.major_deg.is_finite()
            && self.minor_deg.is_finite()
            && self.pa_deg.is_finite()
            && self.major_deg > 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.major_deg == 0.0 && self.minor_deg == 0.0
    }

    pub fn is_circular(&self, rtol: f64) -> bool {
        if self.major_deg == 0.0 {
            return true;
        }
        (self.major_deg - self.minor_deg) / self.major_deg <= rtol
    }

    pub fn area_sr(&self) -> f64 {
        let fwhm_to_area = 2.0 * std::f64::consts::PI / (8.0 * 2_f64.ln());
        self.major_deg.to_radians() * self.minor_deg.to_radians() * fwhm_to_area
    }

    /// Covariance matrix of this beam (entries in deg²).
    fn cov_deg(&self) -> Cov {
        Cov::from_axes(self.major_deg, self.minor_deg, self.pa_deg.to_radians())
    }

    /// Deconvolve `other` from `self` (i.e. `self` = result ⊛ `other`).
    ///
    /// Subtracts the covariance of `other` from that of `self` and reads off the
    /// residual ellipse. Fails if the residual is not positive-definite (the
    /// source beam is not larger than the PSF). Inputs/outputs in degrees.
    pub fn deconvolve(&self, other: &Beam) -> Result<Beam, BeamError> {
        let (new_major, new_minor, new_pa_rad) = deconvolve_deg(
            self.major_deg,
            self.minor_deg,
            self.pa_deg,
            other.major_deg,
            other.minor_deg,
            other.pa_deg,
            false,
        )?;
        Ok(Beam {
            major_deg: new_major,
            minor_deg: new_minor,
            pa_deg: new_pa_rad.to_degrees(),
        })
    }

    /// Like `deconvolve` but returns a zero beam on failure instead of an error.
    pub fn deconvolve_or_zero(&self, other: &Beam) -> Beam {
        match self.deconvolve(other) {
            Ok(b) => b,
            Err(_) => Beam::zero(),
        }
    }

    /// Convolve `self` with `other`: sum the covariance matrices and read off the
    /// resulting ellipse.
    pub fn convolve(&self, other: &Beam) -> Beam {
        let combined = self.cov_deg().add(&other.cov_deg());
        let (lam_major, lam_minor) = combined.eigenvalues();
        Beam {
            major_deg: lam_major.max(0.0).sqrt(),
            minor_deg: lam_minor.max(0.0).sqrt(),
            pa_deg: combined.position_angle().to_degrees(),
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
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(other)
    }
}

/// Deconvolve beam 2 from beam 1 by subtracting covariance matrices.
///
/// All axis inputs in degrees, PAs in degrees; the returned PA is in radians.
/// Returns `(new_major_deg, new_minor_deg, new_pa_rad)`. If the residual
/// covariance is not positive-definite the beams cannot be deconvolved: this is
/// either a `DeconvolveFailed` error or, when `failure_returns_zero` is set, a
/// zero result.
pub(crate) fn deconvolve_deg(
    maj1: f64,
    min1: f64,
    pa1_deg: f64,
    maj2: f64,
    min2: f64,
    pa2_deg: f64,
    failure_returns_zero: bool,
) -> Result<(f64, f64, f64), BeamError> {
    let residual = Cov::from_axes(maj1, min1, pa1_deg.to_radians()).sub(&Cov::from_axes(
        maj2,
        min2,
        pa2_deg.to_radians(),
    ));
    let (lam_major, lam_minor) = residual.eigenvalues();

    // The residual must be positive-(semi)definite to correspond to a real
    // ellipse: both diagonal variances and the smaller eigenvalue stay ≥ 0. The
    // floor on the smaller eigenvalue also rejects the point-source limit
    // (deconvolving a beam from itself).
    let eps = f64::EPSILON;
    let lam_floor = eps / (2.0 * 3600.0_f64.powi(2));
    if residual.xx + eps < 0.0 || residual.yy + eps < 0.0 || lam_minor < lam_floor {
        if failure_returns_zero {
            return Ok((0.0, 0.0, 0.0));
        }
        return Err(BeamError::DeconvolveFailed);
    }

    Ok((
        lam_major.sqrt(),
        lam_minor.max(0.0).sqrt(),
        residual.position_angle(),
    ))
}

/// Flux-scaling factor for Jy/beam images after convolution to a larger beam.
///
/// `conv_beam` is the convolving (difference) beam and `orig_beam` the original
/// restoring beam; both axes in arcsec, PA in degrees. `dx_arcsec`, `dy_arcsec`
/// are the pixel sizes in arcsec.
///
/// The peak amplitude of the convolving Gaussian needed to preserve Jy/beam
/// units is `π/(4 ln 2) · √(det C_orig · det C_conv / det(C_orig + C_conv))`,
/// since the integral of a 2D Gaussian scales as `√det` of its covariance. The
/// returned factor rescales pixel values by the pixel area over that amplitude.
///
/// Returns `(fac, amp, result_bmaj_arcsec, result_bmin_arcsec, result_bpa_deg)`.
pub fn gauss_factor(
    conv_beam: &Beam,
    orig_beam: &Beam,
    dx_arcsec: f64,
    dy_arcsec: f64,
) -> (f64, f64, f64, f64, f64) {
    let c_orig = Cov::from_axes(
        orig_beam.major_arcsec(),
        orig_beam.minor_arcsec(),
        orig_beam.pa_deg.to_radians(),
    );
    let c_conv = Cov::from_axes(
        conv_beam.major_arcsec(),
        conv_beam.minor_arcsec(),
        conv_beam.pa_deg.to_radians(),
    );
    let combined = c_orig.add(&c_conv);

    let (lam_major, lam_minor) = combined.eigenvalues();
    let bmaj_out = lam_major.max(0.0).sqrt();
    let bmin_out = lam_minor.max(0.0).sqrt();
    let bpa_out_rad = combined.position_angle();

    let amp = std::f64::consts::PI / (4.0 * 2_f64.ln())
        * (c_orig.det() * c_conv.det() / combined.det()).sqrt();
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
        assert!(
            (recovered.major_deg - beam_b.major_deg).abs() < 1e-9,
            "major mismatch: {} vs {}",
            recovered.major_deg,
            beam_b.major_deg
        );
        assert!(
            (recovered.minor_deg - beam_b.minor_deg).abs() < 1e-9,
            "minor mismatch: {} vs {}",
            recovered.minor_deg,
            beam_b.minor_deg
        );
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
