//! Elliptical Gaussian beams and their algebra.
//!
//! A beam is handled through its 2×2 covariance matrix: convolution adds
//! matrices, deconvolution subtracts them, and the axes and position angle are
//! the eigen-pairs (Wild 1970, *Aust. J. Phys.* 23, 113).
use std::fmt;

use thiserror::Error;

/// A 2D elliptical Gaussian beam in FITS conventions: FWHM axes in degrees
/// (BMAJ, BMIN) and position angle in degrees East of North (BPA).
///
/// ```
/// use convolve_rs::Beam;
///
/// let beam = Beam::from_arcsec(20.0, 10.0, 45.0)?;
/// assert_eq!(beam.major_arcsec(), 20.0);
/// assert_eq!(beam.major_deg(), 20.0 / 3600.0);
/// # Ok::<(), convolve_rs::BeamError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Beam {
    major_deg: f64,
    minor_deg: f64,
    pa_deg: f64,
}

#[derive(Debug, Error)]
pub enum BeamError {
    #[error("beam could not be deconvolved: source beam is smaller than the PSF")]
    DeconvolveFailed,
    #[error("invalid beam: minor axis ({minor}) > major axis ({major})")]
    InvalidAxes { major: f64, minor: f64 },
    #[error("invalid beam: negative axis ({minor})")]
    NegativeAxis { minor: f64 },
    #[error("beam is not finite (NaN or infinite values)")]
    NotFinite,
}

/// Covariance `[[xx, xy], [xy, yy]]` of a Gaussian in (East, North) axes, in
/// the square of the axis unit.
#[derive(Debug, Clone, Copy)]
struct Cov {
    xx: f64,
    yy: f64,
    xy: f64,
}

impl Cov {
    /// The major axis points along `(sin θ, cos θ)`.
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

    /// `(larger, smaller)`: the squared major and minor axes.
    fn eigenvalues(&self) -> (f64, f64) {
        let mean = 0.5 * (self.xx + self.yy);
        let radius = 0.5 * (self.xx - self.yy).hypot(2.0 * self.xy);
        (mean + radius, mean - radius)
    }

    /// PA of the major axis in radians, in `(−π/2, π/2]`; 0 when circular.
    fn position_angle(&self) -> f64 {
        use std::f64::consts::{FRAC_PI_2, PI};
        let diff = self.xx - self.yy;
        let two_xy = 2.0 * self.xy;
        if diff.hypot(two_xy) <= f64::EPSILON * (self.xx + self.yy).abs() {
            return 0.0;
        }
        // atan2 gives the angle from East; PA is measured from North.
        let mut pa = FRAC_PI_2 - 0.5 * two_xy.atan2(diff);
        if pa > FRAC_PI_2 {
            pa -= PI;
        }
        pa
    }

    fn to_beam(self) -> Beam {
        let (lam_major, lam_minor) = self.eigenvalues();
        Beam {
            major_deg: lam_major.max(0.0).sqrt(),
            minor_deg: lam_minor.max(0.0).sqrt(),
            pa_deg: self.position_angle().to_degrees(),
        }
    }
}

impl Beam {
    /// A beam from FWHM axes in degrees and a PA in degrees East of North.
    ///
    /// # Errors
    ///
    /// If a value is not finite, an axis is negative, or `minor_deg > major_deg`.
    ///
    /// ```
    /// use convolve_rs::Beam;
    ///
    /// let beam = Beam::new(20.0 / 3600.0, 10.0 / 3600.0, 45.0)?;
    /// assert_eq!(beam.pa_deg(), 45.0);
    /// assert!(Beam::new(10.0 / 3600.0, 20.0 / 3600.0, 0.0).is_err());
    /// # Ok::<(), convolve_rs::BeamError>(())
    /// ```
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
        if minor_deg < 0.0 {
            return Err(BeamError::NegativeAxis { minor: minor_deg });
        }
        Ok(Self {
            major_deg,
            minor_deg,
            pa_deg,
        })
    }

    /// A beam from FWHM axes in arcseconds and a PA in degrees East of North.
    ///
    /// ```
    /// use convolve_rs::Beam;
    ///
    /// let beam = Beam::from_arcsec(20.0, 10.0, 45.0)?;
    /// assert_eq!(beam.minor_deg(), 10.0 / 3600.0);
    /// # Ok::<(), convolve_rs::BeamError>(())
    /// ```
    pub fn from_arcsec(
        major_arcsec: f64,
        minor_arcsec: f64,
        pa_deg: f64,
    ) -> Result<Self, BeamError> {
        Self::new(major_arcsec / 3600.0, minor_arcsec / 3600.0, pa_deg)
    }

    /// The zero-size beam, used for masked channels.
    pub fn zero() -> Self {
        Self {
            major_deg: 0.0,
            minor_deg: 0.0,
            pa_deg: 0.0,
        }
    }

    pub fn major_deg(&self) -> f64 {
        self.major_deg
    }

    pub fn minor_deg(&self) -> f64 {
        self.minor_deg
    }

    pub fn pa_deg(&self) -> f64 {
        self.pa_deg
    }

    pub fn major_arcsec(&self) -> f64 {
        self.major_deg * 3600.0
    }

    pub fn minor_arcsec(&self) -> f64 {
        self.minor_deg * 3600.0
    }

    /// True if both axes are positive, so the beam can be convolved.
    pub fn has_area(&self) -> bool {
        self.minor_deg > 0.0
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

    /// Solid angle in steradians.
    pub fn area_sr(&self) -> f64 {
        let fwhm_to_area = 2.0 * std::f64::consts::PI / (8.0 * 2_f64.ln());
        self.major_deg.to_radians() * self.minor_deg.to_radians() * fwhm_to_area
    }

    fn cov_deg(&self) -> Cov {
        Cov::from_axes(self.major_deg, self.minor_deg, self.pa_deg.to_radians())
    }

    /// The beam that, convolved with `other`, gives `self`.
    ///
    /// # Errors
    ///
    /// [`BeamError::DeconvolveFailed`] if the result would not be a beam with
    /// positive area, as in radio_beam.
    ///
    /// ```
    /// use convolve_rs::Beam;
    ///
    /// let a = Beam::from_arcsec(10.0, 8.0, 30.0)?;
    /// let b = Beam::from_arcsec(6.0, 5.0, 15.0)?;
    /// let recovered = a.convolve(&b).deconvolve(&a)?;
    /// assert!((recovered.major_arcsec() - b.major_arcsec()).abs() < 1e-6);
    /// assert!((recovered.minor_arcsec() - b.minor_arcsec()).abs() < 1e-6);
    /// assert!(b.deconvolve(&a).is_err());
    /// # Ok::<(), convolve_rs::BeamError>(())
    /// ```
    pub fn deconvolve(&self, other: &Beam) -> Result<Beam, BeamError> {
        let residual = self.cov_deg().sub(&other.cov_deg());
        let (_, lam_minor) = residual.eigenvalues();
        // radio_beam's thresholds: an absolute epsilon on the diagonal, and a
        // floor on the minor axis that rejects a point-like result.
        let eps = f64::EPSILON;
        let lam_floor = eps / (2.0 * 3600.0_f64.powi(2));
        if residual.xx + eps < 0.0 || residual.yy + eps < 0.0 || lam_minor < lam_floor {
            return Err(BeamError::DeconvolveFailed);
        }
        Ok(residual.to_beam())
    }

    /// Like [`Beam::deconvolve`], but a zero beam on failure.
    pub fn deconvolve_or_zero(&self, other: &Beam) -> Beam {
        self.deconvolve(other).unwrap_or(Beam::zero())
    }

    /// True if `other` can be convolved up to `self`.
    ///
    /// Unlike [`Beam::deconvolve`], this accepts a kernel that is a line or a
    /// point (e.g. 20″×10″ inside 20″×20″), which the UV-plane filter handles.
    ///
    /// ```
    /// use convolve_rs::Beam;
    ///
    /// let target = Beam::from_arcsec(20.0, 20.0, 0.0)?;
    /// assert!(target.contains(&Beam::from_arcsec(20.0, 10.0, 0.0)?));
    /// assert!(!target.contains(&Beam::from_arcsec(21.0, 10.0, 0.0)?));
    /// # Ok::<(), convolve_rs::BeamError>(())
    /// ```
    pub fn contains(&self, other: &Beam) -> bool {
        let own = self.cov_deg();
        let residual = own.sub(&other.cov_deg());
        let tol = 1e-12 * (own.xx + own.yy).abs();
        let (_, lam_minor) = residual.eigenvalues();
        residual.xx >= -tol && residual.yy >= -tol && lam_minor >= -tol
    }

    /// The convolution of two beams.
    ///
    /// ```
    /// use convolve_rs::Beam;
    ///
    /// let a = Beam::from_arcsec(3.0, 3.0, 0.0)?;
    /// let b = Beam::from_arcsec(4.0, 4.0, 0.0)?;
    /// assert!((a.convolve(&b).major_arcsec() - 5.0).abs() < 1e-9);
    /// # Ok::<(), convolve_rs::BeamError>(())
    /// ```
    pub fn convolve(&self, other: &Beam) -> Beam {
        self.cov_deg().add(&other.cov_deg()).to_beam()
    }

    /// Equal axes to within 1e-10 degrees, and equal PA (mod 180°) unless the
    /// beam is circular.
    pub fn approx_eq(&self, other: &Beam) -> bool {
        const ATOL: f64 = 1e-10;
        let pa_eq = self.is_circular(1e-6)
            || (self.pa_deg.rem_euclid(180.0) - other.pa_deg.rem_euclid(180.0)).abs() < ATOL;
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

/// Factor that keeps a Jy/beam image in Jy/beam after convolving a map with
/// beam `orig_beam` by the kernel `conv_beam`. Pixel sizes are in arcsec.
///
/// The kernel amplitude that preserves Jy/beam is
/// `π/(4 ln 2) · √(det C_orig · det C_conv / det(C_orig + C_conv))`, and the
/// factor is the pixel area over that amplitude.
///
/// ```
/// use convolve_rs::{Beam, gauss_factor};
///
/// let orig = Beam::from_arcsec(10.0, 10.0, 0.0)?;
/// let conv = Beam::from_arcsec(5.0, 5.0, 0.0)?;
/// assert!(gauss_factor(&conv, &orig, 2.5, 2.5) > 0.0);
/// # Ok::<(), convolve_rs::BeamError>(())
/// ```
pub fn gauss_factor(conv_beam: &Beam, orig_beam: &Beam, dx_arcsec: f64, dy_arcsec: f64) -> f64 {
    let arcsec_cov =
        |b: &Beam| Cov::from_axes(b.major_arcsec(), b.minor_arcsec(), b.pa_deg.to_radians());
    let (c_orig, c_conv) = (arcsec_cov(orig_beam), arcsec_cov(conv_beam));
    let amp = std::f64::consts::PI / (4.0 * 2_f64.ln())
        * (c_orig.det() * c_conv.det() / c_orig.add(&c_conv).det()).sqrt();
    dx_arcsec.abs() * dy_arcsec.abs() / amp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_deconvolve_roundtrip() {
        let a = Beam::from_arcsec(10.0, 8.0, 30.0).unwrap();
        let b = Beam::from_arcsec(6.0, 5.0, 15.0).unwrap();
        let recovered = a.convolve(&b).deconvolve(&a).unwrap();
        assert!((recovered.major_deg - b.major_deg).abs() < 1e-9);
        assert!((recovered.minor_deg - b.minor_deg).abs() < 1e-9);
    }

    #[test]
    fn test_deconvolve_fails_when_psf_larger() {
        let small = Beam::from_arcsec(5.0, 5.0, 0.0).unwrap();
        let large = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
        assert!(small.deconvolve(&large).is_err());
    }

    #[test]
    fn test_gauss_factor_circular() {
        let conv = Beam::from_arcsec(5.0, 5.0, 0.0).unwrap();
        let orig = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
        let fac = gauss_factor(&conv, &orig, 2.5, 2.5);
        assert!(fac.is_finite() && fac > 0.0);
    }

    #[test]
    fn test_negative_axis_rejected() {
        assert!(Beam::new(1e-3, -1e-3, 0.0).is_err());
        assert!(Beam::new(-1e-3, -2e-3, 0.0).is_err());
    }
}
