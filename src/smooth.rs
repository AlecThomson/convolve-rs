//! Convolution plus the flux scaling a brightness unit needs.
use ndarray::{Array2, CowArray, Ix2};

use crate::beam::Beam;
use crate::convolve_uv::{ConvolveError, FftFloat, FftPlans, convolve_with_gain};

/// Brightness unit of an image, which decides the flux scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrightnessUnit {
    /// Jy/beam: scaled by `Ω_new/Ω_old` so values stay in Jy/beam.
    #[default]
    JyPerBeam,
    /// Kelvin: surface brightness is conserved, so no scaling.
    Kelvin,
}

impl BrightnessUnit {
    /// Classify a FITS `BUNIT` string, or `None` if it is not recognised.
    ///
    /// ```
    /// use convolve_rs::BrightnessUnit;
    ///
    /// assert_eq!(BrightnessUnit::parse("Jy/beam"), Some(BrightnessUnit::JyPerBeam));
    /// assert_eq!(BrightnessUnit::parse(" K "), Some(BrightnessUnit::Kelvin));
    /// assert_eq!(BrightnessUnit::parse("Jy/pixel"), None);
    /// ```
    pub fn parse(bunit: &str) -> Option<Self> {
        let u = bunit.trim().trim_matches('\'').trim().to_ascii_uppercase();
        match u.as_str() {
            "K" | "KELVIN" => Some(BrightnessUnit::Kelvin),
            "JY/BEAM" | "JY BEAM-1" | "JY/BM" | "JYBEAM" => Some(BrightnessUnit::JyPerBeam),
            _ => None,
        }
    }

    /// Classify a FITS `BUNIT` string, warning and assuming Jy/beam if it is
    /// not recognised.
    pub fn from_bunit(bunit: &str) -> Self {
        Self::parse(bunit).unwrap_or_else(|| {
            tracing::warn!("{}", Self::unrecognised_message(bunit));
            BrightnessUnit::JyPerBeam
        })
    }

    /// The warning for a `BUNIT` that is not recognised.
    pub(crate) fn unrecognised_message(bunit: &str) -> String {
        format!(
            "Could not determine brightness unit from BUNIT={bunit:?}; assuming \
             Jy/beam (flux scaling applied). Use a recognised unit (e.g. 'Jy/beam' \
             or 'K') to silence this warning."
        )
    }
}

/// Smooth `image` from `old_beam` to `new_beam`, scaled for `unit`.
///
/// Arguments are as for [`convolve_uv`](crate::convolve_uv()). The result has
/// the input's precision and shape.
///
/// # Errors
///
/// As for [`convolve_uv`](crate::convolve_uv()).
///
/// ```
/// use convolve_rs::{Beam, BrightnessUnit, smooth};
/// use ndarray::Array2;
///
/// let old = Beam::from_arcsec(10.0, 10.0, 0.0)?;
/// let new = Beam::from_arcsec(20.0, 20.0, 0.0)?;
/// let image = Array2::<f32>::from_elem((64, 64), 1.0);
/// let dx = 2.5 / 3600.0;
///
/// // Jy/beam values scale by the beam-area ratio, 4.
/// let jy = smooth(&image, &old, &new, dx, dx, BrightnessUnit::JyPerBeam, None)?;
/// assert!((jy[(32, 32)] - 4.0).abs() < 1e-3);
///
/// // Kelvin: a flat image stays flat.
/// let k = smooth(&image, &old, &new, dx, dx, BrightnessUnit::Kelvin, None)?;
/// assert!((k[(32, 32)] - 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn smooth<'a, T: FftFloat>(
    image: impl Into<CowArray<'a, T, Ix2>>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    unit: BrightnessUnit,
    plans: Option<&FftPlans<T>>,
) -> Result<Array2<T>, ConvolveError> {
    // The filter's DC gain is g_ratio = √(Ω_new/Ω_old); Jy/beam needs its
    // square and Kelvin needs 1.
    let result = convolve_with_gain(
        image.into(),
        old_beam,
        new_beam,
        (dx_deg, dy_deg),
        plans,
        |ratio| match unit {
            BrightnessUnit::JyPerBeam => ratio * ratio,
            BrightnessUnit::Kelvin => 1.0,
        },
    )?;
    Ok(result.image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_parse_recognised() {
        for (s, want) in [
            ("K", BrightnessUnit::Kelvin),
            (" k ", BrightnessUnit::Kelvin),
            ("'K'", BrightnessUnit::Kelvin),
            ("Kelvin", BrightnessUnit::Kelvin),
            ("Jy/beam", BrightnessUnit::JyPerBeam),
            ("JY BEAM-1", BrightnessUnit::JyPerBeam),
        ] {
            assert_eq!(BrightnessUnit::parse(s), Some(want), "{s}");
        }
    }

    #[test]
    fn test_parse_unrecognised_is_none() {
        for s in ["", "Jy/pixel", "mJy"] {
            assert_eq!(BrightnessUnit::parse(s), None);
        }
    }

    #[test]
    fn test_from_bunit_falls_back_to_jy_per_beam() {
        assert_eq!(BrightnessUnit::from_bunit("K"), BrightnessUnit::Kelvin);
        assert_eq!(BrightnessUnit::from_bunit("wat"), BrightnessUnit::JyPerBeam);
    }

    #[test]
    fn test_kelvin_skips_flux_scaling() {
        let old = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
        let new = Beam::from_arcsec(20.0, 20.0, 0.0).unwrap();
        let img = Array2::from_elem((32, 32), 1.0_f32);
        let dx = 2.5 / 3600.0;
        let jy = smooth(&img, &old, &new, dx, dx, BrightnessUnit::JyPerBeam, None).unwrap();
        let k = smooth(&img, &old, &new, dx, dx, BrightnessUnit::Kelvin, None).unwrap();
        assert!(
            (jy[(16, 16)] - 4.0).abs() < 1e-3,
            "Jy/beam {}",
            jy[(16, 16)]
        );
        assert!((k[(16, 16)] - 1.0).abs() < 1e-3, "Kelvin {}", k[(16, 16)]);
    }
}
