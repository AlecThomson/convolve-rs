//! High-level smoothing: convolve + apply Jy/beam flux scaling.
use ndarray::Array2;
use num_traits::cast;
use thiserror::Error;

use crate::beam::Beam;
use crate::convolve_uv::{ConvolveError, FftFloat, FftPlans, convolve_uv_with_plans};

#[derive(Debug, Error)]
pub enum SmoothError {
    #[error("convolution failed: {0}")]
    Convolve(#[from] ConvolveError),
}

/// Brightness unit of an image, determining whether flux scaling applies after
/// convolution to a larger beam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrightnessUnit {
    /// Jy/beam (or any per-beam flux density) — apply the Gaussian flux-scaling
    /// factor so the output stays in the same units.
    #[default]
    JyPerBeam,
    /// Kelvin (brightness temperature) — surface brightness is conserved under
    /// convolution, so the scaling factor is 1.
    Kelvin,
}

impl BrightnessUnit {
    /// Classify a FITS `BUNIT` string, returning `None` if the unit is not
    /// recognised (neither a Kelvin nor a Jy/beam form).
    ///
    /// # Examples
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

    /// Classify a FITS `BUNIT` string.  Anything recognised as a brightness
    /// temperature (Kelvin) skips flux scaling; recognised Jy/beam forms get the
    /// Gaussian factor.  Unrecognised units cannot be determined automatically:
    /// a warning is emitted and Jy/beam is assumed.
    pub fn from_bunit(bunit: &str) -> Self {
        match Self::parse(bunit) {
            Some(unit) => unit,
            None => {
                tracing::warn!(
                    "Could not determine brightness unit from BUNIT={bunit:?}; \
                     assuming Jy/beam (flux scaling applied). Pass a recognised \
                     unit (e.g. 'Jy/beam' or 'K') to silence this warning."
                );
                BrightnessUnit::JyPerBeam
            }
        }
    }
}

/// Smooth `image` from `old_beam` to `new_beam`.
///
/// `dx_deg` / `dy_deg` are pixel sizes in degrees.  `unit` selects the flux
/// scaling: [`BrightnessUnit::JyPerBeam`] applies the Gaussian factor,
/// [`BrightnessUnit::Kelvin`] leaves the data unscaled (factor 1).  The
/// convolution runs in the image's element type `T` (`f32` or `f64`); the
/// result has the same dtype and pixel shape, ready to write back to FITS.
///
/// # Examples
///
/// Smoothing a flat image from a 10″ to a 20″ circular beam:
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
/// // Jy/beam: pixel values scale by the beam-area ratio Ω_new/Ω_old = 4.
/// let jy = smooth(&image, &old, &new, dx, dx, None, BrightnessUnit::JyPerBeam)?;
/// assert!((jy[(32, 32)] - 4.0).abs() < 1e-3);
///
/// // Kelvin: surface brightness is conserved, so a flat image stays at 1.
/// let k = smooth(&image, &old, &new, dx, dx, None, BrightnessUnit::Kelvin)?;
/// assert!((k[(32, 32)] - 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn smooth<T: FftFloat>(
    image: &Array2<T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    unit: BrightnessUnit,
) -> Result<Array2<T>, SmoothError> {
    let (nrows, ncols) = image.dim();
    let plans = FftPlans::<T>::new(nrows, ncols);
    smooth_with_plans(
        image,
        old_beam,
        new_beam,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
        unit,
        &plans,
    )
}

/// Like [`smooth`], but reuses pre-built [`FftPlans`] across calls (e.g. every
/// channel of a cube), avoiding per-call FFT planning. `plans` must match the
/// image dimensions.
#[allow(clippy::too_many_arguments)]
pub fn smooth_with_plans<T: FftFloat>(
    image: &Array2<T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    unit: BrightnessUnit,
    plans: &FftPlans<T>,
) -> Result<Array2<T>, SmoothError> {
    let result = convolve_uv_with_plans(
        image,
        old_beam,
        new_beam,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
        plans,
    )?;
    // `convolve_uv` already bakes one g_ratio (= √(Ω_new/Ω_old)) into the image.
    // Jy/beam needs the full beam-area ratio Ω_new/Ω_old = g_ratio², so multiply
    // by g_ratio once more. Kelvin conserves surface brightness, so the image
    // must be flux-normalised — divide the baked-in g_ratio back out.
    let factor = match unit {
        BrightnessUnit::JyPerBeam => result.scaling_factor,
        BrightnessUnit::Kelvin => 1.0 / result.scaling_factor,
    };
    let factor_t = cast::<f64, T>(factor).expect("scaling factor out of range");
    let scaled = result.image.mapv(|x| factor_t * x);
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_parse_recognised() {
        assert_eq!(BrightnessUnit::parse("K"), Some(BrightnessUnit::Kelvin));
        assert_eq!(BrightnessUnit::parse(" k "), Some(BrightnessUnit::Kelvin));
        assert_eq!(BrightnessUnit::parse("'K'"), Some(BrightnessUnit::Kelvin));
        assert_eq!(
            BrightnessUnit::parse("Kelvin"),
            Some(BrightnessUnit::Kelvin)
        );
        assert_eq!(
            BrightnessUnit::parse("Jy/beam"),
            Some(BrightnessUnit::JyPerBeam)
        );
        assert_eq!(
            BrightnessUnit::parse("JY BEAM-1"),
            Some(BrightnessUnit::JyPerBeam)
        );
    }

    #[test]
    fn test_parse_unrecognised_is_none() {
        // Unknown / ambiguous units cannot be determined automatically.
        assert_eq!(BrightnessUnit::parse(""), None);
        assert_eq!(BrightnessUnit::parse("Jy/pixel"), None);
        assert_eq!(BrightnessUnit::parse("mJy"), None);
    }

    #[test]
    fn test_from_bunit_falls_back_to_jy_per_beam() {
        // Recognised forms classify directly; unrecognised forms warn and
        // assume Jy/beam.
        assert_eq!(BrightnessUnit::from_bunit("K"), BrightnessUnit::Kelvin);
        assert_eq!(
            BrightnessUnit::from_bunit("Jy/beam"),
            BrightnessUnit::JyPerBeam
        );
        assert_eq!(BrightnessUnit::from_bunit("wat"), BrightnessUnit::JyPerBeam);
    }

    #[test]
    fn test_kelvin_skips_flux_scaling() {
        let old = Beam::new(10.0 / 3600.0, 10.0 / 3600.0, 0.0).unwrap();
        let new = Beam::new(20.0 / 3600.0, 20.0 / 3600.0, 0.0).unwrap();
        let img = Array2::from_elem((32, 32), 1.0_f32);
        let dx = 2.5 / 3600.0;

        let jy = smooth(&img, &old, &new, dx, dx, None, BrightnessUnit::JyPerBeam).unwrap();
        let k = smooth(&img, &old, &new, dx, dx, None, BrightnessUnit::Kelvin).unwrap();

        // Jy/beam scales flux up (Ω_new/Ω_old = 4); Kelvin leaves it unscaled.
        let center = (16, 16);
        assert!(jy[center] > k[center] * 1.5, "Jy/beam should be scaled up");
        // Kelvin output is the pure convolution: a flat image stays ~1.
        assert!(
            (k[center] - 1.0).abs() < 1e-3,
            "Kelvin center = {}",
            k[center]
        );
    }
}
