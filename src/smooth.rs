/// High-level smoothing: convolve + apply Jy/beam flux scaling.
use ndarray::Array2;
use thiserror::Error;

use crate::beam::Beam;
use crate::convolve_uv::{ConvolveError, convolve_uv};

#[derive(Debug, Error)]
pub enum SmoothError {
    #[error("convolution failed: {0}")]
    Convolve(#[from] ConvolveError),
}

/// Smooth `image` (Jy/beam) from `old_beam` to `new_beam`.
///
/// `dx_deg` / `dy_deg` are pixel sizes in degrees.  Returns an image with the
/// same dtype (f32) and the same pixel shape, ready to write back to FITS.
pub fn smooth(
    image: &Array2<f32>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
) -> Result<Array2<f32>, SmoothError> {
    let result = convolve_uv(image, old_beam, new_beam, dx_deg, dy_deg, cutoff_arcsec)?;
    let scaled = result.image.mapv(|x| (result.scaling_factor as f32) * x);
    Ok(scaled)
}
