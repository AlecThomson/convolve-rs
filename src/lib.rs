//! Rust port of `beamcon_2D`/`beamcon_3D` from
//! [RACS-tools](https://github.com/AlecThomson/RACS-tools): smooth radio
//! astronomy FITS images and cubes to a common resolution via UV-plane (FFT)
//! convolution.
//!
//! Also available as a [Python package](https://pypi.org/project/convolve-rs/)
//! and a CLI tool (`convolvers`).
//!
//! # Overview
//!
//! - [`Beam`]: a 2D elliptical Gaussian beam (PSF) with convolution /
//!   deconvolution algebra ([`beam`]).
//! - [`common_beam`](fn@common_beam): smallest beam that a set of beams can be
//!   convolved to ([`common_beam`](mod@common_beam) module).
//! - [`convolve_uv`](fn@convolve_uv): FFT-based UV-plane convolution of an
//!   image between two beams ([`convolve_uv`](mod@convolve_uv) module).
//! - [`smooth`](fn@smooth): high-level convolve-plus-flux-scaling for Jy/beam
//!   or Kelvin images ([`smooth`](mod@smooth) module).
//! - [`fits_io`] / [`cube_io`]: FITS image and cube I/O.
//!
//! # Example
//!
//! Smooth an image from a 10″ to a 20″ circular beam:
//!
//! ```
//! use convolve_rs::{Beam, BrightnessUnit, smooth};
//! use ndarray::Array2;
//!
//! let old_beam = Beam::from_arcsec(10.0, 10.0, 0.0)?;
//! let new_beam = Beam::from_arcsec(20.0, 20.0, 0.0)?;
//! let image = Array2::<f32>::from_elem((64, 64), 1.0);
//! let pixel_size_deg = 2.5 / 3600.0;
//!
//! let smoothed = smooth(
//!     &image,
//!     &old_beam,
//!     &new_beam,
//!     pixel_size_deg,
//!     pixel_size_deg,
//!     None,
//!     BrightnessUnit::JyPerBeam,
//! )?;
//! assert_eq!(smoothed.dim(), image.dim());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
pub mod beam;
pub mod common_beam;
pub mod convolve_uv;
pub mod cube_io;
pub mod fits_io;
pub mod smooth;

pub use beam::{Beam, BeamError, gauss_factor};
pub use common_beam::{CommonBeamError, common_beam, find_commonbeam_between};
pub use convolve_uv::{
    ConvolutionResult, ConvolveError, FftFloat, FftPlans, convolve_uv, convolve_uv_with_plans,
    fftfreq, gaussft,
};
pub use fits_io::{FitsError, FitsImageData, output_path, read_fits, write_fits};
pub use smooth::{BrightnessUnit, SmoothError, smooth, smooth_with_plans};

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::_convolve_rs;
