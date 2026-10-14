//! Smooth radio astronomy FITS images and cubes to a common resolution by
//! convolution in the UV plane. A Rust port of `beamcon_2D` and `beamcon_3D`
//! from [RACS-tools](https://github.com/AlecThomson/RACS-tools), also
//! available as a [Python package](https://pypi.org/project/convolve-rs/) and
//! the `convolvers` CLI.
//!
//! - [`Beam`]: an elliptical Gaussian beam and its algebra ([`beam`]).
//! - [`common_beam`](fn@common_beam): the smallest beam a set of beams can be
//!   convolved to.
//! - [`convolve_uv`](fn@convolve_uv): UV-plane convolution between two beams.
//! - [`smooth`](fn@smooth): convolution plus Jy/beam or Kelvin flux scaling.
//! - [`fits_io`], [`cube_io`]: FITS image and cube I/O.
//!
//! ```
//! use convolve_rs::{Beam, BrightnessUnit, smooth};
//! use ndarray::Array2;
//!
//! let old_beam = Beam::from_arcsec(10.0, 10.0, 0.0)?;
//! let new_beam = Beam::from_arcsec(20.0, 20.0, 0.0)?;
//! let image = Array2::<f32>::from_elem((64, 64), 1.0);
//! let pixel_deg = 2.5 / 3600.0;
//!
//! let smoothed = smooth(
//!     &image,
//!     &old_beam,
//!     &new_beam,
//!     pixel_deg,
//!     pixel_deg,
//!     BrightnessUnit::JyPerBeam,
//!     None,
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
    ConvolutionResult, ConvolveError, FftFloat, FftPlans, convolve_uv, fftfreq, gaussft,
};
pub use fits_io::{
    FitsError, FitsImageData, FitsImageHeader, output_path, read_fits, read_fits_header,
    read_fits_plane, write_fits,
};
pub use smooth::{BrightnessUnit, smooth};

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::_convolve_rs;
