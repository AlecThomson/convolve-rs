#![allow(unsafe_op_in_unsafe_fn)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::beam::{Beam, gauss_factor as rust_gauss_factor};
use crate::common_beam::common_beam as rust_common_beam;
use crate::smooth::smooth as rust_smooth;

/// A 2-D Gaussian representation of a radio telescope's PSF (beam).
///
/// All axes use FITS conventions: FWHM major and minor axes in degrees,
/// position angle in degrees East of North.
///
/// Args:
///     major_deg: FWHM major axis in degrees (FITS BMAJ).
///     minor_deg: FWHM minor axis in degrees (FITS BMIN). Must be <= major_deg.
///     pa_deg: Position angle in degrees East of North (FITS BPA).
///
/// Raises:
///     ValueError: If minor_deg > major_deg or any value is non-finite.
///
/// See Also:
///     Beam.from_arcsec: Construct from arcsecond axes.
///     Beam.from_fits_header: Construct from an astropy FITS header.
///     Beam.from_radio_beam: Construct from a ``radio_beam.Beam`` object.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Beam", subclass)]
#[derive(Clone)]
pub struct PyBeam {
    pub inner: Beam,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyBeam {
    #[new]
    fn new(major_deg: f64, minor_deg: f64, pa_deg: f64) -> PyResult<Self> {
        Beam::new(major_deg, minor_deg, pa_deg)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Construct a Beam from arcsecond axes.
    ///
    /// Args:
    ///     major_arcsec: FWHM major axis in arcseconds.
    ///     minor_arcsec: FWHM minor axis in arcseconds. Must be <= major_arcsec.
    ///     pa_deg: Position angle in degrees East of North.
    ///
    /// Returns:
    ///     Beam: The constructed beam.
    ///
    /// Raises:
    ///     ValueError: If minor_arcsec > major_arcsec or any value is non-finite.
    #[classmethod]
    fn from_arcsec(
        _cls: &Bound<'_, PyType>,
        major_arcsec: f64,
        minor_arcsec: f64,
        pa_deg: f64,
    ) -> PyResult<Self> {
        Beam::from_arcsec(major_arcsec, minor_arcsec, pa_deg)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// FWHM major axis in degrees (FITS BMAJ).
    #[getter]
    fn major_deg(&self) -> f64 {
        self.inner.major_deg
    }

    /// FWHM minor axis in degrees (FITS BMIN).
    #[getter]
    fn minor_deg(&self) -> f64 {
        self.inner.minor_deg
    }

    /// Position angle in degrees East of North (FITS BPA).
    #[getter]
    fn pa_deg(&self) -> f64 {
        self.inner.pa_deg
    }

    /// FWHM major axis in arcseconds.
    #[getter]
    fn major_arcsec(&self) -> f64 {
        self.inner.major_arcsec()
    }

    /// FWHM minor axis in arcseconds.
    #[getter]
    fn minor_arcsec(&self) -> f64 {
        self.inner.minor_arcsec()
    }

    /// Solid angle of the beam in steradians.
    ///
    /// Computed as ``(pi / (4 ln 2)) * major_rad * minor_rad``.
    ///
    /// Returns:
    ///     float: Beam solid angle in steradians.
    fn area_sr(&self) -> f64 {
        self.inner.area_sr()
    }

    /// Deconvolve ``other`` from ``self`` (i.e. ``self`` = result ⊛ ``other``).
    ///
    /// Uses the MIRIAD GauDfac algorithm (R. Sault).
    ///
    /// Args:
    ///     other (Beam): The PSF to deconvolve from this beam.
    ///
    /// Returns:
    ///     Beam: The deconvolved beam.
    ///
    /// Raises:
    ///     ValueError: If ``other`` is larger than ``self`` and deconvolution
    ///         is impossible.
    fn deconvolve(&self, other: &PyBeam) -> PyResult<PyBeam> {
        self.inner
            .deconvolve(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Convolve ``self`` with ``other``.
    ///
    /// Uses the MIRIAD GauCvl algorithm (R. Sault).
    ///
    /// Args:
    ///     other (Beam): The beam to convolve with.
    ///
    /// Returns:
    ///     Beam: The convolved beam.
    fn convolve(&self, other: &PyBeam) -> PyBeam {
        Self {
            inner: self.inner.convolve(&other.inner),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Beam(major_deg={}, minor_deg={}, pa_deg={})",
            self.inner.major_deg, self.inner.minor_deg, self.inner.pa_deg
        )
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Find the smallest beam that every beam in ``beams`` can be convolved to.
///
/// Uses the 2-beam analytic CASA algorithm when ``len(beams) == 2``, otherwise
/// the Khachiyan minimum-volume-enclosing-ellipse algorithm — the same as
/// ``radio_beam.Beams.common_beam(method='pts')``.
///
/// Args:
///     beams (list[Beam]): Input beams. Must contain at least one element.
///     tolerance (float): Convergence tolerance for the Khachiyan algorithm.
///         Default ``1e-4``.
///     nsamps (int): Number of points sampled from each beam ellipse boundary.
///         Default 200.
///     epsilon (float): Fractional padding added to each beam before the MVE
///         fit, to ensure the common beam can be marginally deconvolved from
///         all inputs. Default ``5e-4``.
///
/// Returns:
///     Beam: The smallest common beam.
///
/// Raises:
///     ValueError: If ``beams`` is empty or no valid common beam is found.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (beams, tolerance=1e-4, nsamps=200, epsilon=5e-4))]
fn common_beam(
    beams: Vec<PyBeam>,
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
) -> PyResult<PyBeam> {
    let rust_beams: Vec<Beam> = beams.iter().map(|b| b.inner).collect();
    rust_common_beam(&rust_beams, tolerance, nsamps, epsilon)
        .map(|inner| PyBeam { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Smooth a Jy/beam image from ``old_beam`` to ``new_beam``.
///
/// Convolves ``image`` in the UV plane and applies the Jy/beam flux scaling
/// factor so that the output is in the same units as the input.
///
/// Args:
///     image (numpy.ndarray): Input image in Jy/beam, shape ``(ny, nx)``,
///         dtype ``float32``.
///     old_beam (Beam): Current (input) restoring beam.
///     new_beam (Beam): Target (output) restoring beam. Must be larger than
///         ``old_beam``.
///     dx_deg (float): Pixel size along the x (RA) axis in degrees
///         (FITS CDELT1, may be negative).
///     dy_deg (float): Pixel size along the y (Dec) axis in degrees
///         (FITS CDELT2).
///     cutoff_arcsec (float, optional): If given, raise ``ValueError`` if the
///         deconvolved kernel FWHM exceeds this value in arcseconds.
///
/// Returns:
///     numpy.ndarray: Smoothed image in Jy/beam, shape ``(ny, nx)``,
///         dtype ``float32``.
///
/// Raises:
///     ValueError: If ``new_beam`` is smaller than ``old_beam``, all pixels
///         are NaN, or the kernel exceeds ``cutoff_arcsec``.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (image, old_beam, new_beam, dx_deg, dy_deg, cutoff_arcsec=None))]
fn smooth<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f32>,
    old_beam: &PyBeam,
    new_beam: &PyBeam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let owned = image.as_array().to_owned();
    rust_smooth(
        &owned,
        &old_beam.inner,
        &new_beam.inner,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
    )
    .map(|arr| arr.into_pyarray(py))
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute the MIRIAD ``gaufac`` flux-scaling factor for a Jy/beam convolution.
///
/// Returns the factor by which pixel values must be multiplied after
/// convolving a Jy/beam image from ``orig_beam`` to ``conv_beam``.
///
/// Args:
///     conv_beam (Beam): The convolving beam (the kernel applied on top of
///         ``orig_beam``).
///     orig_beam (Beam): The original restoring beam of the image.
///     dx_arcsec (float): Pixel size along the x axis in arcseconds.
///     dy_arcsec (float): Pixel size along the y axis in arcseconds.
///
/// Returns:
///     tuple: ``(fac, amp, bmaj_out, bmin_out, bpa_out_deg)`` where
///         ``fac`` is the pixel scaling factor, ``amp`` is the Gaussian kernel
///         integral, and the remaining three are the output beam parameters
///         (major/minor FWHM in arcseconds, PA in degrees).
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
fn gauss_factor(
    conv_beam: &PyBeam,
    orig_beam: &PyBeam,
    dx_arcsec: f64,
    dy_arcsec: f64,
) -> (f64, f64, f64, f64, f64) {
    rust_gauss_factor(&conv_beam.inner, &orig_beam.inner, dx_arcsec, dy_arcsec)
}

#[cfg(feature = "stubgen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

#[cfg(feature = "stubgen")]
#[pyfunction]
fn _generate_stubs() -> PyResult<()> {
    // CARGO_MANIFEST_DIR is only set by Cargo; fall back to cwd when called from Python
    if std::env::var("CARGO_MANIFEST_DIR").is_err() {
        let cwd = std::env::current_dir()
            .map_err(|e| PyValueError::new_err(format!("cannot get cwd: {e}")))?;
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("CARGO_MANIFEST_DIR", cwd);
        }
    }
    stub_info()
        .and_then(|s| s.generate())
        .map_err(|e| PyValueError::new_err(format!("stub generation failed: {e}")))
}

#[pymodule]
pub fn _convolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBeam>()?;
    m.add_function(wrap_pyfunction!(common_beam, m)?)?;
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_factor, m)?)?;
    #[cfg(feature = "stubgen")]
    m.add_function(wrap_pyfunction!(_generate_stubs, m)?)?;
    Ok(())
}
