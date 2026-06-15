#![allow(unsafe_op_in_unsafe_fn)]

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::beam::{Beam, gauss_factor as rust_gauss_factor};
use crate::common_beam::common_beam as rust_common_beam;
use crate::convolve_uv::FftFloat;
use crate::smooth::{BrightnessUnit, smooth as rust_smooth};

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
///
/// Examples:
///     >>> from convolve_rs import Beam
///     >>> beam = Beam(0.005, 0.004, 30.0)
///     >>> beam.major_deg
///     0.005
///     >>> round(beam.major_arcsec, 6)
///     18.0
///     >>> Beam(0.004, 0.005, 0.0)
///     Traceback (most recent call last):
///     ...
///     ValueError: invalid beam: minor axis (0.005) > major axis (0.004)
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
    ///
    /// Examples:
    ///     >>> from convolve_rs import Beam
    ///     >>> beam = Beam.from_arcsec(18.0, 14.4, 30.0)
    ///     >>> beam.major_deg
    ///     0.005
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
    /// Subtracts the Gaussian covariance matrices and reads off the residual
    /// ellipse (Wild 1970).
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
    ///
    /// Examples:
    ///     Deconvolution inverts convolution:
    ///
    ///     >>> from convolve_rs import Beam
    ///     >>> a = Beam.from_arcsec(3.0, 3.0, 0.0)
    ///     >>> b = Beam.from_arcsec(4.0, 4.0, 0.0)
    ///     >>> c = a.convolve(b)
    ///     >>> round(c.deconvolve(a).major_arcsec, 6)
    ///     4.0
    ///     >>> b.deconvolve(c)
    ///     Traceback (most recent call last):
    ///     ...
    ///     ValueError: beam could not be deconvolved: source beam is smaller than the PSF
    fn deconvolve(&self, other: &PyBeam) -> PyResult<PyBeam> {
        self.inner
            .deconvolve(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Convolve ``self`` with ``other``.
    ///
    /// Adds the Gaussian covariance matrices and reads off the resulting
    /// ellipse (Wild 1970).
    ///
    /// Args:
    ///     other (Beam): The beam to convolve with.
    ///
    /// Returns:
    ///     Beam: The convolved beam.
    ///
    /// Examples:
    ///     Convolving two circular beams adds their axes in quadrature:
    ///
    ///     >>> from convolve_rs import Beam
    ///     >>> a = Beam.from_arcsec(3.0, 3.0, 0.0)
    ///     >>> b = Beam.from_arcsec(4.0, 4.0, 0.0)
    ///     >>> round(a.convolve(b).major_arcsec, 6)
    ///     5.0
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
///
/// Examples:
///     >>> from convolve_rs import Beam, common_beam
///     >>> b1 = Beam.from_arcsec(10.0, 8.0, 30.0)
///     >>> b2 = Beam.from_arcsec(12.0, 6.0, 60.0)
///     >>> cb = common_beam([b1, b2])
///     >>> cb.major_arcsec >= 12.0
///     True
///     >>> cb.area_sr() >= max(b1.area_sr(), b2.area_sr())
///     True
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

/// Resolve a FITS `BUNIT` string to a [`BrightnessUnit`], warning (and assuming
/// Jy/beam) when the string is given but not recognised.
fn resolve_unit(py: Python<'_>, bunit: Option<&str>) -> PyResult<BrightnessUnit> {
    match bunit {
        Some(s) => match BrightnessUnit::parse(s) {
            Some(unit) => Ok(unit),
            None => {
                let msg = std::ffi::CString::new(format!(
                    "Could not determine brightness unit from bunit={s:?}; \
                     assuming Jy/beam (flux scaling applied). Pass a recognised \
                     unit (e.g. 'Jy/beam' or 'K') to silence this warning."
                ))?;
                PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 2)?;
                Ok(BrightnessUnit::JyPerBeam)
            }
        },
        None => Ok(BrightnessUnit::default()),
    }
}

/// Convolve one already-extracted `T`-typed array and box the result as a numpy
/// array of the same dtype. Shared by the f32 and f64 arms of [`smooth`] so the
/// two precisions cannot drift apart.
#[allow(clippy::too_many_arguments)]
fn smooth_typed<'py, T>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    old_beam: &Beam,
    new_beam: &Beam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    unit: BrightnessUnit,
) -> PyResult<Bound<'py, PyAny>>
where
    T: FftFloat + numpy::Element,
{
    let owned = arr.as_array().to_owned();
    let out = rust_smooth(
        &owned,
        old_beam,
        new_beam,
        dx_deg,
        dy_deg,
        cutoff_arcsec,
        unit,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py).into_any())
}

/// Smooth an image from ``old_beam`` to ``new_beam``.
///
/// Convolves ``image`` in the UV plane and applies the flux scaling
/// appropriate for ``bunit`` so that the output is in the same units as the
/// input: Jy/beam images get the Gaussian beam-area factor, Kelvin
/// (brightness temperature) images conserve surface brightness and are left
/// unscaled.
///
/// Args:
///     image (numpy.ndarray): Input image, shape ``(ny, nx)``, dtype
///         ``float32`` or ``float64``. The convolution runs in the input's
///         precision and the output keeps the same dtype.
///     old_beam (Beam): Current (input) restoring beam.
///     new_beam (Beam): Target (output) restoring beam. Must be larger than
///         ``old_beam``.
///     dx_deg (float): Pixel size along the x (RA) axis in degrees
///         (FITS CDELT1, may be negative).
///     dy_deg (float): Pixel size along the y (Dec) axis in degrees
///         (FITS CDELT2).
///     cutoff_arcsec (float, optional): If given, raise ``ValueError`` if the
///         deconvolved kernel FWHM exceeds this value in arcseconds.
///     bunit (str, optional): FITS ``BUNIT`` brightness unit. If it denotes
///         Kelvin (e.g. ``"K"``), surface brightness is conserved and no flux
///         scaling is applied; if it denotes Jy/beam, the Gaussian
///         flux-scaling factor is applied. An unrecognised string emits a
///         ``UserWarning`` and is treated as Jy/beam. Defaults to Jy/beam.
///
/// Returns:
///     numpy.ndarray: Smoothed image, shape ``(ny, nx)``, same dtype as the
///         input (``float32`` or ``float64``).
///
/// Raises:
///     ValueError: If ``new_beam`` is smaller than ``old_beam``, all pixels
///         are NaN, or the kernel exceeds ``cutoff_arcsec``.
///
/// Warns:
///     UserWarning: If ``bunit`` is given but not recognised as either a
///         Kelvin or Jy/beam unit (Jy/beam is then assumed).
///
/// Examples:
///     Smoothing a flat Jy/beam image from a 10″ to a 20″ circular beam
///     scales pixel values by the beam-area ratio (4); in Kelvin, surface
///     brightness is conserved:
///
///     >>> import numpy as np
///     >>> from convolve_rs import Beam, smooth
///     >>> image = np.ones((64, 64), dtype=np.float32)
///     >>> old = Beam.from_arcsec(10.0, 10.0, 0.0)
///     >>> new = Beam.from_arcsec(20.0, 20.0, 0.0)
///     >>> dx = 2.5 / 3600.0
///     >>> jy = smooth(image, old, new, dx, dx)
///     >>> round(float(jy[32, 32]), 3)
///     4.0
///     >>> k = smooth(image, old, new, dx, dx, bunit="K")
///     >>> round(float(k[32, 32]), 3)
///     1.0
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (image, old_beam, new_beam, dx_deg, dy_deg, cutoff_arcsec=None, bunit=None))]
#[allow(clippy::too_many_arguments)]
fn smooth<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    old_beam: &PyBeam,
    new_beam: &PyBeam,
    dx_deg: f64,
    dy_deg: f64,
    cutoff_arcsec: Option<f64>,
    bunit: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let unit = resolve_unit(py, bunit)?;

    // Dispatch on the input dtype so the convolution runs at the array's native
    // precision and the output keeps that dtype. f32 (the common case) is tried
    // first; f64 arrays take the f64 path.
    if let Ok(arr) = image.extract::<PyReadonlyArray2<f32>>() {
        return smooth_typed(
            py,
            arr,
            &old_beam.inner,
            &new_beam.inner,
            dx_deg,
            dy_deg,
            cutoff_arcsec,
            unit,
        );
    }
    if let Ok(arr) = image.extract::<PyReadonlyArray2<f64>>() {
        return smooth_typed(
            py,
            arr,
            &old_beam.inner,
            &new_beam.inner,
            dx_deg,
            dy_deg,
            cutoff_arcsec,
            unit,
        );
    }

    Err(PyValueError::new_err(
        "image must be a 2-D float32 or float64 numpy array",
    ))
}

/// Compute the flux-scaling factor for a Jy/beam convolution.
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
///
/// Examples:
///     >>> from convolve_rs import Beam, gauss_factor
///     >>> conv = Beam.from_arcsec(5.0, 5.0, 0.0)
///     >>> orig = Beam.from_arcsec(10.0, 10.0, 0.0)
///     >>> fac, amp, bmaj, bmin, bpa = gauss_factor(conv, orig, 2.5, 2.5)
///     >>> fac > 0.0
///     True
///     >>> round(bmaj, 5)  # √(10² + 5²)
///     11.18034
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
