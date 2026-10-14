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

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A 2D elliptical Gaussian beam (PSF).
///
/// Args:
///     major_deg: FWHM major axis in degrees (FITS BMAJ).
///     minor_deg: FWHM minor axis in degrees (FITS BMIN), at most ``major_deg``.
///     pa_deg: Position angle in degrees East of North (FITS BPA).
///
/// Raises:
///     ValueError: If ``minor_deg > major_deg``, an axis is negative, or a
///         value is not finite.
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
            .map_err(value_error)
    }

    /// A beam from FWHM axes in arcseconds and a PA in degrees.
    ///
    /// Examples:
    ///     >>> from convolve_rs import Beam
    ///     >>> Beam.from_arcsec(18.0, 14.4, 30.0).major_deg
    ///     0.005
    #[classmethod]
    fn from_arcsec<'py>(
        cls: &Bound<'py, PyType>,
        major_arcsec: f64,
        minor_arcsec: f64,
        pa_deg: f64,
    ) -> PyResult<Bound<'py, PyBeam>> {
        let beam = Beam::from_arcsec(major_arcsec, minor_arcsec, pa_deg).map_err(value_error)?;
        new_instance(cls, beam)
    }

    /// FWHM major axis in degrees (FITS BMAJ).
    #[getter]
    fn major_deg(&self) -> f64 {
        self.inner.major_deg()
    }

    /// FWHM minor axis in degrees (FITS BMIN).
    #[getter]
    fn minor_deg(&self) -> f64 {
        self.inner.minor_deg()
    }

    /// Position angle in degrees East of North (FITS BPA).
    #[getter]
    fn pa_deg(&self) -> f64 {
        self.inner.pa_deg()
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

    /// Solid angle in steradians, ``pi / (4 ln 2) * major_rad * minor_rad``.
    fn area_sr(&self) -> f64 {
        self.inner.area_sr()
    }

    /// The beam that, convolved with ``other``, gives this one.
    ///
    /// Raises:
    ///     ValueError: If ``other`` does not fit inside this beam.
    ///
    /// Examples:
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
    fn deconvolve<'py>(slf: &Bound<'py, Self>, other: &PyBeam) -> PyResult<Bound<'py, PyBeam>> {
        let beam = slf
            .borrow()
            .inner
            .deconvolve(&other.inner)
            .map_err(value_error)?;
        new_instance(&slf.get_type(), beam)
    }

    /// This beam convolved with ``other``.
    ///
    /// Examples:
    ///     >>> from convolve_rs import Beam
    ///     >>> a = Beam.from_arcsec(3.0, 3.0, 0.0)
    ///     >>> b = Beam.from_arcsec(4.0, 4.0, 0.0)
    ///     >>> round(a.convolve(b).major_arcsec, 6)
    ///     5.0
    fn convolve<'py>(slf: &Bound<'py, Self>, other: &PyBeam) -> PyResult<Bound<'py, PyBeam>> {
        let beam = slf.borrow().inner.convolve(&other.inner);
        new_instance(&slf.get_type(), beam)
    }

    fn __repr__(&self) -> String {
        format!(
            "Beam(major_deg={}, minor_deg={}, pa_deg={})",
            self.inner.major_deg(),
            self.inner.minor_deg(),
            self.inner.pa_deg()
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// An instance of `cls`, which may be a Python subclass of `Beam`.
fn new_instance<'py>(cls: &Bound<'py, PyType>, beam: Beam) -> PyResult<Bound<'py, PyBeam>> {
    Ok(cls
        .call1((beam.major_deg(), beam.minor_deg(), beam.pa_deg()))?
        .downcast_into::<PyBeam>()?)
}

/// The smallest beam every beam in ``beams`` can be convolved to.
///
/// Uses the CASA formula for two beams and otherwise the minimum-volume
/// ellipse fit of ``radio_beam.Beams.common_beam(method='pts')``, with the
/// same ``tolerance``, ``nsamps`` and ``epsilon``.
///
/// Returns:
///     Beam: The common beam, of the same class as ``beams[0]``.
///
/// Raises:
///     ValueError: If ``beams`` is empty or no common beam is found.
///
/// Examples:
///     >>> from convolve_rs import Beam, common_beam
///     >>> b1 = Beam.from_arcsec(10.0, 8.0, 30.0)
///     >>> b2 = Beam.from_arcsec(12.0, 6.0, 60.0)
///     >>> cb = common_beam([b1, b2])
///     >>> cb.major_arcsec >= 12.0
///     True
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (beams, tolerance=1e-4, nsamps=200, epsilon=5e-4))]
fn common_beam<'py>(
    beams: Vec<Bound<'py, PyBeam>>,
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
) -> PyResult<Bound<'py, PyBeam>> {
    let rust_beams: Vec<Beam> = beams.iter().map(|b| b.borrow().inner).collect();
    let beam = rust_common_beam(&rust_beams, tolerance, nsamps, epsilon).map_err(value_error)?;
    new_instance(&beams[0].get_type(), beam)
}

fn resolve_unit(py: Python<'_>, bunit: Option<&str>) -> PyResult<BrightnessUnit> {
    let Some(s) = bunit else {
        return Ok(BrightnessUnit::default());
    };
    if let Some(unit) = BrightnessUnit::parse(s) {
        return Ok(unit);
    }
    let msg = std::ffi::CString::new(BrightnessUnit::unrecognised_message(s))?;
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 2)?;
    Ok(BrightnessUnit::JyPerBeam)
}

/// Smooth a numpy array in its own dtype, with the GIL released so other
/// Python threads can run.
fn smooth_typed<'py, T: FftFloat + numpy::Element>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    beams: (&Beam, &Beam),
    (dx_deg, dy_deg): (f64, f64),
    unit: BrightnessUnit,
) -> PyResult<Bound<'py, PyAny>> {
    let view = arr.as_array();
    let out = py
        .detach(|| rust_smooth(view, beams.0, beams.1, dx_deg, dy_deg, unit, None))
        .map_err(value_error)?;
    Ok(out.into_pyarray(py).into_any())
}

/// `image` in native byte order. FITS data is big-endian, which numpy's typed
/// extraction rejects.
fn native_byte_order<'py>(image: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let Ok(dtype) = image.getattr("dtype") else {
        return Ok(image.clone());
    };
    if dtype.getattr("isnative")?.is_truthy()? {
        return Ok(image.clone());
    }
    let native = dtype.call_method1("newbyteorder", ("=",))?;
    image.call_method1("astype", (native,))
}

/// Smooth an image from ``old_beam`` to ``new_beam``.
///
/// Jy/beam images are scaled by the beam-area ratio so they stay in Jy/beam;
/// Kelvin images are not scaled. The GIL is released while the image is
/// convolved, so calls from several threads run in parallel.
///
/// Args:
///     image (numpy.ndarray): 2D ``float32`` or ``float64`` array, in either
///         byte order. The output has the same dtype in native byte order.
///     old_beam (Beam): Current beam.
///     new_beam (Beam): Target beam; ``old_beam`` convolved with some
///         Gaussian must give it.
///     dx_deg (float): Pixel size along x in degrees (CDELT1; sign ignored).
///     dy_deg (float): Pixel size along y in degrees (CDELT2; sign ignored).
///     cutoff_arcsec (float, optional): Raise ``ValueError`` if the BMAJ of
///         ``old_beam`` exceeds this.
///     bunit (str, optional): FITS ``BUNIT``. ``"K"`` means no flux scaling;
///         an unrecognised value warns and is treated as Jy/beam (the default).
///
/// Returns:
///     numpy.ndarray: The smoothed image. An all-NaN image is returned as is.
///
/// Raises:
///     ValueError: If ``new_beam`` cannot be reached from ``old_beam``,
///         ``old_beam`` has zero size, its BMAJ exceeds ``cutoff_arcsec``, or
///         ``image`` is not a 2D float array.
///
/// Examples:
///     >>> import numpy as np
///     >>> from convolve_rs import Beam, smooth
///     >>> image = np.ones((64, 64), dtype=np.float32)
///     >>> old = Beam.from_arcsec(10.0, 10.0, 0.0)
///     >>> new = Beam.from_arcsec(20.0, 20.0, 0.0)
///     >>> dx = 2.5 / 3600.0
///     >>> round(float(smooth(image, old, new, dx, dx)[32, 32]), 3)
///     4.0
///     >>> round(float(smooth(image, old, new, dx, dx, bunit="K")[32, 32]), 3)
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
    let bmaj = old_beam.inner.major_arcsec();
    if let Some(cutoff) = cutoff_arcsec
        && bmaj > cutoff
    {
        return Err(PyValueError::new_err(format!(
            "old_beam BMAJ ({bmaj:.2}\") exceeds the cutoff ({cutoff}\")"
        )));
    }
    let unit = resolve_unit(py, bunit)?;
    let image = &native_byte_order(image)?;
    let beams = (&old_beam.inner, &new_beam.inner);
    if let Ok(arr) = image.extract::<PyReadonlyArray2<f32>>() {
        return smooth_typed(py, arr, beams, (dx_deg, dy_deg), unit);
    }
    if let Ok(arr) = image.extract::<PyReadonlyArray2<f64>>() {
        return smooth_typed(py, arr, beams, (dx_deg, dy_deg), unit);
    }
    Err(PyValueError::new_err(
        "image must be a 2-D float32 or float64 numpy array",
    ))
}

/// The factor that keeps a Jy/beam image in Jy/beam after convolving it from
/// ``orig_beam`` with the kernel ``conv_beam``. Pixel sizes are in arcsec.
///
/// Returns:
///     tuple: ``(fac, amp, bmaj, bmin, bpa)``: the pixel factor, the kernel
///     amplitude, and the resulting beam (arcsec, arcsec, degrees).
///
/// Examples:
///     >>> from convolve_rs import Beam, gauss_factor
///     >>> conv = Beam.from_arcsec(5.0, 5.0, 0.0)
///     >>> orig = Beam.from_arcsec(10.0, 10.0, 0.0)
///     >>> fac, amp, bmaj, bmin, bpa = gauss_factor(conv, orig, 2.5, 2.5)
///     >>> round(bmaj, 5)
///     11.18034
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
fn gauss_factor(
    conv_beam: &PyBeam,
    orig_beam: &PyBeam,
    dx_arcsec: f64,
    dy_arcsec: f64,
) -> (f64, f64, f64, f64, f64) {
    let fac = rust_gauss_factor(&conv_beam.inner, &orig_beam.inner, dx_arcsec, dy_arcsec);
    let amp = dx_arcsec.abs() * dy_arcsec.abs() / fac;
    let out = orig_beam.inner.convolve(&conv_beam.inner);
    (
        fac,
        amp,
        out.major_arcsec(),
        out.minor_arcsec(),
        out.pa_deg(),
    )
}

#[cfg(feature = "stubgen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

#[cfg(feature = "stubgen")]
#[pyfunction]
fn _generate_stubs() -> PyResult<()> {
    // CARGO_MANIFEST_DIR is only set by cargo.
    if std::env::var("CARGO_MANIFEST_DIR").is_err() {
        let cwd = std::env::current_dir().map_err(value_error)?;
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
