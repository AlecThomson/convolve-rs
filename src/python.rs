#![allow(unsafe_op_in_unsafe_fn)]

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use std::path::PathBuf;

use pyo3::exceptions::PyFileNotFoundError;

use crate::beam::{Beam, gauss_factor as rust_gauss_factor};
use crate::common_beam::common_beam as rust_common_beam;
use crate::convolve_uv::{DEFAULT_NAN_FRACTION, FftFloat, PixelScale};
use crate::cube_io::{CubeMode, RefChan};
use crate::pipeline::{
    CubeOptions, SmoothOptions, SmoothedCube, SmoothedImage, cube_beams as rust_cube_beams,
    round_common_beam as rust_round_common_beam, smooth_cubes as rust_smooth_cubes,
    smooth_images as rust_smooth_images,
};
use crate::smooth::{BrightnessUnit, smooth_with as rust_smooth};

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A pipeline error, with its causes, as a ValueError.
fn pipeline_error(e: anyhow::Error) -> PyErr {
    PyValueError::new_err(format!("{e:#}"))
}

/// FileNotFoundError for the first of `paths` that does not exist.
fn check_exist(paths: &[PathBuf]) -> PyResult<()> {
    match paths.iter().find(|p| !p.exists()) {
        Some(p) => Err(PyFileNotFoundError::new_err(format!(
            "{} does not exist",
            p.display()
        ))),
        None => Ok(()),
    }
}

/// `beam` as a ``convolve_rs.Beam``.
fn py_beam(py: Python<'_>, beam: Beam) -> PyResult<Bound<'_, PyBeam>> {
    let cls = py
        .import("convolve_rs")?
        .getattr("Beam")?
        .cast_into::<PyType>()?;
    new_instance(&cls, beam)
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
        .cast_into::<PyBeam>()?)
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

/// ``nan_fraction``, checked to lie between 0 and 1.
fn check_nan_fraction(nan_fraction: f64) -> PyResult<f64> {
    if (0.0..=1.0).contains(&nan_fraction) {
        Ok(nan_fraction)
    } else {
        Err(value_error(format!(
            "nan_fraction must be between 0 and 1, not {nan_fraction}"
        )))
    }
}

/// Smooth a numpy array in its own dtype, with the GIL released so other
/// Python threads can run.
fn smooth_typed<'py, T: FftFloat + numpy::Element>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    beams: (&Beam, &Beam),
    scale: PixelScale,
    unit: BrightnessUnit,
    nan_fraction: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let view = arr.as_array();
    let out = py
        .detach(|| rust_smooth(view, beams.0, beams.1, scale, unit, nan_fraction, None))
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
///     dx_deg (float, optional): Pixel size along x in degrees (CDELT1; sign
///         ignored).
///     dy_deg (float, optional): Pixel size along y in degrees (CDELT2; sign
///         ignored).
///     cutoff_arcsec (float, optional): Raise ``ValueError`` if the BMAJ of
///         ``old_beam`` exceeds this.
///     bunit (str, optional): FITS ``BUNIT``. ``"K"`` means no flux scaling;
///         an unrecognised value warns and is treated as Jy/beam (the default).
///     pixel_scale_matrix (array-like, optional): The 2×2 pixel-to-sky matrix
///         in degrees, in place of ``dx_deg`` and ``dy_deg``, as astropy's
///         ``WCS.pixel_scale_matrix`` gives it. For rotated or skewed grids,
///         and grids with east to the right.
///     nan_fraction (float): Blank pixels where more than this fraction of
///         the kernel's weight fell on missing data: NaN pixels, or the sky
///         beyond the image. Pixels kept near missing data are divided by the
///         share that fell on data, so their flux scale is right.
///
/// Returns:
///     numpy.ndarray: The smoothed image. An all-NaN image is returned as is.
///
/// Raises:
///     ValueError: If ``new_beam`` cannot be reached from ``old_beam``,
///         ``old_beam`` has zero size, its BMAJ exceeds ``cutoff_arcsec``,
///         ``image`` is not a 2D float array, the pixel scale is not given
///         exactly one way, or the matrix is singular.
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
#[pyo3(signature = (
    image,
    old_beam,
    new_beam,
    dx_deg=None,
    dy_deg=None,
    cutoff_arcsec=None,
    bunit=None,
    pixel_scale_matrix=None,
    nan_fraction=DEFAULT_NAN_FRACTION,
))]
#[allow(clippy::too_many_arguments)]
fn smooth<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    old_beam: &PyBeam,
    new_beam: &PyBeam,
    dx_deg: Option<f64>,
    dy_deg: Option<f64>,
    cutoff_arcsec: Option<f64>,
    bunit: Option<&str>,
    pixel_scale_matrix: Option<[[f64; 2]; 2]>,
    nan_fraction: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let scale = match (dx_deg, dy_deg, pixel_scale_matrix) {
        (Some(dx), Some(dy), None) => PixelScale::cdelt(dx, dy),
        (None, None, Some(m)) => PixelScale::matrix(m)
            .ok_or_else(|| value_error("pixel_scale_matrix must be finite and invertible"))?,
        _ => {
            return Err(value_error(
                "give either dx_deg and dy_deg, or pixel_scale_matrix",
            ));
        }
    };
    let nan_fraction = check_nan_fraction(nan_fraction)?;
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
        return smooth_typed(py, arr, beams, scale, unit, nan_fraction);
    }
    if let Ok(arr) = image.extract::<PyReadonlyArray2<f64>>() {
        return smooth_typed(py, arr, beams, scale, unit, nan_fraction);
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

/// One image written by ``smooth_images``.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "SmoothedImage", module = "convolve_rs._convolve_rs", frozen)]
struct PySmoothedImage(SmoothedImage);

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PySmoothedImage {
    /// The input image.
    #[getter]
    fn input(&self) -> PathBuf {
        self.0.input.clone()
    }

    /// The smoothed image.
    #[getter]
    fn output(&self) -> PathBuf {
        self.0.output.clone()
    }

    /// The input's beam.
    #[getter]
    fn old_beam<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBeam>> {
        py_beam(py, self.0.old_beam)
    }

    /// The beam the image was convolved with, or ``None`` if it was blanked.
    #[getter]
    fn kernel<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBeam>>> {
        self.0.kernel.map(|b| py_beam(py, b)).transpose()
    }

    fn __repr__(&self) -> String {
        format!(
            "SmoothedImage(input={:?}, output={:?}, blanked={})",
            self.0.input,
            self.0.output,
            self.0.kernel.is_none()
        )
    }
}

/// The result of ``smooth_images``.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "SmoothedImages", module = "convolve_rs._convolve_rs", frozen)]
struct PySmoothedImages {
    target: Beam,
    images: Vec<SmoothedImage>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PySmoothedImages {
    /// The beam every usable image was smoothed to.
    #[getter]
    fn target<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBeam>> {
        py_beam(py, self.target)
    }

    /// One entry per input, in input order.
    #[getter]
    fn images(&self) -> Vec<PySmoothedImage> {
        self.images.iter().cloned().map(PySmoothedImage).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SmoothedImages(target={}, images=<{} images>)",
            self.target,
            self.images.len()
        )
    }
}

/// One cube written by ``smooth_cubes``.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "SmoothedCube", module = "convolve_rs._convolve_rs", frozen)]
struct PySmoothedCube(SmoothedCube);

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PySmoothedCube {
    /// The input cube.
    #[getter]
    fn input(&self) -> PathBuf {
        self.0.input.clone()
    }

    /// The smoothed cube.
    #[getter]
    fn output(&self) -> PathBuf {
        self.0.output.clone()
    }

    /// The target beam of each channel, ``None`` for a channel with no target.
    #[getter]
    fn targets<'py>(&self, py: Python<'py>) -> PyResult<Vec<Option<Bound<'py, PyBeam>>>> {
        beam_list(py, &self.0.targets)
    }

    fn __repr__(&self) -> String {
        format!(
            "SmoothedCube(input={:?}, output={:?}, channels={})",
            self.0.input,
            self.0.output,
            self.0.targets.len()
        )
    }
}

fn beam_list<'py>(
    py: Python<'py>,
    beams: &[Option<Beam>],
) -> PyResult<Vec<Option<Bound<'py, PyBeam>>>> {
    beams
        .iter()
        .map(|b| b.map(|b| py_beam(py, b)).transpose())
        .collect()
}

/// Smooth 2D FITS images to one common beam, as ``convolvers 2d`` does.
///
/// Each image is written to ``<prefix><name>.<suffix>.fits`` in ``outdir``
/// (default: beside the input). Images with no usable beam, or a BMAJ over
/// ``cutoff``, are written as NaN with a zero beam. The GIL is released.
///
/// Args:
///     paths (list[str]): The images.
///     target_beam (Beam, optional): Beam to smooth to. Default: the common
///         beam of the inputs, rounded up to 0.1 arcsec as racs_tools does.
///     cutoff (float, optional): Blank images whose BMAJ exceeds this (arcsec).
///     circularise (bool): Make the target circular.
///     suffix (str): Output name suffix.
///     prefix (str, optional): Output name prefix.
///     outdir (str, optional): Output directory.
///     tolerance, nsamps, epsilon: Settings of the common-beam fit.
///     nan_fraction (float, optional): As for ``smooth``.
///     dryrun (bool): Find the target beam but write nothing.
///     ncores (int, optional): Threads to use. Default: one per core.
///
/// Returns:
///     SmoothedImages: The target beam and, per input, its output and kernel.
///
/// Raises:
///     FileNotFoundError: If an input does not exist.
///     ValueError: If no image has a usable beam, the target is too small, or
///         an image cannot be read or written.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (
    paths,
    target_beam=None,
    cutoff=None,
    circularise=false,
    suffix="sm",
    prefix=None,
    outdir=None,
    tolerance=1e-4,
    nsamps=200,
    epsilon=5e-4,
    nan_fraction=DEFAULT_NAN_FRACTION,
    dryrun=false,
    ncores=None,
))]
#[allow(clippy::too_many_arguments)]
fn smooth_images(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    target_beam: Option<PyRef<'_, PyBeam>>,
    cutoff: Option<f64>,
    circularise: bool,
    suffix: &str,
    prefix: Option<String>,
    outdir: Option<PathBuf>,
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
    nan_fraction: f64,
    dryrun: bool,
    ncores: Option<usize>,
) -> PyResult<PySmoothedImages> {
    check_exist(&paths)?;
    let opts = SmoothOptions {
        target_beam: target_beam.map(|b| b.inner),
        circularise,
        cutoff,
        suffix: suffix.into(),
        prefix,
        outdir,
        tolerance,
        nsamps,
        epsilon,
        nan_fraction: check_nan_fraction(nan_fraction)?,
        dryrun,
        ncores,
        progress: false,
    };
    let result = py
        .detach(|| rust_smooth_images(&paths, &opts))
        .map_err(pipeline_error)?;
    Ok(PySmoothedImages {
        target: result.target,
        images: result.images,
    })
}

/// Smooth FITS cubes channel by channel, as ``convolvers 3d`` does.
///
/// Every cube must have the same number of channels and at most one Stokes
/// plane. A channel with no target, no usable beam, or a BMAJ over ``cutoff``
/// is written as NaN and masked in the BEAMS table. Each output gets a
/// beamlog of its targets. Arguments not listed here are as for
/// ``smooth_images``. The GIL is released.
///
/// Args:
///     paths (list[str]): The cubes.
///     mode (str): ``"natural"`` for a common beam per channel, ``"total"``
///         for one beam for every channel.
///     target_beam (Beam, optional): One beam for every channel.
///     target_beams (list[Beam | None], optional): A beam per channel;
///         ``None`` blanks the channel.
///     ref_chan (str, optional): Channel whose beam goes in the header's
///         BMAJ/BMIN/BPA: ``"first"``, ``"last"`` or ``"mid"``. Default: the
///         spectral axis' CRPIX.
///     uselogs (bool): Reuse the beamlogs of a previous run as the targets.
///
/// Returns:
///     list[SmoothedCube]: Per input, its output and per-channel targets.
///
/// Raises:
///     FileNotFoundError: If an input does not exist.
///     ValueError: On bad options, cubes that cannot be smoothed together, or
///         a read or write error.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (
    paths,
    mode="natural",
    target_beam=None,
    target_beams=None,
    ref_chan=None,
    uselogs=false,
    cutoff=None,
    circularise=false,
    suffix="sm",
    prefix=None,
    outdir=None,
    tolerance=1e-4,
    nsamps=200,
    epsilon=5e-4,
    nan_fraction=DEFAULT_NAN_FRACTION,
    dryrun=false,
    ncores=None,
))]
#[allow(clippy::too_many_arguments)]
fn smooth_cubes(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    mode: &str,
    target_beam: Option<PyRef<'_, PyBeam>>,
    target_beams: Option<Vec<Option<PyRef<'_, PyBeam>>>>,
    ref_chan: Option<&str>,
    uselogs: bool,
    cutoff: Option<f64>,
    circularise: bool,
    suffix: &str,
    prefix: Option<String>,
    outdir: Option<PathBuf>,
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
    nan_fraction: f64,
    dryrun: bool,
    ncores: Option<usize>,
) -> PyResult<Vec<PySmoothedCube>> {
    check_exist(&paths)?;
    let mode = match mode {
        "natural" => CubeMode::Natural,
        "total" => CubeMode::Total,
        _ => return Err(value_error(format!("unknown mode {mode:?}"))),
    };
    let ref_chan = match ref_chan {
        None => RefChan::Crpix,
        Some("first") => RefChan::First,
        Some("last") => RefChan::Last,
        Some("mid") => RefChan::Mid,
        Some(r) => return Err(value_error(format!("unknown ref_chan {r:?}"))),
    };
    let opts = CubeOptions {
        smooth: SmoothOptions {
            target_beam: target_beam.map(|b| b.inner),
            circularise,
            cutoff,
            suffix: suffix.into(),
            prefix,
            outdir,
            tolerance,
            nsamps,
            epsilon,
            nan_fraction: check_nan_fraction(nan_fraction)?,
            dryrun,
            ncores,
            progress: false,
        },
        mode,
        ref_chan,
        uselogs,
        target_beams: target_beams
            .map(|bs| bs.iter().map(|b| b.as_ref().map(|b| b.inner)).collect()),
    };
    let cubes = py
        .detach(|| rust_smooth_cubes(&paths, &opts))
        .map_err(pipeline_error)?;
    Ok(cubes.into_iter().map(PySmoothedCube).collect())
}

/// The beam of each channel of a cube, ``None`` where it is masked.
///
/// Beams come from the BEAMS table, else a beamlog beside the cube, else the
/// header's BMAJ/BMIN/BPA.
///
/// Raises:
///     FileNotFoundError: If the cube does not exist.
///     ValueError: If it cannot be read or has no beams.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
fn cube_beams(py: Python<'_>, path: PathBuf) -> PyResult<Vec<Option<Bound<'_, PyBeam>>>> {
    check_exist(std::slice::from_ref(&path))?;
    let beams = py
        .detach(|| rust_cube_beams(&path))
        .map_err(pipeline_error)?;
    beam_list(py, &beams)
}

/// ``beam`` rounded up to 0.1 arcsec axes and 0.01 degree PA, as racs_tools
/// rounds a common beam, with 0.1 arcsec more if the rounded PA would leave
/// one of ``beams`` outside it.
///
/// Examples:
///     >>> from convolve_rs import Beam, round_common_beam
///     >>> b = Beam.from_arcsec(10.03, 8.01, 30.004)
///     >>> r = round_common_beam(b, [b])
///     >>> round(r.major_arcsec, 6), round(r.minor_arcsec, 6), round(r.pa_deg, 6)
///     (10.1, 8.1, 30.01)
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
fn round_common_beam<'py>(
    beam: &Bound<'py, PyBeam>,
    beams: Vec<PyRef<'py, PyBeam>>,
) -> PyResult<Bound<'py, PyBeam>> {
    let inputs: Vec<Beam> = beams.iter().map(|b| b.inner).collect();
    let rounded = rust_round_common_beam(beam.borrow().inner, &inputs).map_err(pipeline_error)?;
    new_instance(&beam.get_type(), rounded)
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
    m.add_class::<PySmoothedImage>()?;
    m.add_class::<PySmoothedImages>()?;
    m.add_class::<PySmoothedCube>()?;
    m.add_function(wrap_pyfunction!(smooth_images, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_cubes, m)?)?;
    m.add_function(wrap_pyfunction!(cube_beams, m)?)?;
    m.add_function(wrap_pyfunction!(round_common_beam, m)?)?;
    #[cfg(feature = "stubgen")]
    m.add_function(wrap_pyfunction!(_generate_stubs, m)?)?;
    Ok(())
}
