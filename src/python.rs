#![allow(unsafe_op_in_unsafe_fn)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::beam::{gauss_factor as rust_gauss_factor, Beam};
use crate::common_beam::common_beam as rust_common_beam;
use crate::smooth::smooth as rust_smooth;

#[pyclass(name = "Beam", subclass)]
#[derive(Clone)]
pub struct PyBeam {
    pub inner: Beam,
}

#[pymethods]
impl PyBeam {
    #[new]
    fn new(major_deg: f64, minor_deg: f64, pa_deg: f64) -> PyResult<Self> {
        Beam::new(major_deg, minor_deg, pa_deg)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

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

    #[getter]
    fn major_deg(&self) -> f64 {
        self.inner.major_deg
    }

    #[getter]
    fn minor_deg(&self) -> f64 {
        self.inner.minor_deg
    }

    #[getter]
    fn pa_deg(&self) -> f64 {
        self.inner.pa_deg
    }

    #[getter]
    fn major_arcsec(&self) -> f64 {
        self.inner.major_arcsec()
    }

    #[getter]
    fn minor_arcsec(&self) -> f64 {
        self.inner.minor_arcsec()
    }

    fn area_sr(&self) -> f64 {
        self.inner.area_sr()
    }

    fn deconvolve(&self, other: &PyBeam) -> PyResult<PyBeam> {
        self.inner
            .deconvolve(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn convolve(&self, other: &PyBeam) -> PyBeam {
        Self { inner: self.inner.convolve(&other.inner) }
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
    rust_smooth(&owned, &old_beam.inner, &new_beam.inner, dx_deg, dy_deg, cutoff_arcsec)
        .map(|arr| arr.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn gauss_factor(
    conv_beam: &PyBeam,
    orig_beam: &PyBeam,
    dx_arcsec: f64,
    dy_arcsec: f64,
) -> (f64, f64, f64, f64, f64) {
    rust_gauss_factor(&conv_beam.inner, &orig_beam.inner, dx_arcsec, dy_arcsec)
}

#[pymodule]
pub fn _convolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBeam>()?;
    m.add_function(wrap_pyfunction!(common_beam, m)?)?;
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_factor, m)?)?;
    Ok(())
}
