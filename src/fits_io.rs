//! FITS image reading and writing.
//!
//! Handles 2D images (NAXIS = 2) and 4D images (NAXIS = 4). Only the first
//! plane of a 4D image is used, as in racs_tools.
use std::path::{Path, PathBuf};

use atfits_rs::{HeaderGeom, PixelType, create_cube_open, pixel_scale_deg, update_key_f64};
use fitsio::FitsFile;
use fitsio::hdu::FitsHdu;
use ndarray::Array2;
use thiserror::Error;

use crate::beam::{Beam, BeamError};
use crate::cube_io::CubeElem;
use crate::smooth::BrightnessUnit;

pub use atfits_rs::output_path;

#[derive(Debug, Error)]
pub enum FitsError {
    #[error("FITS I/O error: {0}")]
    Fitsio(#[from] fitsio::errors::Error),
    #[error("missing keyword: {0}")]
    MissingKeyword(String),
    #[error("unsupported NAXIS={0} (expected 2 or 4)")]
    UnsupportedNaxis(i64),
    #[error("invalid header beam: {0}")]
    Beam(#[from] BeamError),
    #[error("shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<atfits_rs::AtfitsError> for FitsError {
    fn from(e: atfits_rs::AtfitsError) -> Self {
        use atfits_rs::AtfitsError as A;
        match e {
            A::Fits(e) => FitsError::Fitsio(e),
            A::Io(e) => FitsError::Io(e),
            A::MissingKeyword(s) | A::TargetAxisMissing(s) => FitsError::MissingKeyword(s),
            A::UnsupportedNaxis(n) => FitsError::UnsupportedNaxis(n),
            A::Other(s) => FitsError::Io(std::io::Error::other(s)),
        }
    }
}

/// Header metadata of a 2D image, or of the first plane of a 4D one.
#[derive(Debug, Clone)]
pub struct FitsImageHeader {
    pub path: PathBuf,
    pub nx: usize,
    pub ny: usize,
    /// Planes beyond the first (1 for a plain image).
    pub nplanes: usize,
    /// Pixel size along x in degrees.
    pub dx_deg: f64,
    /// Pixel size along y in degrees.
    pub dy_deg: f64,
    pub beam: Beam,
    pub unit: BrightnessUnit,
    pub dtype: PixelType,
}

pub struct FitsImageData<T = f32> {
    pub header: FitsImageHeader,
    pub image: Array2<T>,
}

/// Read the shape, pixel scale, beam and unit of a 2D or 4D image.
pub fn read_fits_header(path: &Path) -> Result<FitsImageHeader, FitsError> {
    let mut fptr = FitsFile::open(path)?;
    let geom = HeaderGeom::from_file(&mut fptr)?;
    if geom.naxis != 2 && geom.naxis != 4 {
        return Err(FitsError::UnsupportedNaxis(geom.naxis as i64));
    }
    let nplanes = geom.dims[2..].iter().product();
    if nplanes > 1 {
        tracing::warn!(
            "{}: {nplanes} planes beyond the image axes; only the first is smoothed",
            path.display()
        );
    }
    let hdu = fptr.primary_hdu()?;
    let (dx_deg, dy_deg) = pixel_scale_deg(&mut fptr)?;
    let beam = read_header_beam(&mut fptr)?.ok_or(FitsError::MissingKeyword("BMAJ".into()))?;
    Ok(FitsImageHeader {
        path: path.to_path_buf(),
        nx: geom.dims[0],
        ny: geom.dims[1],
        nplanes,
        dx_deg,
        dy_deg,
        beam,
        unit: read_unit(&mut fptr, &hdu, path),
        dtype: PixelType::from_bitpix(geom.bitpix),
    })
}

/// Read a 2D image, or the first plane of a 4D one, in precision `T`.
pub fn read_fits<T: CubeElem>(path: &Path) -> Result<FitsImageData<T>, FitsError> {
    let header = read_fits_header(path)?;
    let image = read_fits_plane(&header)?;
    Ok(FitsImageData { header, image })
}

/// Read the first plane of the image `header` describes.
pub fn read_fits_plane<T: CubeElem>(header: &FitsImageHeader) -> Result<Array2<T>, FitsError> {
    let mut fptr = FitsFile::open(&header.path)?;
    let data = T::read_section(&mut fptr, 0, header.nx * header.ny)?;
    Ok(Array2::from_shape_vec((header.ny, header.nx), data)?)
}

/// Write `image` to `out_path` with the header of `template_path`, the beam
/// set to `new_beam`, and any planes beyond the first dropped.
pub fn write_fits<T: CubeElem>(
    image: &Array2<T>,
    out_path: &Path,
    template_path: &Path,
    new_beam: &Beam,
) -> Result<(), FitsError> {
    let (ny, nx) = image.dim();
    let mut dims = vec![1; HeaderGeom::read(template_path)?.naxis];
    dims[0] = nx;
    dims[1] = ny;
    let mut fptr = create_output(template_path, out_path, Some(&dims))?;
    update_key_f64(&mut fptr, "BMAJ", new_beam.major_deg())?;
    update_key_f64(&mut fptr, "BMIN", new_beam.minor_deg())?;
    update_key_f64(&mut fptr, "BPA", new_beam.pa_deg())?;

    let flat = image.as_standard_layout();
    let slice = flat
        .as_slice()
        .ok_or_else(|| FitsError::Io(std::io::Error::other("image is not contiguous")))?;
    T::write_section(&mut fptr, 0, nx * ny, slice)?;
    Ok(())
}

/// Create `output` with `template`'s header and shape `dims` (default: the
/// template's), open for writing. Integer templates give a float image, since
/// smoothed values are fractional and blanked pixels are NaN.
pub(crate) fn create_output(
    template: &Path,
    output: &Path,
    dims: Option<&[usize]>,
) -> Result<FitsFile, atfits_rs::AtfitsError> {
    let geom = HeaderGeom::read(template)?;
    let bitpix = if geom.bitpix > 0 { -32 } else { geom.bitpix };
    create_cube_open(template, output, bitpix, dims.unwrap_or(&geom.dims))
}

/// BMAJ/BMIN/BPA from the primary header, or `None` without BMAJ. BMIN
/// defaults to BMAJ and BPA to 0.
pub(crate) fn read_header_beam(fptr: &mut FitsFile) -> Result<Option<Beam>, FitsError> {
    let Some(bmaj) = atfits_rs::key_f64(fptr, "BMAJ")? else {
        return Ok(None);
    };
    let bmin = atfits_rs::key_f64(fptr, "BMIN")?.unwrap_or(bmaj);
    let bpa = atfits_rs::key_f64(fptr, "BPA")?.unwrap_or(0.0);
    Ok(Some(Beam::new(bmaj, bmin, bpa)?))
}

/// The brightness unit from BUNIT, warning and assuming Jy/beam if it is
/// missing or not recognised.
pub(crate) fn read_unit(fptr: &mut FitsFile, hdu: &FitsHdu, path: &Path) -> BrightnessUnit {
    match hdu.read_key::<String>(fptr, "BUNIT") {
        Ok(s) => BrightnessUnit::from_bunit(&s),
        Err(_) => {
            tracing::warn!(
                "No BUNIT keyword in {}; assuming Jy/beam (flux scaling applied).",
                path.display()
            );
            BrightnessUnit::default()
        }
    }
}
