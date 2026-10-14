//! FITS image reading and writing.
//!
//! Handles 2D images (NAXIS=2) and 4D images (NAXIS=4) as produced by
//! ASKAP/CASA. Only the first plane of a 4D image is used, as racs_tools does.
use std::path::{Path, PathBuf};

use atfits_rs::{PixelType, copy_header_only_open, resize_image, update_key_f64};
use fitsio::FitsFile;
use fitsio::hdu::FitsHdu;
use ndarray::Array2;
use thiserror::Error;

use crate::beam::Beam;
use crate::cube_io::CubeElem;
use crate::smooth::BrightnessUnit;

#[derive(Debug, Error)]
pub enum FitsError {
    #[error("FITS I/O error: {0}")]
    Fitsio(#[from] fitsio::errors::Error),
    #[error("missing keyword: {0}")]
    MissingKeyword(String),
    #[error("unsupported NAXIS={0} (expected 2 or 4)")]
    UnsupportedNaxis(i64),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Map the shared [`atfits_rs::AtfitsError`] onto this module's error type.
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
    /// Number of planes behind the first one's axes (1 for a plain image).
    pub nplanes: usize,
    pub is_4d: bool,
    /// |CDELT1| in degrees (x / RA pixel size)
    pub dx_deg: f64,
    /// |CDELT2| in degrees (y / Dec pixel size)
    pub dy_deg: f64,
    pub beam: Beam,
    /// Brightness unit from BUNIT (defaults to Jy/beam if absent).
    pub unit: BrightnessUnit,
    /// Pixel precision, from BITPIX.
    pub dtype: PixelType,
}

pub struct FitsImageData<T = f32> {
    pub path: PathBuf,
    pub image: Array2<T>,
    pub is_4d: bool,
    /// |CDELT1| in degrees (x / RA pixel size)
    pub dx_deg: f64,
    /// |CDELT2| in degrees (y / Dec pixel size)
    pub dy_deg: f64,
    pub beam: Beam,
    /// Brightness unit from BUNIT (defaults to Jy/beam if absent).
    pub unit: BrightnessUnit,
    /// Full header keyword list (key, value strings) for re-writing.
    pub header_cards: Vec<(String, String)>,
}

/// Read the shape, pixel scale, beam and unit of a 2D or 4D FITS image without
/// reading its pixels.
pub fn read_fits_header(path: &Path) -> Result<FitsImageHeader, FitsError> {
    let mut fptr = FitsFile::open(path.to_string_lossy().as_ref())?;
    let hdu = fptr.primary_hdu()?;

    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
    if naxis != 2 && naxis != 4 {
        return Err(FitsError::UnsupportedNaxis(naxis));
    }
    let mut dims = Vec::with_capacity(naxis as usize);
    for i in 1..=naxis {
        dims.push(hdu.read_key::<i64>(&mut fptr, &format!("NAXIS{i}"))? as usize);
    }
    let nplanes = dims[2..].iter().product();
    if nplanes > 1 {
        tracing::warn!(
            "{}: {nplanes} planes beyond the image axes; only the first is smoothed",
            path.display()
        );
    }
    let bitpix: i64 = hdu.read_key(&mut fptr, "BITPIX")?;

    let (dx_deg, dy_deg) = pixel_scales(&mut fptr, &hdu).map_err(FitsError::MissingKeyword)?;

    let bmaj: f64 = hdu
        .read_key(&mut fptr, "BMAJ")
        .map_err(|_| FitsError::MissingKeyword("BMAJ".into()))?;
    let bmin: f64 = hdu.read_key(&mut fptr, "BMIN").unwrap_or(bmaj);
    let bpa: f64 = hdu.read_key(&mut fptr, "BPA").unwrap_or(0.0);
    let beam = Beam::new(bmaj, bmin, bpa)
        .map_err(|e| FitsError::Io(std::io::Error::other(e.to_string())))?;

    // Brightness unit (BUNIT); warn and default to Jy/beam when absent.
    let unit = match hdu.read_key::<String>(&mut fptr, "BUNIT") {
        Ok(s) => BrightnessUnit::from_bunit(&s),
        Err(_) => {
            tracing::warn!(
                "No BUNIT keyword in {}; assuming Jy/beam (flux scaling applied).",
                path.display()
            );
            BrightnessUnit::default()
        }
    };

    Ok(FitsImageHeader {
        path: path.to_path_buf(),
        nx: dims[0],
        ny: dims[1],
        nplanes,
        is_4d: naxis == 4,
        dx_deg,
        dy_deg,
        beam,
        unit,
        dtype: PixelType::from_bitpix(bitpix),
    })
}

/// Read a 2D FITS image, or the first plane of a 4D one, in precision `T`.
pub fn read_fits<T: CubeElem>(path: &Path) -> Result<FitsImageData<T>, FitsError> {
    let header = read_fits_header(path)?;
    let image = read_fits_plane(&header)?;
    Ok(FitsImageData {
        path: header.path,
        image,
        is_4d: header.is_4d,
        dx_deg: header.dx_deg,
        dy_deg: header.dy_deg,
        beam: header.beam,
        unit: header.unit,
        header_cards: vec![],
    })
}

/// Read the first plane of the image described by `header`.
pub fn read_fits_plane<T: CubeElem>(header: &FitsImageHeader) -> Result<Array2<T>, FitsError> {
    let mut fptr = FitsFile::open(header.path.to_string_lossy().as_ref())?;
    let data = T::read_section_vec(&mut fptr, 0, header.nx * header.ny)
        .map_err(|e| FitsError::Io(std::io::Error::other(e.to_string())))?;
    Array2::from_shape_vec((header.ny, header.nx), data)
        .map_err(|e| FitsError::Io(std::io::Error::other(e.to_string())))
}

/// `(|CDELT1|, |CDELT2|)` in degrees. Headers that only carry a CD matrix get
/// an error naming the missing keyword.
pub(crate) fn pixel_scales(fptr: &mut FitsFile, hdu: &FitsHdu) -> Result<(f64, f64), String> {
    let mut read = |key: &str| -> Result<f64, String> {
        hdu.read_key::<f64>(fptr, key).map(f64::abs).map_err(|_| {
            if hdu.read_key::<f64>(fptr, "CD1_1").is_ok() {
                format!("{key} (CD-matrix headers are not supported; add CDELT1/CDELT2)")
            } else {
                key.to_string()
            }
        })
    };
    Ok((read("CDELT1")?, read("CDELT2")?))
}

/// Write a smoothed image to `out_path`.
///
/// The header is copied from `template_path`, with BMAJ/BMIN/BPA replaced and
/// any planes beyond the first dropped.
pub fn write_fits<T: CubeElem>(
    image: &Array2<T>,
    out_path: &Path,
    template_path: &Path,
    new_beam: &Beam,
    _was_4d: bool,
) -> Result<(), FitsError> {
    let mut fptr = copy_header_only_open(template_path, out_path)?;
    let hdu = fptr.primary_hdu()?;
    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
    let bitpix: i64 = hdu.read_key(&mut fptr, "BITPIX")?;
    let (ny, nx) = image.dim();
    if naxis > 2 {
        let mut dims = vec![1; naxis as usize];
        dims[0] = nx;
        dims[1] = ny;
        resize_image(&mut fptr, bitpix, &dims)?;
    }

    // The template already has these cards; appending would leave the old
    // beam first, which is the one readers pick up.
    update_key_f64(&mut fptr, "BMAJ", new_beam.major_deg)?;
    update_key_f64(&mut fptr, "BMIN", new_beam.minor_deg)?;
    update_key_f64(&mut fptr, "BPA", new_beam.pa_deg)?;

    let flat = image.as_standard_layout();
    let slice = flat
        .as_slice()
        .ok_or_else(|| FitsError::Io(std::io::Error::other("image is not contiguous")))?;
    T::write_section_vec(&mut fptr, 0, nx * ny, slice)
        .map_err(|e| FitsError::Io(std::io::Error::other(e.to_string())))?;
    Ok(())
}

/// Build the output path from the input path with an optional suffix/prefix/outdir
/// (re-exported from [`atfits_rs`]).
pub use atfits_rs::output_path;
