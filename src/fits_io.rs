//! FITS image reading and writing.
//!
//! Handles both 2D (NAXIS=2, shape `[ny, nx]`) and 4D (NAXIS=4, shape
//! `[1,1,ny,nx]`) images as produced by ASKAP/CASA.
use std::path::{Path, PathBuf};

use fitsio::FitsFile;
use ndarray::Array2;
use thiserror::Error;

use crate::beam::Beam;
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

pub struct FitsImageData {
    pub path: PathBuf,
    pub image: Array2<f32>,
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

/// Read a 2D or 4D FITS image.
pub fn read_fits(path: &Path) -> Result<FitsImageData, FitsError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str)?;
    let hdu = fptr.primary_hdu()?;

    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
    let naxis1: i64 = hdu.read_key(&mut fptr, "NAXIS1")?; // x / RA (cols)
    let naxis2: i64 = hdu.read_key(&mut fptr, "NAXIS2")?; // y / Dec (rows)

    if naxis != 2 && naxis != 4 {
        return Err(FitsError::UnsupportedNaxis(naxis));
    }

    let nx = naxis1 as usize;
    let ny = naxis2 as usize;

    // Read flat image data (works for both 2D and 4D when extra axes are size 1).
    let data: Vec<f32> = hdu.read_image(&mut fptr)?;
    let image = Array2::from_shape_vec((ny, nx), data)
        .map_err(|e| FitsError::Io(std::io::Error::other(e.to_string())))?;

    // Pixel scales — use absolute values since CDELT1 is negative for RA.
    let cdelt1: f64 = hdu.read_key(&mut fptr, "CDELT1")?;
    let cdelt2: f64 = hdu.read_key(&mut fptr, "CDELT2")?;
    let dx_deg = cdelt1.abs();
    let dy_deg = cdelt2.abs();

    // Beam.
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

    Ok(FitsImageData {
        path: path.to_path_buf(),
        image,
        is_4d: naxis == 4,
        dx_deg,
        dy_deg,
        beam,
        unit,
        header_cards: vec![],
    })
}

/// Write a smoothed image to `out_path`.
///
/// Copies `template_path` to `out_path`, then overwrites the pixel data and
/// updates BMAJ/BMIN/BPA in the header.  This preserves all other keywords
/// (WCS, HISTORY, etc.) from the original file.
pub fn write_fits(
    image: &Array2<f32>,
    out_path: &Path,
    template_path: &Path,
    new_beam: &Beam,
    _was_4d: bool,
) -> Result<(), FitsError> {
    // Initialise the destination cheaply: copy only the template's primary-HDU
    // header (preserving WCS/HISTORY/etc.), not its pixel data.  The data unit is
    // defined by the copied NAXIS keywords; we overwrite every pixel below, so
    // reading the template's data via `std::fs::copy` would be wasted I/O.
    copy_header_only(template_path, out_path)?;

    let out_str = out_path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::edit(&out_str)?;
    let hdu = fptr.primary_hdu()?;

    // Flatten to Vec<f32> in row-major order (C order = FITS row-major).
    let flat: Vec<f32> = image.iter().cloned().collect();
    hdu.write_image(&mut fptr, &flat)?;

    // Update beam keywords.
    hdu.write_key(&mut fptr, "BMAJ", new_beam.major_deg)?;
    hdu.write_key(&mut fptr, "BMIN", new_beam.minor_deg)?;
    hdu.write_key(&mut fptr, "BPA", new_beam.pa_deg)?;

    Ok(())
}

/// Create `output` containing only the primary-HDU header of `input` — no pixel data.
///
/// Uses cfitsio `fits_copy_header` (ffcphd): the output data unit is defined by the
/// copied NAXIS keywords and zero-filled (sparsely) on close, avoiding a full read of
/// the template's pixel data.
fn copy_header_only(input: &Path, output: &Path) -> Result<(), FitsError> {
    let mut in_fptr = FitsFile::open(input.to_string_lossy().into_owned())?;
    in_fptr.primary_hdu()?; // position at the primary HDU to copy
    let mut out_fptr = FitsFile::create(output).overwrite().open()?;

    let mut status = 0;
    unsafe {
        fitsio::sys::ffcphd(in_fptr.as_raw(), out_fptr.as_raw(), &mut status);
    }
    fitsio::errors::check_status(status)?;
    Ok(())
}

/// Build the output path from the input path with an optional suffix/prefix/outdir.
pub fn output_path(
    input: &Path,
    suffix: Option<&str>,
    prefix: Option<&str>,
    outdir: Option<&Path>,
) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = input.extension().unwrap_or_default().to_string_lossy();

    let filename = match suffix {
        Some(s) => format!("{stem}.{s}.{ext}"),
        None => format!("{stem}.{ext}"),
    };
    let filename = match prefix {
        Some(p) => format!("{p}{filename}"),
        None => filename,
    };

    let dir = outdir.unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")));
    dir.join(filename)
}
