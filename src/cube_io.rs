//! FITS spectral cube reading and writing with per-channel beams.
//!
//! Cubes are 3D (RA, Dec, FREQ) or 4D with one Stokes plane, in either axis
//! order. Per-channel beams come from, in order of preference:
//!   1. a CASA BEAMS binary table (`CASAMBM = T`),
//!   2. a beamlog text file `{dir}/beamlog.{stem}.txt`,
//!   3. BMAJ/BMIN/BPA in the primary header, used for every channel.
use std::path::{Path, PathBuf};

use atfits_rs::{key_bool, pixel_scale_deg, update_key_f64, update_key_logical};
use fitsio::{
    FitsFile,
    tables::{ColumnDataType, ColumnDescription},
};
use ndarray::Array2;
use thiserror::Error;

use crate::beam::{Beam, BeamError};
use crate::convolve_uv::FftFloat;
use crate::fits_io::{FitsError, create_output, read_header_beam, read_unit};
use crate::smooth::BrightnessUnit;

pub use atfits_rs::PixelType;

/// Pixel types a plane is read, convolved and written in: `f32` or `f64`.
pub trait CubeElem: atfits_rs::CubeElem + FftFloat {}

impl<T: atfits_rs::CubeElem + FftFloat> CubeElem for T {}

#[derive(Debug, Error)]
pub enum CubeError {
    #[error("FITS I/O error: {0}")]
    Fits(#[from] fitsio::errors::Error),
    #[error(transparent)]
    Header(#[from] FitsError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("invalid beam: {0}")]
    Beam(#[from] BeamError),
    #[error("unsupported NAXIS={0} (expected 3 or 4)")]
    UnsupportedNaxis(i64),
    #[error("missing header keyword: {0}")]
    MissingKeyword(String),
    #[error("channel count mismatch in beam table: expected {expected}, got {got}")]
    BeamCountMismatch { expected: usize, got: usize },
    #[error("beamlog parse error at line {line}: {msg}")]
    BeamlogParse { line: usize, msg: String },
    #[error("no per-channel beam source found (no CASAMBM, no beamlog, no header beam)")]
    NoBeams,
}

impl From<atfits_rs::AtfitsError> for CubeError {
    fn from(e: atfits_rs::AtfitsError) -> Self {
        CubeError::Header(e.into())
    }
}

/// Metadata of a FITS spectral cube.
#[derive(Debug)]
pub struct CubeMeta {
    pub path: PathBuf,
    /// NAXIS1 (RA).
    pub nx: usize,
    /// NAXIS2 (Dec).
    pub ny: usize,
    pub nfreq: usize,
    pub nstokes: usize,
    /// Pixel size along x in degrees.
    pub dx_deg: f64,
    /// Pixel size along y in degrees.
    pub dy_deg: f64,
    /// CRPIX of the spectral axis, truncated to a whole pixel as racs_tools
    /// does. Picks the channel whose beam goes in the output header; may lie
    /// outside `1..=nfreq`.
    pub crpix_freq: f64,
    /// Per-channel beams; `None` for a masked channel.
    pub beams: Vec<Option<Beam>>,
    pub is_4d: bool,
    pub unit: BrightnessUnit,
    pub dtype: PixelType,
}

impl CubeMeta {
    /// Flat pixel range `[start, end)` of channel `chan` (first Stokes plane).
    /// With one Stokes plane both axis orders put channel `c` at `c·ny·nx`.
    pub fn channel_range(&self, chan: usize) -> (usize, usize) {
        let plane = self.ny * self.nx;
        (chan * plane, (chan + 1) * plane)
    }
}

/// The beamlog next to a FITS file: `{dir}/beamlog.{stem}.txt`.
pub fn beamlog_path(fits: &Path) -> PathBuf {
    let dir = fits.parent().unwrap_or(Path::new("."));
    let stem = fits.file_stem().unwrap_or_default();
    dir.join(format!("beamlog.{}.txt", stem.to_string_lossy()))
}

/// Read the shape, pixel scale and per-channel beams of a cube.
pub fn read_cube_meta(path: &Path) -> Result<CubeMeta, CubeError> {
    let mut fptr = FitsFile::open(path)?;
    let hdu = fptr.primary_hdu()?;

    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
    if naxis != 3 && naxis != 4 {
        return Err(CubeError::UnsupportedNaxis(naxis));
    }
    let mut dims = Vec::with_capacity(naxis as usize);
    for i in 1..=naxis {
        dims.push(hdu.read_key::<i64>(&mut fptr, &format!("NAXIS{i}"))? as usize);
    }

    // CASA writes RA, DEC, STOKES, FREQ; ASKAP writes RA, DEC, FREQ, STOKES.
    let ctype3 = hdu
        .read_key::<String>(&mut fptr, "CTYPE3")
        .map(|s| s.trim().to_ascii_uppercase())
        .unwrap_or_default();
    let (nstokes, nfreq, freq_axis) = match dims[..] {
        [_, _, n3, n4] if ctype3 == "STOKES" => (n3, n4, 4),
        [_, _, n3, n4] => (n4, n3, 3),
        _ => (1, dims[2], 3),
    };

    let (dx_deg, dy_deg) = pixel_scale_deg(&mut fptr)?;
    let crpix_freq = atfits_rs::key_f64(&mut fptr, &format!("CRPIX{freq_axis}"))?
        .unwrap_or(1.0)
        .floor();
    let bitpix: i64 = hdu.read_key(&mut fptr, "BITPIX")?;
    let unit = read_unit(&mut fptr, &hdu, path);

    // CASA writes CASAMBM as a logical, but some tools write a string.
    let casambm = key_bool(&mut fptr, "CASAMBM")?.unwrap_or(false);

    let beamlog = beamlog_path(path);
    let beams = if casambm {
        read_casambm_beams(&mut fptr, nfreq, path)?
    } else if beamlog.exists() {
        let parsed = read_beamlog(&beamlog)?;
        if parsed.len() != nfreq {
            return Err(CubeError::BeamCountMismatch {
                expected: nfreq,
                got: parsed.len(),
            });
        }
        parsed.into_iter().map(Some).collect()
    } else {
        let beam = read_header_beam(&mut fptr)?.ok_or(CubeError::NoBeams)?;
        vec![Some(beam); nfreq]
    };

    Ok(CubeMeta {
        path: path.to_path_buf(),
        nx: dims[0],
        ny: dims[1],
        nfreq,
        nstokes,
        dx_deg,
        dy_deg,
        crpix_freq,
        beams,
        is_4d: naxis == 4,
        unit,
        dtype: PixelType::from_bitpix(bitpix),
    })
}

/// Per-channel beams from the CASA BEAMS table (BMAJ, BMIN in arcsec; BPA in
/// degrees).
fn read_casambm_beams(
    fptr: &mut FitsFile,
    nfreq: usize,
    path: &Path,
) -> Result<Vec<Option<Beam>>, CubeError> {
    let hdu = fptr
        .hdu("BEAMS")
        .map_err(|_| CubeError::MissingKeyword("BEAMS extension".into()))?;
    let bmaj: Vec<f32> = hdu.read_col(fptr, "BMAJ")?;
    let bmin: Vec<f32> = hdu.read_col(fptr, "BMIN")?;
    let bpa: Vec<f32> = hdu.read_col(fptr, "BPA")?;
    if bmaj.len() != nfreq || bmin.len() != nfreq || bpa.len() != nfreq {
        return Err(CubeError::BeamCountMismatch {
            expected: nfreq,
            got: bmaj.len(),
        });
    }
    Ok((0..nfreq)
        .map(|c| {
            beam_from_row(
                f64::from(bmaj[c]),
                f64::from(bmin[c]),
                f64::from(bpa[c]),
                c,
                path,
            )
        })
        .collect())
}

/// Beam from one row of a beam table (arcsec, arcsec, deg).
///
/// `None` for a masked channel: a non-finite value, or an axis no larger than
/// the `f32::MIN_POSITIVE` written for masked channels. Swapped axes are
/// fixed with a warning.
fn beam_from_row(maj: f64, min: f64, pa: f64, chan: usize, path: &Path) -> Option<Beam> {
    let tiny = f64::from(f32::MIN_POSITIVE);
    if !(maj.is_finite() && min.is_finite() && pa.is_finite()) || maj <= tiny || min <= tiny {
        return None;
    }
    if min > maj {
        tracing::warn!(
            "{}: channel {chan} has BMIN > BMAJ ({min}\" > {maj}\"); swapping the axes",
            path.display()
        );
        let pa = if pa > 0.0 { pa - 90.0 } else { pa + 90.0 };
        return Beam::from_arcsec(min, maj, pa).ok();
    }
    Beam::from_arcsec(maj, min, pa).ok()
}

/// An open input cube for reading channels. cfitsio handles are not `Send`,
/// so each thread opens its own.
pub struct CubeReader {
    fptr: FitsFile,
}

impl CubeReader {
    pub fn open(path: &Path) -> Result<Self, CubeError> {
        Ok(Self {
            fptr: FitsFile::open(path)?,
        })
    }

    /// Channel `chan` (first Stokes plane) as an `ny × nx` array.
    pub fn read_channel<T: CubeElem>(
        &mut self,
        chan: usize,
        meta: &CubeMeta,
    ) -> Result<Array2<T>, CubeError> {
        let (start, end) = meta.channel_range(chan);
        let data = T::read_section(&mut self.fptr, start, end)?;
        Ok(Array2::from_shape_vec((meta.ny, meta.nx), data)?)
    }
}

/// Mode for choosing the common beam of a cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeMode {
    /// A common beam per channel, written to a BEAMS table.
    Natural,
    /// One common beam for every channel, written to the primary header.
    Total,
}

/// An output cube held open while channels are written, so cfitsio writes the
/// data unit once, on [`CubeWriter::finish`]. Use from a single thread.
pub struct CubeWriter {
    fptr: FitsFile,
    /// Natural mode's BEAMS table. Creating it makes cfitsio flush the primary
    /// data unit, so it is written after the channels.
    pending_beams: Option<Vec<Option<Beam>>>,
}

impl CubeWriter {
    /// Create `output_path` with the header of `input_path` and the beam
    /// keywords for `target_beams`.
    pub fn create(
        input_path: &Path,
        output_path: &Path,
        target_beams: &[Option<Beam>],
        mode: CubeMode,
        meta: &CubeMeta,
    ) -> Result<Self, CubeError> {
        let mut fptr = create_output(input_path, output_path, None)?;
        write_primary_beam_keys(
            &mut fptr,
            ref_beam_for(target_beams, meta),
            mode == CubeMode::Natural,
        )?;
        Ok(Self {
            fptr,
            pending_beams: (mode == CubeMode::Natural).then(|| target_beams.to_vec()),
        })
    }

    pub fn write_channel<T: CubeElem>(
        &mut self,
        chan: usize,
        data: &Array2<T>,
        meta: &CubeMeta,
    ) -> Result<(), CubeError> {
        let (start, end) = meta.channel_range(chan);
        let flat = data.as_standard_layout();
        let slice = flat
            .as_slice()
            .ok_or_else(|| std::io::Error::other("plane is not contiguous"))?;
        T::write_section(&mut self.fptr, start, end, slice)?;
        Ok(())
    }

    /// Append the BEAMS table (natural mode) and close the file. Channels never
    /// written read as zero.
    pub fn finish(mut self) -> Result<(), CubeError> {
        if let Some(beams) = self.pending_beams.take() {
            write_beams_table(&mut self.fptr, &beams)?;
        }
        Ok(())
    }
}

/// The beam for the primary header: the target at the spectral CRPIX (clamped
/// to the channel range), else the first valid one, else zero.
fn ref_beam_for(target_beams: &[Option<Beam>], meta: &CubeMeta) -> Beam {
    let last = meta.nfreq.saturating_sub(1) as f64;
    let ref_idx = (meta.crpix_freq - 1.0).clamp(0.0, last) as usize;
    target_beams[ref_idx]
        .or_else(|| target_beams.iter().find_map(|b| *b))
        .unwrap_or(Beam::zero())
}

/// Set BMAJ/BMIN/BPA and CASAMBM in the primary header, replacing the copied
/// cards. CASAMBM must be a logical: casacore reads it with `asBool`.
fn write_primary_beam_keys(
    fptr: &mut FitsFile,
    ref_beam: Beam,
    natural: bool,
) -> Result<(), CubeError> {
    fptr.primary_hdu()?;
    update_key_f64(fptr, "BMAJ", ref_beam.major_deg())?;
    update_key_f64(fptr, "BMIN", ref_beam.minor_deg())?;
    update_key_f64(fptr, "BPA", ref_beam.pa_deg())?;
    update_key_logical(fptr, "CASAMBM", natural)?;
    Ok(())
}

/// Append a CASA BEAMS table. Masked channels get `f32::MIN_POSITIVE`, which
/// [`read_cube_meta`] reads back as masked.
fn write_beams_table(fptr: &mut FitsFile, target_beams: &[Option<Beam>]) -> Result<(), CubeError> {
    let nfreq = target_beams.len();
    let tiny = f32::MIN_POSITIVE;
    let column = |f: fn(&Beam) -> f64| -> Vec<f32> {
        target_beams
            .iter()
            .map(|b| b.map_or(tiny, |b| f(&b) as f32))
            .collect()
    };
    let bmaj = column(Beam::major_arcsec);
    let bmin = column(Beam::minor_arcsec);
    let bpa = column(Beam::pa_deg);
    let chan: Vec<i32> = (0..nfreq as i32).collect();
    let pol = vec![0i32; nfreq];

    let col = |name: &str, ty: ColumnDataType| ColumnDescription::new(name).with_type(ty).create();
    let cols = [
        col("BMAJ", ColumnDataType::Float)?,
        col("BMIN", ColumnDataType::Float)?,
        col("BPA", ColumnDataType::Float)?,
        col("CHAN", ColumnDataType::Int)?,
        col("POL", ColumnDataType::Int)?,
    ];
    let table = fptr.create_table("BEAMS", &cols)?;
    table.write_col(fptr, "BMAJ", &bmaj)?;
    table.write_col(fptr, "BMIN", &bmin)?;
    table.write_col(fptr, "BPA", &bpa)?;
    table.write_col(fptr, "CHAN", &chan)?;
    table.write_col(fptr, "POL", &pol)?;

    // casacore needs the column units.
    let table = fptr.hdu("BEAMS")?;
    table.write_key(fptr, "TUNIT1", "arcsec")?;
    table.write_key(fptr, "TUNIT2", "arcsec")?;
    table.write_key(fptr, "TUNIT3", "deg")?;
    table.write_key(fptr, "NCHAN", nfreq as i64)?;
    table.write_key(fptr, "NPOL", 1i64)?;
    Ok(())
}

/// Read a beamlog as written by RACS-tools or [`write_beamlog`]:
///
/// ```text
/// # Channel BMAJ[arcsec] BMIN[arcsec] BPA[deg]
/// 0  20.0  10.0  10.0
/// ```
///
/// Masked rows come back as [`Beam::zero`].
pub fn read_beamlog(path: &Path) -> Result<Vec<Beam>, CubeError> {
    let content = std::fs::read_to_string(path)?;
    let mut beams = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(CubeError::BeamlogParse {
                line: i + 1,
                msg: format!("expected 4 fields, got {}", fields.len()),
            });
        }
        let parse = |s: &str, n: &str| -> Result<f64, CubeError> {
            s.parse::<f64>().map_err(|_| CubeError::BeamlogParse {
                line: i + 1,
                msg: format!("cannot parse {n}={s:?} as float"),
            })
        };
        let (bmaj, bmin, bpa) = (
            parse(fields[1], "BMAJ")?,
            parse(fields[2], "BMIN")?,
            parse(fields[3], "BPA")?,
        );
        let chan = beams.len();
        beams.push(beam_from_row(bmaj, bmin, bpa, chan, path).unwrap_or(Beam::zero()));
    }
    Ok(beams)
}

/// Write per-channel beams as a beamlog; masked channels are written as NaN.
pub fn write_beamlog(path: &Path, beams: &[Option<Beam>]) -> Result<(), CubeError> {
    let mut out = String::from("# Channel BMAJ[arcsec] BMIN[arcsec] BPA[deg]\n");
    for (i, b) in beams.iter().enumerate() {
        out += &match b {
            Some(b) => format!(
                "{i} {} {} {}\n",
                b.major_arcsec(),
                b.minor_arcsec(),
                b.pa_deg()
            ),
            None => format!("{i} nan nan nan\n"),
        };
    }
    std::fs::write(path, out)?;
    Ok(())
}
