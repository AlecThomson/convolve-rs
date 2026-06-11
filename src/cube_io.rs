//! FITS spectral cube reading and writing with per-channel beam support.
//!
//! Supports 3D cubes (NAXIS=3: freq×dec×ra) and 4D cubes (NAXIS=4: stokes×freq×dec×ra).
//! Per-channel beams are read from, in priority order:
//!   1. CASA BEAMS binary-table extension (CASAMBM=T in header)
//!   2. Co-located beamlog text file: `{dir}/beamlog.{stem}.txt`
//!   3. Single BMAJ/BMIN/BPA from the primary header (broadcast to all channels)
use std::path::{Path, PathBuf};

use fitsio::{
    FitsFile,
    tables::{ColumnDataType, ColumnDescription},
};
use ndarray::Array2;
use thiserror::Error;

use crate::beam::{Beam, BeamError};
use crate::smooth::BrightnessUnit;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum CubeError {
    #[error("FITS I/O error: {0}")]
    Fits(#[from] fitsio::errors::Error),
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
    #[error("channel count mismatch in BEAMS extension: expected {expected}, got {got}")]
    BeamCountMismatch { expected: usize, got: usize },
    #[error("beamlog parse error at line {line}: {msg}")]
    BeamlogParse { line: usize, msg: String },
    #[error("no per-channel beam source found (no CASAMBM, no beamlog, no header beam)")]
    NoBeans,
}

// ── Public metadata struct ────────────────────────────────────────────────────

/// Metadata for a FITS spectral cube (3D or 4D).
#[derive(Debug)]
pub struct CubeMeta {
    pub path: PathBuf,
    /// Fastest-varying spatial axis size (RA/x pixels).
    pub nx: usize,
    /// Slower spatial axis size (Dec/y pixels).
    pub ny: usize,
    /// Number of frequency channels.
    pub nfreq: usize,
    /// Number of Stokes planes (1 for most ASKAP data).
    pub nstokes: usize,
    /// |CDELT1| in degrees.
    pub dx_deg: f64,
    /// |CDELT2| in degrees.
    pub dy_deg: f64,
    /// FITS 1-based CRPIX for the spectral axis (used as the header reference channel).
    pub crpix_freq: i64,
    /// Per-channel beams.  `None` means the channel is masked / has no valid beam.
    pub beams: Vec<Option<Beam>>,
    /// True for 4D input (has a Stokes axis in the header).
    pub is_4d: bool,
    /// Brightness unit from BUNIT (defaults to Jy/beam if absent).
    pub unit: BrightnessUnit,
}

impl CubeMeta {
    /// Beamlog path co-located with the FITS file.
    pub fn beamlog_path(&self) -> PathBuf {
        let dir = self.path.parent().unwrap_or(Path::new("."));
        let stem = self.path.file_stem().unwrap_or_default();
        dir.join(format!("beamlog.{}.txt", stem.to_string_lossy()))
    }
}

// ── Reading cube metadata ─────────────────────────────────────────────────────

/// Read metadata (shape, pixel scale, per-channel beams) from a FITS cube.
pub fn read_cube_meta(path: &Path) -> Result<CubeMeta, CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str)?;
    let hdu = fptr.primary_hdu()?;

    let naxis: i64 = hdu.read_key(&mut fptr, "NAXIS")?;
    if naxis != 3 && naxis != 4 {
        return Err(CubeError::UnsupportedNaxis(naxis));
    }

    let naxis1: i64 = hdu.read_key(&mut fptr, "NAXIS1")?; // x / RA
    let naxis2: i64 = hdu.read_key(&mut fptr, "NAXIS2")?; // y / Dec
    let naxis3: i64 = hdu.read_key(&mut fptr, "NAXIS3")?; // freq

    let (nstokes, nfreq, is_4d) = if naxis == 4 {
        let naxis4: i64 = hdu.read_key(&mut fptr, "NAXIS4")?;
        (naxis4 as usize, naxis3 as usize, true)
    } else {
        (1, naxis3 as usize, false)
    };

    let nx = naxis1 as usize;
    let ny = naxis2 as usize;

    let cdelt1: f64 = hdu.read_key(&mut fptr, "CDELT1")?;
    let cdelt2: f64 = hdu.read_key(&mut fptr, "CDELT2")?;
    let dx_deg = cdelt1.abs();
    let dy_deg = cdelt2.abs();

    // Reference channel for the spectral axis (CRPIX3 for 3D, CRPIX3 for 4D where freq=axis 3)
    let crpix_freq: i64 = hdu.read_key(&mut fptr, "CRPIX3").unwrap_or(1);

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

    // Check for CASAMBM.  CASA/beamcon write it as a FITS logical, but some tools
    // (and our own older outputs) wrote a quoted string — accept both.
    let casambm = hdu
        .read_key::<bool>(&mut fptr, "CASAMBM")
        .ok()
        .or_else(|| {
            hdu.read_key::<String>(&mut fptr, "CASAMBM")
                .ok()
                .map(|s| matches!(s.trim(), "T" | "TRUE"))
        })
        .unwrap_or(false);
    drop(fptr); // close for next reads

    let beams: Vec<Option<Beam>> = if casambm {
        read_casambm_beams(path, nfreq)?
    } else {
        let beamlog = CubeMeta {
            path: path.to_path_buf(),
            nx,
            ny,
            nfreq,
            nstokes,
            dx_deg,
            dy_deg,
            crpix_freq,
            beams: vec![],
            is_4d,
            unit,
        }
        .beamlog_path();

        if beamlog.exists() {
            let parsed = read_beamlog(&beamlog)?;
            if parsed.len() != nfreq {
                return Err(CubeError::BeamCountMismatch {
                    expected: nfreq,
                    got: parsed.len(),
                });
            }
            parsed.into_iter().map(Some).collect()
        } else {
            // Fall back to single header beam broadcast to all channels.
            let mut fptr2 = FitsFile::open(path.to_string_lossy().into_owned())?;
            let hdu2 = fptr2.primary_hdu()?;
            let bmaj: f64 = hdu2
                .read_key(&mut fptr2, "BMAJ")
                .map_err(|_| CubeError::NoBeans)?;
            let bmin: f64 = hdu2.read_key(&mut fptr2, "BMIN").unwrap_or(bmaj);
            let bpa: f64 = hdu2.read_key(&mut fptr2, "BPA").unwrap_or(0.0);
            let b = Beam::new(bmaj, bmin, bpa)?;
            vec![Some(b); nfreq]
        }
    };

    Ok(CubeMeta {
        path: path.to_path_buf(),
        nx,
        ny,
        nfreq,
        nstokes,
        dx_deg,
        dy_deg,
        crpix_freq,
        beams,
        is_4d,
        unit,
    })
}

/// Read per-channel beams from the CASA BEAMS binary-table extension.
///
/// Columns: BMAJ [arcsec], BMIN [arcsec], BPA [deg], CHAN [int], POL [int].
fn read_casambm_beams(path: &Path, nfreq: usize) -> Result<Vec<Option<Beam>>, CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str)?;
    let hdu = fptr
        .hdu("BEAMS")
        .map_err(|_| CubeError::MissingKeyword("BEAMS extension".into()))?;

    let bmaj: Vec<f32> = hdu.read_col(&mut fptr, "BMAJ")?;
    let bmin: Vec<f32> = hdu.read_col(&mut fptr, "BMIN")?;
    let bpa: Vec<f32> = hdu.read_col(&mut fptr, "BPA")?;

    if bmaj.len() != nfreq {
        return Err(CubeError::BeamCountMismatch {
            expected: nfreq,
            got: bmaj.len(),
        });
    }

    let tiny = f32::MIN_POSITIVE as f64;
    let beams = bmaj
        .iter()
        .zip(bmin.iter())
        .zip(bpa.iter())
        .map(|((&maj_as, &min_as), &pa_deg)| {
            let maj_deg = maj_as as f64 / 3600.0;
            let min_deg = min_as as f64 / 3600.0;
            let pa = pa_deg as f64;
            // Treat tiny/zero beams as masked.
            if maj_deg < tiny || !maj_deg.is_finite() {
                None
            } else {
                Beam::new(maj_deg, min_deg.max(tiny), pa).ok()
            }
        })
        .collect();
    Ok(beams)
}

// ── Reading / writing channel planes ─────────────────────────────────────────

/// Read a single frequency channel from a cube into a 2D array (ny × nx).
///
/// Reads stokes=0 (the first Stokes plane).  For 3D [nfreq, ny, nx] and 4D
/// [nstokes=1, nfreq, ny, nx] cubes the flat offset is identical: `chan * ny * nx`.
pub fn read_channel(path: &Path, chan: usize, meta: &CubeMeta) -> Result<Array2<f32>, CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str)?;
    let hdu = fptr.primary_hdu()?;

    let plane = meta.ny * meta.nx;
    let start = chan * plane;
    let end = start + plane;

    let data: Vec<f32> = hdu.read_section(&mut fptr, start, end)?;
    Ok(Array2::from_shape_vec((meta.ny, meta.nx), data)?)
}

/// Write a single frequency channel plane back into an existing FITS cube.
///
/// The output cube must have already been initialised by `init_output_cube`.
pub fn write_channel(
    path: &Path,
    chan: usize,
    data: &Array2<f32>,
    meta: &CubeMeta,
) -> Result<(), CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::edit(&path_str)?;
    let hdu = fptr.primary_hdu()?;

    let plane = meta.ny * meta.nx;
    let start = chan * plane;
    let end = start + plane;

    let flat: Vec<f32> = data.iter().copied().collect();
    hdu.write_section(&mut fptr, start, end, &flat)?;
    Ok(())
}

/// A streaming writer that holds an initialised output cube open for the lifetime
/// of a processing run, so channels can be written one at a time without the
/// per-call file open/close overhead of [`write_channel`].
///
/// cfitsio drives a single file through one internal cursor and is **not**
/// thread-safe, so a `CubeWriter` must be owned and driven by a single thread
/// (the consumer end of the streaming pipeline in `main`).
pub struct CubeWriter {
    fptr: FitsFile,
}

impl CubeWriter {
    /// Open an already-initialised output cube (see [`init_output_cube`]) for
    /// sequential channel writes.
    pub fn open(path: &Path) -> Result<Self, CubeError> {
        let fptr = FitsFile::edit(path.to_string_lossy().into_owned())?;
        Ok(Self { fptr })
    }

    /// Write one frequency channel plane into the open cube.
    pub fn write_channel(
        &mut self,
        chan: usize,
        data: &Array2<f32>,
        meta: &CubeMeta,
    ) -> Result<(), CubeError> {
        let hdu = self.fptr.primary_hdu()?;
        let plane = meta.ny * meta.nx;
        let start = chan * plane;
        let end = start + plane;
        let flat: Vec<f32> = data.iter().copied().collect();
        hdu.write_section(&mut self.fptr, start, end, &flat)?;
        Ok(())
    }
}

// ── Output cube initialisation ────────────────────────────────────────────────

/// Mode for common-beam determination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeMode {
    /// Each channel gets its own common beam (written to BEAMS extension).
    Natural,
    /// All channels share a single common beam (written to primary header only).
    Total,
}

/// Create `output` containing only the primary-HDU header of `input` — no pixel data.
///
/// Uses cfitsio `fits_copy_header` (ffcphd): the output data unit is defined by the
/// copied NAXIS keywords and zero-filled (sparsely) when the file is closed.  This is
/// far cheaper than `std::fs::copy` for large cubes because the input pixel data is
/// never read or written.
fn copy_header_only(input: &Path, output: &Path) -> Result<(), CubeError> {
    use std::ffi::CString;

    let mut in_fptr = FitsFile::open(input.to_string_lossy().into_owned())?;
    in_fptr.primary_hdu()?; // position at the primary HDU to copy

    // Create a *truly empty* output file with cfitsio `fits_create_file` (ffinit):
    // it has no HDUs yet, so `fits_copy_header` (ffcphd) below initialises the
    // primary HDU directly from the copied header.  We cannot use
    // `FitsFile::create().open()` here — that eagerly writes a default empty
    // (NAXIS=0) primary HDU, and ffcphd then copies *nothing* into the already
    // existing primary, leaving a zero-dimensional image.  Writing pixel data to
    // such an HDU later fails with the misleading cfitsio error 302
    // ("column number < 1 or > tfields").
    if output.exists() {
        std::fs::remove_file(output)?;
    }
    let out_name = CString::new(output.to_string_lossy().into_owned())
        .map_err(|e| CubeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;

    let mut status = 0;
    let mut raw_out: *mut fitsio::sys::fitsfile = std::ptr::null_mut();
    unsafe {
        fitsio::sys::ffinit(&mut raw_out, out_name.as_ptr(), &mut status);
        fitsio::errors::check_status(status)?;

        fitsio::sys::ffcphd(in_fptr.as_raw(), raw_out, &mut status);
        let copy_status = fitsio::errors::check_status(status);

        // Always close the raw output pointer, even if the copy failed.
        let mut close_status = 0;
        fitsio::sys::ffclos(raw_out, &mut close_status);
        copy_status?;
        fitsio::errors::check_status(close_status)?;
    }
    Ok(())
}

/// Update (or create) a floating-point header keyword *in place*.
///
/// fitsio's `write_key` calls cfitsio `ffpky*`, which **appends** a new card even
/// when the keyword already exists — producing duplicate cards (cfitsio then reads
/// the *first*, stale value).  Real cubes carry BMAJ/CASAMBM in the copied primary
/// header, so we must use `fits_update_key` (ffuky*) to overwrite in place.
fn update_key_f64(fptr: &mut FitsFile, name: &str, value: f64) -> Result<(), CubeError> {
    let c_name = std::ffi::CString::new(name)
        .map_err(|e| CubeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukyd(
            fptr.as_raw(),
            c_name.as_ptr(),
            value,
            -15, // use the shortest decimal representation that round-trips
            std::ptr::null_mut(),
            &mut status,
        );
    }
    fitsio::errors::check_status(status)?;
    Ok(())
}

/// Update (or create) a string header keyword *in place* (see [`update_key_f64`]).
#[allow(dead_code)]
fn update_key_str(fptr: &mut FitsFile, name: &str, value: &str) -> Result<(), CubeError> {
    let c_name = std::ffi::CString::new(name)
        .map_err(|e| CubeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    let c_value = std::ffi::CString::new(value)
        .map_err(|e| CubeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukys(
            fptr.as_raw(),
            c_name.as_ptr(),
            c_value.as_ptr(),
            std::ptr::null_mut(),
            &mut status,
        );
    }
    fitsio::errors::check_status(status)?;
    Ok(())
}

/// Update (or create) a *logical* (boolean) header keyword in place.
///
/// `CASAMBM` is a FITS logical keyword (`T`/`F`, unquoted).  casacore / CARTA read
/// it with `asBool`, which **throws** if the card is a quoted string (`'T'`) — that
/// makes the cube unreadable.  Always write it as a true logical to match CASA and
/// beamcon output.
fn update_key_logical(fptr: &mut FitsFile, name: &str, value: bool) -> Result<(), CubeError> {
    let c_name = std::ffi::CString::new(name)
        .map_err(|e| CubeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    let mut status = 0;
    unsafe {
        fitsio::sys::ffukyl(
            fptr.as_raw(),
            c_name.as_ptr(),
            value as std::os::raw::c_int,
            std::ptr::null_mut(),
            &mut status,
        );
    }
    fitsio::errors::check_status(status)?;
    Ok(())
}

/// Initialise an output cube by copying the input, then updating the beam headers.
///
/// For `Natural` mode a BEAMS binary-table extension is appended.
/// For `Total` mode only the primary BMAJ/BMIN/BPA keywords are updated.
pub fn init_output_cube(
    input_path: &Path,
    output_path: &Path,
    target_beams: &[Option<Beam>],
    mode: CubeMode,
    meta: &CubeMeta,
) -> Result<(), CubeError> {
    // Initialise the output on disk cheaply: copy only the primary-HDU header
    // from the input (NAXIS/WCS/HISTORY/etc.), not the pixel data.  cfitsio
    // defines the data unit from the copied NAXIS keywords and zero-fills it
    // (sparsely) on close, so we never read the multi-GB input cube — every
    // plane is overwritten by `write_channel` anyway.
    copy_header_only(input_path, output_path)?;

    // Reference channel: CRPIX3 (1-based) → 0-based index clamped to valid range.
    let ref_idx = ((meta.crpix_freq - 1) as usize).min(meta.nfreq.saturating_sub(1));
    let ref_beam = target_beams[ref_idx].unwrap_or_else(|| {
        // Find first valid beam if the reference channel is masked.
        target_beams.iter().find_map(|b| *b).unwrap_or(Beam::zero())
    });

    let tiny = f32::MIN_POSITIVE as f64;

    {
        let path_str = output_path.to_string_lossy().into_owned();
        let mut fptr = FitsFile::edit(&path_str)?;
        fptr.primary_hdu()?; // position at the primary HDU

        // Update primary header PSF in place (the input header — copied verbatim —
        // may already contain these keywords; appending would duplicate them).
        update_key_f64(&mut fptr, "BMAJ", ref_beam.major_deg)?;
        update_key_f64(&mut fptr, "BMIN", ref_beam.minor_deg)?;
        update_key_f64(&mut fptr, "BPA", ref_beam.pa_deg)?;

        // CASAMBM must be a FITS *logical*, not a quoted string, or casacore/CARTA
        // fail to open the cube (they read it with `asBool`).
        update_key_logical(&mut fptr, "CASAMBM", mode == CubeMode::Natural)?;
    }

    if mode == CubeMode::Natural {
        // Build per-channel beam arrays (BMAJ/BMIN in arcsec, BPA in deg).
        let bmaj: Vec<f32> = target_beams
            .iter()
            .map(|b| b.map_or(tiny as f32, |b| b.major_arcsec() as f32))
            .collect();
        let bmin: Vec<f32> = target_beams
            .iter()
            .map(|b| b.map_or(tiny as f32, |b| b.minor_arcsec() as f32))
            .collect();
        let bpa: Vec<f32> = target_beams
            .iter()
            .map(|b| b.map_or(tiny as f32, |b| b.pa_deg as f32))
            .collect();
        let chan: Vec<i32> = (0..meta.nfreq as i32).collect();
        let pol: Vec<i32> = vec![0i32; meta.nfreq];

        let col_bmaj = ColumnDescription::new("BMAJ")
            .with_type(ColumnDataType::Float)
            .create()?;
        let col_bmin = ColumnDescription::new("BMIN")
            .with_type(ColumnDataType::Float)
            .create()?;
        let col_bpa = ColumnDescription::new("BPA")
            .with_type(ColumnDataType::Float)
            .create()?;
        let col_chan = ColumnDescription::new("CHAN")
            .with_type(ColumnDataType::Int)
            .create()?;
        let col_pol = ColumnDescription::new("POL")
            .with_type(ColumnDataType::Int)
            .create()?;

        let path_str = output_path.to_string_lossy().into_owned();
        let mut fptr = FitsFile::edit(&path_str)?;

        let table_hdu =
            fptr.create_table("BEAMS", &[col_bmaj, col_bmin, col_bpa, col_chan, col_pol])?;
        table_hdu.write_col(&mut fptr, "BMAJ", &bmaj)?;
        table_hdu.write_col(&mut fptr, "BMIN", &bmin)?;
        table_hdu.write_col(&mut fptr, "BPA", &bpa)?;
        table_hdu.write_col(&mut fptr, "CHAN", &chan)?;
        table_hdu.write_col(&mut fptr, "POL", &pol)?;

        // Standard BEAMS extension keywords.  `create_table` already wrote EXTNAME,
        // so we do not re-write it (that would append a duplicate card).  Column
        // units (TUNITn) are required by casacore/CARTA to interpret the beam table:
        // BMAJ/BMIN in arcsec, BPA in deg.
        let beam_hdu = fptr.hdu("BEAMS")?;
        beam_hdu.write_key(&mut fptr, "TUNIT1", "arcsec")?;
        beam_hdu.write_key(&mut fptr, "TUNIT2", "arcsec")?;
        beam_hdu.write_key(&mut fptr, "TUNIT3", "deg")?;
        beam_hdu.write_key(&mut fptr, "NCHAN", meta.nfreq as i64)?;
        beam_hdu.write_key(&mut fptr, "NPOL", 1i64)?;
    }

    Ok(())
}

// ── Beamlog ───────────────────────────────────────────────────────────────────

/// Read per-channel beams from a plain-text beamlog.
///
/// Format (produced by RACS-tools or our own writer):
/// ```text
/// # Channel BMAJ[arcsec] BMIN[arcsec] BPA[deg]
/// 0  20.0  10.0  10.0
/// 1  21.0  10.5  10.0
/// ```
/// Column names may include bracketed units (stripped automatically).
/// Returns beams in channel order; returns `Beam::zero()` for masked/zero rows.
pub fn read_beamlog(path: &Path) -> Result<Vec<Beam>, CubeError> {
    let content = std::fs::read_to_string(path)?;
    let mut beams = Vec::new();
    let tiny = f64::from(f32::MIN_POSITIVE);

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
        // fields[0] = channel index (ignored)
        let bmaj_as = parse(fields[1], "BMAJ")?;
        let bmin_as = parse(fields[2], "BMIN")?;
        let bpa_deg = parse(fields[3], "BPA")?;

        let beam = if bmaj_as < tiny || !bmaj_as.is_finite() {
            Beam::zero()
        } else {
            Beam::from_arcsec(bmaj_as, bmin_as.max(tiny), bpa_deg)?
        };
        beams.push(beam);
    }
    Ok(beams)
}

/// Write per-channel beams to a plain-text beamlog.
pub fn write_beamlog(path: &Path, beams: &[Option<Beam>]) -> Result<(), CubeError> {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(out, "# Channel BMAJ[arcsec] BMIN[arcsec] BPA[deg]").unwrap();
    for (i, b) in beams.iter().enumerate() {
        match b {
            Some(b) => writeln!(
                out,
                "{} {} {} {}",
                i,
                b.major_arcsec(),
                b.minor_arcsec(),
                b.pa_deg
            ),
            None => writeln!(out, "{i} nan nan nan"),
        }
        .unwrap();
    }
    std::fs::write(path, out)?;
    Ok(())
}
