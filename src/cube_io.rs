/// FITS spectral cube reading and writing with per-channel beam support.
///
/// Supports 3D cubes (NAXIS=3: freq×dec×ra) and 4D cubes (NAXIS=4: stokes×freq×dec×ra).
/// Per-channel beams are read from, in priority order:
///   1. CASA BEAMS binary-table extension (CASAMBM=T in header)
///   2. Co-located beamlog text file: `{dir}/beamlog.{stem}.txt`
///   3. Single BMAJ/BMIN/BPA from the primary header (broadcast to all channels)
use std::path::{Path, PathBuf};

use fitsio::{FitsFile, tables::{ColumnDataType, ColumnDescription}};
use ndarray::Array2;
use thiserror::Error;

use crate::beam::{Beam, BeamError};

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
}

impl CubeMeta {
    /// Beamlog path co-located with the FITS file.
    pub fn beamlog_path(&self) -> PathBuf {
        let dir  = self.path.parent().unwrap_or(Path::new("."));
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

    // Check for CASAMBM
    let casambm: String = hdu.read_key(&mut fptr, "CASAMBM").unwrap_or_default();
    drop(fptr); // close for next reads

    let beams: Vec<Option<Beam>> = if casambm.trim() == "T" || casambm.trim() == "TRUE" {
        read_casambm_beams(path, nfreq)?
    } else {
        let beamlog = CubeMeta {
            path: path.to_path_buf(),
            nx, ny, nfreq, nstokes,
            dx_deg, dy_deg,
            crpix_freq,
            beams: vec![],
            is_4d,
        }.beamlog_path();

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
            let mut fptr2 = FitsFile::open(&path.to_string_lossy().into_owned())?;
            let hdu2 = fptr2.primary_hdu()?;
            let bmaj: f64 = hdu2.read_key(&mut fptr2, "BMAJ")
                .map_err(|_| CubeError::NoBeans)?;
            let bmin: f64 = hdu2.read_key(&mut fptr2, "BMIN").unwrap_or(bmaj);
            let bpa:  f64 = hdu2.read_key(&mut fptr2, "BPA").unwrap_or(0.0);
            let b = Beam::new(bmaj, bmin, bpa)?;
            vec![Some(b); nfreq]
        }
    };

    Ok(CubeMeta {
        path: path.to_path_buf(),
        nx, ny, nfreq, nstokes,
        dx_deg, dy_deg,
        crpix_freq,
        beams,
        is_4d,
    })
}

/// Read per-channel beams from the CASA BEAMS binary-table extension.
///
/// Columns: BMAJ [arcsec], BMIN [arcsec], BPA [deg], CHAN [int], POL [int].
fn read_casambm_beams(path: &Path, nfreq: usize) -> Result<Vec<Option<Beam>>, CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str)?;
    let hdu = fptr.hdu("BEAMS").map_err(|_| CubeError::MissingKeyword("BEAMS extension".into()))?;

    let bmaj: Vec<f32> = hdu.read_col(&mut fptr, "BMAJ")?;
    let bmin: Vec<f32> = hdu.read_col(&mut fptr, "BMIN")?;
    let bpa:  Vec<f32> = hdu.read_col(&mut fptr, "BPA")?;

    if bmaj.len() != nfreq {
        return Err(CubeError::BeamCountMismatch { expected: nfreq, got: bmaj.len() });
    }

    let tiny = f32::MIN_POSITIVE as f64;
    let beams = bmaj.iter().zip(bmin.iter()).zip(bpa.iter())
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
    let end   = start + plane;

    let data: Vec<f32> = hdu.read_section(&mut fptr, start, end)?;
    Ok(Array2::from_shape_vec((meta.ny, meta.nx), data)?)
}

/// Write a single frequency channel plane back into an existing FITS cube.
///
/// The output cube must have already been initialised by `init_output_cube`.
pub fn write_channel(path: &Path, chan: usize, data: &Array2<f32>, meta: &CubeMeta) -> Result<(), CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::edit(&path_str)?;
    let hdu = fptr.primary_hdu()?;

    let plane = meta.ny * meta.nx;
    let start = chan * plane;
    let end   = start + plane;

    let flat: Vec<f32> = data.iter().copied().collect();
    hdu.write_section(&mut fptr, start, end, &flat)?;
    Ok(())
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
    // Copy file, preserving all existing data and header.
    std::fs::copy(input_path, output_path)?;

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
        let hdu = fptr.primary_hdu()?;

        // Update primary header PSF.
        hdu.write_key(&mut fptr, "BMAJ", ref_beam.major_deg)?;
        hdu.write_key(&mut fptr, "BMIN", ref_beam.minor_deg)?;
        hdu.write_key(&mut fptr, "BPA",  ref_beam.pa_deg)?;

        if mode == CubeMode::Natural {
            hdu.write_key(&mut fptr, "CASAMBM", "T")?;
        } else {
            let _ = hdu.write_key(&mut fptr, "CASAMBM", "F");
        }
    }

    if mode == CubeMode::Natural {
        // Build per-channel beam arrays (BMAJ/BMIN in arcsec, BPA in deg).
        let bmaj: Vec<f32> = target_beams.iter()
            .map(|b| b.map_or(tiny as f32, |b| b.major_arcsec() as f32))
            .collect();
        let bmin: Vec<f32> = target_beams.iter()
            .map(|b| b.map_or(tiny as f32, |b| b.minor_arcsec() as f32))
            .collect();
        let bpa: Vec<f32> = target_beams.iter()
            .map(|b| b.map_or(tiny as f32, |b| b.pa_deg as f32))
            .collect();
        let chan: Vec<i32> = (0..meta.nfreq as i32).collect();
        let pol: Vec<i32> = vec![0i32; meta.nfreq];

        let col_bmaj = ColumnDescription::new("BMAJ").with_type(ColumnDataType::Float).create()?;
        let col_bmin = ColumnDescription::new("BMIN").with_type(ColumnDataType::Float).create()?;
        let col_bpa  = ColumnDescription::new("BPA") .with_type(ColumnDataType::Float).create()?;
        let col_chan = ColumnDescription::new("CHAN") .with_type(ColumnDataType::Int)  .create()?;
        let col_pol  = ColumnDescription::new("POL") .with_type(ColumnDataType::Int)  .create()?;

        let path_str = output_path.to_string_lossy().into_owned();
        let mut fptr = FitsFile::edit(&path_str)?;

        let table_hdu = fptr.create_table(
            "BEAMS",
            &[col_bmaj, col_bmin, col_bpa, col_chan, col_pol],
        )?;
        table_hdu.write_col(&mut fptr, "BMAJ", &bmaj)?;
        table_hdu.write_col(&mut fptr, "BMIN", &bmin)?;
        table_hdu.write_col(&mut fptr, "BPA",  &bpa)?;
        table_hdu.write_col(&mut fptr, "CHAN", &chan)?;
        table_hdu.write_col(&mut fptr, "POL",  &pol)?;

        // Set standard BEAMS extension keywords.
        let beam_hdu = fptr.hdu("BEAMS")?;
        beam_hdu.write_key(&mut fptr, "EXTNAME", "BEAMS")?;
        beam_hdu.write_key(&mut fptr, "NCHAN", meta.nfreq as i64)?;
        beam_hdu.write_key(&mut fptr, "NPOL",  1i64)?;
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
            Some(b) => writeln!(out, "{} {} {} {}", i, b.major_arcsec(), b.minor_arcsec(), b.pa_deg),
            None    => writeln!(out, "{} nan nan nan", i),
        }.unwrap();
    }
    std::fs::write(path, out)?;
    Ok(())
}
