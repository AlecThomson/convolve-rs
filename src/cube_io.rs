//! FITS spectral cube reading and writing with per-channel beam support.
//!
//! Supports 3D cubes (NAXIS=3: freq×dec×ra) and 4D cubes (NAXIS=4: stokes×freq×dec×ra).
//! Per-channel beams are read from, in priority order:
//!   1. CASA BEAMS binary-table extension (CASAMBM=T in header)
//!   2. Co-located beamlog text file: `{dir}/beamlog.{stem}.txt`
//!   3. Single BMAJ/BMIN/BPA from the primary header (broadcast to all channels)
use std::path::{Path, PathBuf};

use atfits_rs::{copy_header_only, update_key_f64, update_key_logical};
use fitsio::{
    FitsFile,
    tables::{ColumnDataType, ColumnDescription},
};
use ndarray::Array2;
use thiserror::Error;

use crate::beam::{Beam, BeamError};
use crate::convolve_uv::FftFloat;
use crate::smooth::BrightnessUnit;

// ── Pixel element type ──────────────────────────────────────────────────────────

/// In-memory pixel precision for streaming a cube, derived from FITS `BITPIX`
/// (re-exported from [`atfits_rs`]): `-64` → f64, everything else → f32.
pub use atfits_rs::PixelType;

/// Pixel element types a cube can be streamed in: `f32` or `f64`.
///
/// Implemented only for `f32`/`f64`. Bundles the FITS section read/write (which
/// delegate to the shared monomorphic cfitsio I/O in [`atfits_rs`]) behind the
/// [`FftFloat`] bound, so the convolution pipeline can stay generic over
/// precision and run the FFT at the data's native precision.
pub trait CubeElem: FftFloat {
    fn read_section_vec(
        fptr: &mut FitsFile,
        start: usize,
        end: usize,
    ) -> Result<Vec<Self>, CubeError>;
    fn write_section_vec(
        fptr: &mut FitsFile,
        start: usize,
        end: usize,
        data: &[Self],
    ) -> Result<(), CubeError>;
}

macro_rules! impl_cube_elem {
    ($t:ty) => {
        impl CubeElem for $t {
            fn read_section_vec(
                fptr: &mut FitsFile,
                start: usize,
                end: usize,
            ) -> Result<Vec<Self>, CubeError> {
                Ok(<$t as atfits_rs::CubeElem>::read_section(fptr, start, end)?)
            }
            fn write_section_vec(
                fptr: &mut FitsFile,
                start: usize,
                end: usize,
                data: &[Self],
            ) -> Result<(), CubeError> {
                <$t as atfits_rs::CubeElem>::write_section(fptr, start, end, data)?;
                Ok(())
            }
        }
    };
}
impl_cube_elem!(f32);
impl_cube_elem!(f64);

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

/// Map the shared [`atfits_rs::AtfitsError`] (from the low-level cfitsio helpers)
/// onto the convolve-rs error hierarchy.
impl From<atfits_rs::AtfitsError> for CubeError {
    fn from(e: atfits_rs::AtfitsError) -> Self {
        use atfits_rs::AtfitsError as A;
        match e {
            A::Fits(e) => CubeError::Fits(e),
            A::Io(e) => CubeError::Io(e),
            A::MissingKeyword(s) | A::TargetAxisMissing(s) => CubeError::MissingKeyword(s),
            A::UnsupportedNaxis(n) => CubeError::UnsupportedNaxis(n),
            A::Other(s) => CubeError::Io(std::io::Error::other(s)),
        }
    }
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
    /// Working pixel precision, derived from FITS `BITPIX`.
    pub dtype: PixelType,
}

impl CubeMeta {
    /// Flat element range `[start, end)` of frequency channel `chan` (Stokes 0)
    /// in the primary data unit. For 3D `[nfreq, ny, nx]` and 4D
    /// `[nstokes=1, nfreq, ny, nx]` cubes the offset is `chan * ny * nx`.
    pub fn channel_range(&self, chan: usize) -> (usize, usize) {
        let plane = self.ny * self.nx;
        let start = chan * plane;
        (start, start + plane)
    }

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

    // Pixel precision: convolve in the data's native precision (f32 for -32 and
    // integer cubes, f64 for -64) instead of always upcasting to f64.
    let bitpix: i64 = hdu.read_key(&mut fptr, "BITPIX").unwrap_or(-32);
    let dtype = PixelType::from_bitpix(bitpix);
    if bitpix > 0 {
        // Integer cubes are convolved in f32, but the output header (BITPIX) is
        // copied verbatim, so the floating-point result is rounded back to
        // integers on write. Warn rather than silently lose precision.
        tracing::warn!(
            "{}: integer BITPIX={}; convolution runs in f32 but the output is \
             written at integer precision (fractional flux is rounded). Convert \
             to a floating-point cube (BITPIX=-32) to avoid this.",
            path.display(),
            bitpix
        );
    }

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
            dtype,
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
        dtype,
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
            // Treat tiny/zero beams as masked. `<=` (not `<`) so the `tiny`
            // sentinel that `init_output_cube` writes for a masked channel is
            // detected as masked on read-back — otherwise it round-trips to a
            // bogus ~1e-38° beam.
            if maj_deg <= tiny || !maj_deg.is_finite() {
                None
            } else {
                Beam::new(maj_deg, min_deg.max(tiny), pa).ok()
            }
        })
        .collect();
    Ok(beams)
}

// ── Reading / writing channel planes ─────────────────────────────────────────

/// Read a single frequency channel from a cube into a 2D array (ny × nx), in the
/// requested precision `T`.
///
/// Reads stokes=0 (the first Stokes plane).  For 3D [nfreq, ny, nx] and 4D
/// [nstokes=1, nfreq, ny, nx] cubes the flat offset is identical: `chan * ny * nx`.
pub fn read_channel_as<T: CubeElem>(
    path: &Path,
    chan: usize,
    meta: &CubeMeta,
) -> Result<Array2<T>, CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str)?;

    let (start, end) = meta.channel_range(chan);
    let data = T::read_section_vec(&mut fptr, start, end)?;
    Ok(Array2::from_shape_vec((meta.ny, meta.nx), data)?)
}

/// Read a single frequency channel as `f32` (see [`read_channel_as`]).
pub fn read_channel(path: &Path, chan: usize, meta: &CubeMeta) -> Result<Array2<f32>, CubeError> {
    read_channel_as::<f32>(path, chan, meta)
}

/// Write a single frequency channel plane (precision `T`) back into an existing
/// FITS cube.
///
/// The output cube must have already been initialised by `init_output_cube`.
pub fn write_channel_as<T: CubeElem>(
    path: &Path,
    chan: usize,
    data: &Array2<T>,
    meta: &CubeMeta,
) -> Result<(), CubeError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::edit(&path_str)?;

    let (start, end) = meta.channel_range(chan);
    let flat = data.as_standard_layout();
    let slice = flat.as_slice().expect("standard-layout plane");
    T::write_section_vec(&mut fptr, start, end, slice)?;
    Ok(())
}

/// Write a single `f32` frequency channel plane (see [`write_channel_as`]).
pub fn write_channel(
    path: &Path,
    chan: usize,
    data: &Array2<f32>,
    meta: &CubeMeta,
) -> Result<(), CubeError> {
    write_channel_as::<f32>(path, chan, data, meta)
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
    /// BEAMS table to append on [`CubeWriter::finish`] (Natural mode only).
    ///
    /// Deferred so the extension lands *after* the primary data unit (matching
    /// [`init_output_cube`]'s on-disk layout) and the data unit is written in a
    /// single pass: creating the extension forces cfitsio to flush the primary
    /// data unit, so it must happen after every channel is written, not before.
    /// `None` in Total mode or when opened against an already-initialised cube.
    pending_beams: Option<PendingBeams>,
}

/// Per-channel beam table buffered until [`CubeWriter::finish`].
struct PendingBeams {
    beams: Vec<Option<Beam>>,
    nfreq: usize,
}

impl CubeWriter {
    /// Create a fresh output cube from `input_path`'s primary header and hold the
    /// FITS handle open for streaming channel writes.
    ///
    /// Unlike [`init_output_cube`] (create → close → reopen), the single handle
    /// stays open from creation through every [`CubeWriter::write_channel_as`]
    /// until [`CubeWriter::finish`], so cfitsio writes the data unit exactly once
    /// — avoiding the wasted full zero-fill pass a create-close incurs on the data
    /// unit of a multi-GB cube.  Only data-unit gaps no channel covered are
    /// zero-filled on the final close.
    ///
    /// Primary-header beam keywords (BMAJ/BMIN/BPA/CASAMBM) are written up front;
    /// in `Natural` mode the BEAMS extension is buffered and appended by `finish`.
    pub fn create(
        input_path: &Path,
        output_path: &Path,
        target_beams: &[Option<Beam>],
        mode: CubeMode,
        meta: &CubeMeta,
    ) -> Result<Self, CubeError> {
        // Copy the primary header from the input and keep the handle open (no
        // data written yet). cfitsio defines the data unit from the copied NAXIS
        // keywords; it is written once, when this handle is finally dropped.
        let mut fptr = atfits_rs::copy_header_only_open(input_path, output_path)?;

        let ref_beam = ref_beam_for(target_beams, meta);
        write_primary_beam_keys(&mut fptr, ref_beam, mode == CubeMode::Natural)?;

        // Reposition at the primary HDU so subsequent channel writes target the
        // primary data unit (key updates above already sit there, but be explicit).
        fptr.primary_hdu()?;

        let pending_beams = (mode == CubeMode::Natural).then(|| PendingBeams {
            beams: target_beams.to_vec(),
            nfreq: meta.nfreq,
        });
        Ok(Self {
            fptr,
            pending_beams,
        })
    }

    /// Open an already-initialised output cube (see [`init_output_cube`]) for
    /// sequential channel writes.
    pub fn open(path: &Path) -> Result<Self, CubeError> {
        let fptr = FitsFile::edit(path.to_string_lossy().into_owned())?;
        Ok(Self {
            fptr,
            pending_beams: None,
        })
    }

    /// Write one frequency channel plane (precision `T`) into the open cube.
    pub fn write_channel_as<T: CubeElem>(
        &mut self,
        chan: usize,
        data: &Array2<T>,
        meta: &CubeMeta,
    ) -> Result<(), CubeError> {
        let (start, end) = meta.channel_range(chan);
        let flat = data.as_standard_layout();
        let slice = flat.as_slice().expect("standard-layout plane");
        T::write_section_vec(&mut self.fptr, start, end, slice)?;
        Ok(())
    }

    /// Write one `f32` frequency channel plane (see [`CubeWriter::write_channel_as`]).
    pub fn write_channel(
        &mut self,
        chan: usize,
        data: &Array2<f32>,
        meta: &CubeMeta,
    ) -> Result<(), CubeError> {
        self.write_channel_as::<f32>(chan, data, meta)
    }

    /// Finish the cube: append the buffered BEAMS extension (Natural mode) and
    /// close the FITS file exactly once.
    ///
    /// Must be called after the final channel write. The single close zero-fills
    /// only the data-unit gaps no channel covered; creating the BEAMS extension
    /// here (not at `create`) keeps the primary data unit written in one pass.
    pub fn finish(mut self) -> Result<(), CubeError> {
        if let Some(p) = self.pending_beams.take() {
            write_beams_table(&mut self.fptr, &p.beams, p.nfreq)?;
        }
        // `self.fptr` drops at end of scope → cfitsio closes and flushes once.
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

// The header-only copy (`copy_header_only`) and update-in-place keyword editors
// (`update_key_f64`, `update_key_logical`) now live in `atfits_rs` and are
// imported at the top of this module. For the streaming write path that keeps
// the handle open so the data unit is written exactly once, see
// [`atfits_rs::copy_header_only_open`].

/// Reference beam for the primary header: the beam at CRPIX3 (clamped to range),
/// falling back to the first valid beam, then to a zero beam.
fn ref_beam_for(target_beams: &[Option<Beam>], meta: &CubeMeta) -> Beam {
    let ref_idx = ((meta.crpix_freq - 1) as usize).min(meta.nfreq.saturating_sub(1));
    target_beams[ref_idx].unwrap_or_else(|| {
        // Find first valid beam if the reference channel is masked.
        target_beams.iter().find_map(|b| *b).unwrap_or(Beam::zero())
    })
}

/// Write the primary-header PSF keywords (BMAJ/BMIN/BPA + CASAMBM) in place.
///
/// `fptr` must be positioned at the primary HDU. Uses `update_key_*` (ffuky*),
/// which overwrites in place — the input header is copied verbatim and may
/// already carry these cards, so appending would duplicate them. CASAMBM is
/// written as a FITS *logical* (not a quoted string), or casacore/CARTA fail to
/// open the cube (they read it with `asBool`).
fn write_primary_beam_keys(
    fptr: &mut FitsFile,
    ref_beam: Beam,
    natural: bool,
) -> Result<(), CubeError> {
    fptr.primary_hdu()?; // position at the primary HDU
    update_key_f64(fptr, "BMAJ", ref_beam.major_deg)?;
    update_key_f64(fptr, "BMIN", ref_beam.minor_deg)?;
    update_key_f64(fptr, "BPA", ref_beam.pa_deg)?;
    update_key_logical(fptr, "CASAMBM", natural)?;
    Ok(())
}

/// Append the CASA BEAMS binary-table extension (per-channel beams) to `fptr`.
///
/// Creating this extension forces cfitsio to flush the primary data unit, so on
/// the streaming write path it must be called *after* every channel is written
/// (see [`CubeWriter::finish`]) to keep the data unit written in a single pass.
fn write_beams_table(
    fptr: &mut FitsFile,
    target_beams: &[Option<Beam>],
    nfreq: usize,
) -> Result<(), CubeError> {
    let tiny = f32::MIN_POSITIVE as f64;

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
    let chan: Vec<i32> = (0..nfreq as i32).collect();
    let pol: Vec<i32> = vec![0i32; nfreq];

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

    let table_hdu =
        fptr.create_table("BEAMS", &[col_bmaj, col_bmin, col_bpa, col_chan, col_pol])?;
    table_hdu.write_col(fptr, "BMAJ", &bmaj)?;
    table_hdu.write_col(fptr, "BMIN", &bmin)?;
    table_hdu.write_col(fptr, "BPA", &bpa)?;
    table_hdu.write_col(fptr, "CHAN", &chan)?;
    table_hdu.write_col(fptr, "POL", &pol)?;

    // Standard BEAMS extension keywords.  `create_table` already wrote EXTNAME,
    // so we do not re-write it (that would append a duplicate card).  Column
    // units (TUNITn) are required by casacore/CARTA to interpret the beam table:
    // BMAJ/BMIN in arcsec, BPA in deg.
    let beam_hdu = fptr.hdu("BEAMS")?;
    beam_hdu.write_key(fptr, "TUNIT1", "arcsec")?;
    beam_hdu.write_key(fptr, "TUNIT2", "arcsec")?;
    beam_hdu.write_key(fptr, "TUNIT3", "deg")?;
    beam_hdu.write_key(fptr, "NCHAN", nfreq as i64)?;
    beam_hdu.write_key(fptr, "NPOL", 1i64)?;
    Ok(())
}

/// Initialise an output cube by copying the input header, then updating the beam
/// headers, closing the file once. The data unit is zero-filled by the close.
///
/// For `Natural` mode a BEAMS binary-table extension is appended.
/// For `Total` mode only the primary BMAJ/BMIN/BPA keywords are updated.
///
/// The streaming cube write path uses [`CubeWriter::create`] instead, which keeps
/// the handle open so the data unit is written a single time. This function
/// remains for callers that initialise then write planes through a separate
/// handle (e.g. [`write_channel`]).
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

    let ref_beam = ref_beam_for(target_beams, meta);

    {
        let path_str = output_path.to_string_lossy().into_owned();
        let mut fptr = FitsFile::edit(&path_str)?;
        write_primary_beam_keys(&mut fptr, ref_beam, mode == CubeMode::Natural)?;

        if mode == CubeMode::Natural {
            write_beams_table(&mut fptr, target_beams, meta.nfreq)?;
        }
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

        let beam = if bmaj_as <= tiny || !bmaj_as.is_finite() {
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
