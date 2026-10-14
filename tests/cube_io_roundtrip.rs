//! Cube I/O round trips and end-to-end `convolvers 3d` runs.

use convolve_rs::beam::Beam;
use convolve_rs::cube_io::{self, CubeElem, CubeMode, CubeReader, CubeWriter};
use fitsio::FitsFile;
use fitsio::tables::{ColumnDataType, ColumnDescription};
use ndarray::Array2;

const NX: usize = 8;
const NY: usize = 8;
const NFREQ: usize = 3;

/// Build a 3D image cube with a CASA `BEAMS` binary-table extension.
fn make_cube(path: &std::path::Path) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type: fitsio::images::ImageType::Float,
            dimensions: &[NFREQ, NY, NX],
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let data = vec![1.0f32; NX * NY * NFREQ];
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "CDELT1", -0.001f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 0.001f64).unwrap();
    hdu.write_key(&mut f, "CRPIX3", 1i64).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "CASAMBM", "T").unwrap();

    let cols = vec![
        ColumnDescription::new("BMAJ")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BMIN")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BPA")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("CHAN")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
        ColumnDescription::new("POL")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
    ];
    let t = f.create_table("BEAMS", &cols).unwrap();
    t.write_col(&mut f, "BMAJ", &[20.0f32; NFREQ]).unwrap();
    t.write_col(&mut f, "BMIN", &[15.0f32; NFREQ]).unwrap();
    t.write_col(&mut f, "BPA", &[0.0f32; NFREQ]).unwrap();
    t.write_col(&mut f, "CHAN", &(0..NFREQ as i32).collect::<Vec<_>>())
        .unwrap();
    t.write_col(&mut f, "POL", &[0i32; NFREQ]).unwrap();
}

/// Write `planes` (if any) to a new cube made from `input`'s header.
fn write_cube<T: CubeElem>(
    input: &std::path::Path,
    out: &std::path::Path,
    target: &[Option<Beam>],
    mode: CubeMode,
    meta: &cube_io::CubeMeta,
    planes: &[Array2<T>],
) -> Result<(), cube_io::CubeError> {
    let mut w = CubeWriter::create(input, out, target, mode, meta)?;
    for (c, p) in planes.iter().enumerate() {
        w.write_channel(c, p, meta)?;
    }
    w.finish()
}

fn read_channel<T: CubeElem>(
    path: &std::path::Path,
    c: usize,
    meta: &cube_io::CubeMeta,
) -> Array2<T> {
    CubeReader::open(path)
        .unwrap()
        .read_channel(c, meta)
        .unwrap()
}

fn workdir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("convolve_rs_test_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Total mode: a channel can be written back and read with the right value.
#[test]
fn total_mode_write_channel_roundtrip() {
    let dir = workdir("total");
    let path = dir.join("in.fits");
    make_cube(&path);

    let meta = cube_io::read_cube_meta(&path).unwrap();
    assert_eq!(meta.nfreq, NFREQ);
    assert_eq!((meta.ny, meta.nx), (NY, NX));

    let out = dir.join("out.fits");
    let target = vec![Some(Beam::from_arcsec(25.0, 20.0, 0.0).unwrap()); NFREQ];
    let planes: Vec<_> = (0..NFREQ)
        .map(|c| Array2::from_elem((NY, NX), c as f32 + 0.5))
        .collect();
    write_cube(&path, &out, &target, CubeMode::Total, &meta, &planes).unwrap();

    // The output primary HDU must carry the copied 3D image structure.
    {
        let mut f = FitsFile::edit(out.to_string_lossy().to_string()).unwrap();
        let hdu = f.primary_hdu().unwrap();
        let naxis: i64 = hdu.read_key(&mut f, "NAXIS").unwrap();
        assert_eq!(naxis, 3, "output primary HDU lost its NAXIS=3 structure");
    }

    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    for c in 0..NFREQ {
        let plane = read_channel::<f32>(&out, c, &out_meta);
        assert!(plane.iter().all(|&v| (v - (c as f32 + 0.5)).abs() < 1e-6));
    }
}

/// A masked (None) target beam must round-trip through the BEAMS table as None,
/// not as a bogus ~1e-38° beam (the `tiny` sentinel must be detected on read).
#[test]
fn masked_target_beam_round_trips_as_none() {
    let dir = workdir("masked_roundtrip");
    let path = dir.join("in.fits");
    make_cube(&path);

    let meta = cube_io::read_cube_meta(&path).unwrap();
    let out = dir.join("out.fits");
    // Channel 1 is masked; channels 0 and 2 are valid.
    let target: Vec<Option<Beam>> = (0..NFREQ)
        .map(|c| (c != 1).then(|| Beam::from_arcsec(25.0, 20.0, 0.0).unwrap()))
        .collect();
    write_cube::<f32>(&path, &out, &target, CubeMode::Natural, &meta, &[]).unwrap();

    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    assert!(
        out_meta.beams[1].is_none(),
        "masked channel must read back as None, got {:?}",
        out_meta.beams[1]
    );
    assert!(out_meta.beams[0].is_some());
    assert!(out_meta.beams[2].is_some());
}

/// An output path that resolves to the input must be rejected, not clobber the
/// input (which `copy_header_only` would otherwise delete before recreating).
#[test]
fn output_equal_to_input_is_rejected() {
    let dir = workdir("out_eq_in");
    let path = dir.join("in.fits");
    make_cube(&path);

    let meta = cube_io::read_cube_meta(&path).unwrap();
    let target = vec![Some(Beam::from_arcsec(25.0, 20.0, 0.0).unwrap()); NFREQ];
    let res = write_cube::<f32>(&path, &path, &target, CubeMode::Total, &meta, &[]);
    assert!(
        res.is_err(),
        "must refuse output that resolves to the input"
    );
    // The input cube must survive intact.
    assert!(
        cube_io::read_cube_meta(&path).is_ok(),
        "input cube was destroyed"
    );
}

/// Natural mode: a per-channel BEAMS table is written and round-trips.
#[test]
fn natural_mode_write_channel_and_beams() {
    let dir = workdir("natural");
    let path = dir.join("in.fits");
    make_cube(&path);

    let meta = cube_io::read_cube_meta(&path).unwrap();
    let out = dir.join("out.fits");
    let target: Vec<Option<Beam>> = (0..NFREQ)
        .map(|c| Some(Beam::from_arcsec(25.0 + c as f64, 20.0, 0.0).unwrap()))
        .collect();
    let planes: Vec<_> = (0..NFREQ)
        .map(|c| Array2::from_elem((NY, NX), c as f32 + 0.5))
        .collect();
    write_cube(&path, &out, &target, CubeMode::Natural, &meta, &planes).unwrap();

    // BEAMS extension must round-trip the per-channel target beams.
    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    for c in 0..NFREQ {
        let b = out_meta.beams[c].expect("channel beam present");
        assert!((b.major_arcsec() - (25.0 + c as f64)).abs() < 1e-3);
    }
}

/// CASAMBM must be a FITS *logical* (readable as `bool`), not a quoted string:
/// casacore/CARTA read it with `asBool` and fail to open the cube otherwise.
/// Also pins the BEAMS-table column units that casacore needs.
#[test]
fn casambm_is_logical_not_string() {
    let dir = workdir("casambm");
    let path = dir.join("in.fits");
    make_cube(&path);
    let meta = cube_io::read_cube_meta(&path).unwrap();
    let target = vec![Some(Beam::from_arcsec(25.0, 20.0, 0.0).unwrap()); NFREQ];

    // Total mode → CASAMBM = F (logical), single HDU.
    let total = dir.join("total.fits");
    write_cube::<f32>(&path, &total, &target, CubeMode::Total, &meta, &[]).unwrap();
    {
        let mut f = FitsFile::edit(total.to_string_lossy().to_string()).unwrap();
        let hdu = f.primary_hdu().unwrap();
        let v: bool = hdu
            .read_key(&mut f, "CASAMBM")
            .expect("CASAMBM must be readable as a logical");
        assert!(!v, "total mode CASAMBM should be F");
    }

    // Natural mode → CASAMBM = T (logical) + BEAMS table with column units.
    let nat = dir.join("natural.fits");
    write_cube::<f32>(&path, &nat, &target, CubeMode::Natural, &meta, &[]).unwrap();
    {
        let mut f = FitsFile::edit(nat.to_string_lossy().to_string()).unwrap();
        let hdu = f.primary_hdu().unwrap();
        let v: bool = hdu
            .read_key(&mut f, "CASAMBM")
            .expect("CASAMBM must be readable as a logical");
        assert!(v, "natural mode CASAMBM should be T");

        let beams = f.hdu("BEAMS").unwrap();
        let u1: String = beams.read_key(&mut f, "TUNIT1").unwrap();
        let u2: String = beams.read_key(&mut f, "TUNIT2").unwrap();
        let u3: String = beams.read_key(&mut f, "TUNIT3").unwrap();
        assert_eq!(
            (u1.trim(), u2.trim(), u3.trim()),
            ("arcsec", "arcsec", "deg")
        );
    }
}

// ── End-to-end CLI smoke tests ─────────────────────────────────────────────────
//
// Drive the actual `convolvers` binary on a synthetic cube, the same path that
// hit the cfitsio 302 crash.  These exercise argument parsing, the full
// init → smooth → write pipeline, and beamlog output.

/// Build a 3D cube whose per-channel beams *vary*, so a common-beam smooth has a
/// non-trivial (non-zero) convolving kernel.
fn make_varied_cube(path: &std::path::Path) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type: fitsio::images::ImageType::Float,
            dimensions: &[NFREQ, NY, NX],
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    // A single bright central pixel per channel; well-defined to convolve.
    let mut data = vec![0.0f32; NX * NY * NFREQ];
    for c in 0..NFREQ {
        data[c * NX * NY + (NY / 2) * NX + NX / 2] = 1.0;
    }
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "CDELT1", -0.0005f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 0.0005f64).unwrap();
    hdu.write_key(&mut f, "CRPIX3", 1i64).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "CASAMBM", "T").unwrap();

    let bmaj: Vec<f32> = (0..NFREQ).map(|c| 16.0 + c as f32).collect();
    let bmin: Vec<f32> = (0..NFREQ).map(|c| 12.0 + c as f32).collect();
    let bpa: Vec<f32> = (0..NFREQ).map(|c| c as f32 * 5.0).collect();
    let cols = vec![
        ColumnDescription::new("BMAJ")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BMIN")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BPA")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("CHAN")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
        ColumnDescription::new("POL")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
    ];
    let t = f.create_table("BEAMS", &cols).unwrap();
    t.write_col(&mut f, "BMAJ", &bmaj).unwrap();
    t.write_col(&mut f, "BMIN", &bmin).unwrap();
    t.write_col(&mut f, "BPA", &bpa).unwrap();
    t.write_col(&mut f, "CHAN", &(0..NFREQ as i32).collect::<Vec<_>>())
        .unwrap();
    t.write_col(&mut f, "POL", &[0i32; NFREQ]).unwrap();
}

/// Run the `convolvers` binary, returning (success, combined stdout+stderr).
fn run_cli(args: &[&str]) -> (bool, String) {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_convolvers"))
        .args(args)
        .output()
        .expect("failed to spawn convolvers binary");
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), log)
}

#[test]
fn cli_total_mode_smooths_cube() {
    let dir = workdir("cli_total");
    let path = dir.join("in.fits");
    make_varied_cube(&path);

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "total"]);
    assert!(ok, "binary failed:\n{log}");

    // Output cube must exist, be readable, and carry a single common beam ≥ inputs.
    let out = dir.join("in.sm.fits");
    assert!(out.exists(), "output cube not written:\n{log}");
    let meta = cube_io::read_cube_meta(&out).unwrap();
    assert_eq!(meta.nfreq, NFREQ);
    let common = meta.beams[0].expect("common beam");
    assert!(common.major_arcsec() >= 16.0 + (NFREQ as f64 - 1.0));

    // Every channel the streaming writer produced must hold real, finite pixels
    // (a point source convolved to a Gaussian), not zeros or NaNs.
    for c in 0..NFREQ {
        let plane = read_channel::<f32>(&out, c, &meta);
        assert!(
            plane.iter().all(|v| v.is_finite()),
            "channel {c} has non-finite pixels"
        );
        let peak = plane.iter().cloned().fold(f32::MIN, f32::max);
        assert!(peak > 0.0, "channel {c} is empty (peak {peak})");
    }

    // Beamlog should be written alongside the output.
    assert!(
        dir.join("beamlog.in.sm.txt").exists(),
        "beamlog missing:\n{log}"
    );
}

#[test]
fn cli_natural_mode_smooths_cube() {
    let dir = workdir("cli_natural");
    let path = dir.join("in.fits");
    make_varied_cube(&path);

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "natural"]);
    assert!(ok, "binary failed:\n{log}");

    let out = dir.join("in.sm.fits");
    assert!(out.exists(), "output cube not written:\n{log}");
    let meta = cube_io::read_cube_meta(&out).unwrap();
    // Natural mode keeps the CASA BEAMS extension with one beam per channel.
    assert_eq!(meta.beams.iter().filter(|b| b.is_some()).count(), NFREQ);
}

/// Build a varied-beam cube with `BITPIX = -64` (f64 pixels) to exercise the
/// native double-precision streaming path.
fn make_varied_cube_f64(path: &std::path::Path) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type: fitsio::images::ImageType::Double,
            dimensions: &[NFREQ, NY, NX],
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let mut data = vec![0.0f64; NX * NY * NFREQ];
    for c in 0..NFREQ {
        data[c * NX * NY + (NY / 2) * NX + NX / 2] = 1.0;
    }
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "CDELT1", -0.0005f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 0.0005f64).unwrap();
    hdu.write_key(&mut f, "CRPIX3", 1i64).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "CASAMBM", "T").unwrap();

    let bmaj: Vec<f32> = (0..NFREQ).map(|c| 16.0 + c as f32).collect();
    let bmin: Vec<f32> = (0..NFREQ).map(|c| 12.0 + c as f32).collect();
    let bpa: Vec<f32> = (0..NFREQ).map(|c| c as f32 * 5.0).collect();
    let cols = vec![
        ColumnDescription::new("BMAJ")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BMIN")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BPA")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("CHAN")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
        ColumnDescription::new("POL")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
    ];
    let t = f.create_table("BEAMS", &cols).unwrap();
    t.write_col(&mut f, "BMAJ", &bmaj).unwrap();
    t.write_col(&mut f, "BMIN", &bmin).unwrap();
    t.write_col(&mut f, "BPA", &bpa).unwrap();
    t.write_col(&mut f, "CHAN", &(0..NFREQ as i32).collect::<Vec<_>>())
        .unwrap();
    t.write_col(&mut f, "POL", &[0i32; NFREQ]).unwrap();
}

/// A `BITPIX = -64` cube is detected as f64 and streams channels in native
/// double precision: a value f32 cannot represent must round-trip exactly.
#[test]
fn f64_cube_detects_dtype_and_roundtrips() {
    use cube_io::PixelType;

    let dir = workdir("f64_roundtrip");
    let path = dir.join("in.fits");
    make_varied_cube_f64(&path);

    let meta = cube_io::read_cube_meta(&path).unwrap();
    assert_eq!(meta.dtype, PixelType::F64, "BITPIX -64 should map to F64");

    let out = dir.join("out.fits");
    let target = vec![Some(Beam::from_arcsec(25.0, 20.0, 0.0).unwrap()); NFREQ];
    // 1 + 1e-10 is not representable in f32, so an f32 round-trip would lose it.
    let val = 1.0 + 1e-10_f64;
    let planes = vec![Array2::from_elem((NY, NX), val); NFREQ];
    write_cube(&path, &out, &target, CubeMode::Total, &meta, &planes).unwrap();

    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    assert_eq!(out_meta.dtype, PixelType::F64, "output kept BITPIX -64");
    for c in 0..NFREQ {
        let plane = read_channel::<f64>(&out, c, &out_meta);
        assert!(
            plane.iter().all(|&v| v == val),
            "f64 channel {c} did not round-trip exactly"
        );
    }
}

/// End-to-end: the CLI smooths an f64 cube through `process_cube::<f64>` and the
/// output cube stays double precision with finite, non-empty channels.
#[test]
fn cli_total_mode_smooths_f64_cube() {
    use cube_io::PixelType;

    let dir = workdir("cli_f64");
    let path = dir.join("in.fits");
    make_varied_cube_f64(&path);

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "total"]);
    assert!(ok, "binary failed:\n{log}");

    let out = dir.join("in.sm.fits");
    assert!(out.exists(), "output cube not written:\n{log}");
    let meta = cube_io::read_cube_meta(&out).unwrap();
    assert_eq!(
        meta.dtype,
        PixelType::F64,
        "f64 output must stay BITPIX -64"
    );
    assert_eq!(meta.nfreq, NFREQ);

    for c in 0..NFREQ {
        let plane = read_channel::<f64>(&out, c, &meta);
        assert!(
            plane.iter().all(|v| v.is_finite()),
            "channel {c} has non-finite pixels"
        );
        let peak = plane.iter().cloned().fold(f64::MIN, f64::max);
        assert!(peak > 0.0, "channel {c} is empty (peak {peak})");
    }
}

/// Read the primary BMAJ/BMIN/BPA (deg) and CASAMBM (logical) keywords.
fn read_primary_beam_keys(path: &std::path::Path) -> (f64, f64, f64, bool) {
    let mut f = FitsFile::edit(path.to_string_lossy().to_string()).unwrap();
    let hdu = f.primary_hdu().unwrap();
    (
        hdu.read_key(&mut f, "BMAJ").unwrap(),
        hdu.read_key(&mut f, "BMIN").unwrap(),
        hdu.read_key(&mut f, "BPA").unwrap(),
        hdu.read_key(&mut f, "CASAMBM").unwrap(),
    )
}

/// Build a 3D point-source cube with the given per-channel BEAMS table.
fn make_cube_with_beams(path: &std::path::Path, bmaj: &[f32], bmin: &[f32], bpa: &[f32]) {
    make_cube_typed(path, fitsio::images::ImageType::Float, bmaj, bmin, bpa);
}

fn make_cube_typed(
    path: &std::path::Path,
    data_type: fitsio::images::ImageType,
    bmaj: &[f32],
    bmin: &[f32],
    bpa: &[f32],
) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type,
            dimensions: &[NFREQ, NY, NX],
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let mut data = vec![0.0f32; NX * NY * NFREQ];
    for c in 0..NFREQ {
        data[c * NX * NY + (NY / 2) * NX + NX / 2] = 1.0;
    }
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "CDELT1", -0.0005f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 0.0005f64).unwrap();
    hdu.write_key(&mut f, "CRPIX3", 1i64).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "CASAMBM", "T").unwrap();

    let cols = vec![
        ColumnDescription::new("BMAJ")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BMIN")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("BPA")
            .with_type(ColumnDataType::Float)
            .create()
            .unwrap(),
        ColumnDescription::new("CHAN")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
        ColumnDescription::new("POL")
            .with_type(ColumnDataType::Int)
            .create()
            .unwrap(),
    ];
    let t = f.create_table("BEAMS", &cols).unwrap();
    t.write_col(&mut f, "BMAJ", bmaj).unwrap();
    t.write_col(&mut f, "BMIN", bmin).unwrap();
    t.write_col(&mut f, "BPA", bpa).unwrap();
    t.write_col(&mut f, "CHAN", &(0..NFREQ as i32).collect::<Vec<_>>())
        .unwrap();
    t.write_col(&mut f, "POL", &[0i32; NFREQ]).unwrap();
}

/// Like `make_varied_cube` but with channel 1's beam masked (BMAJ = 0).
fn make_cube_masked_chan1(path: &std::path::Path) {
    make_cube_with_beams(
        path,
        &[16.0, 0.0, 18.0],
        &[12.0, 0.0, 14.0],
        &[0.0, 0.0, 10.0],
    );
}

/// A masked-beam channel must be written as NaN (explicit no-data), while the
/// valid channels stay finite, never zero-filled or corrupted to infinity by a
/// divide-by-zero in the analytic UV filter.
#[test]
fn cli_masked_channel_is_blanked_not_infinite() {
    let dir = workdir("cli_masked");
    let path = dir.join("in.fits");
    make_cube_masked_chan1(&path);

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "total"]);
    assert!(ok, "binary failed:\n{log}");

    let out = dir.join("in.sm.fits");
    let meta = cube_io::read_cube_meta(&out).unwrap();

    // Masked channel 1: entirely NaN (blanked), with no infinities.
    let masked = read_channel::<f32>(&out, 1, &meta);
    assert!(
        masked.iter().all(|v| v.is_nan()),
        "masked channel should be all-NaN, got finite/zero pixels"
    );

    // Valid channels: finite, non-empty, and crucially not infinite.
    for c in [0, 2] {
        let plane = read_channel::<f32>(&out, c, &meta);
        assert!(
            plane.iter().all(|v| v.is_finite()),
            "channel {c} has non-finite pixels"
        );
        assert!(
            plane.iter().cloned().fold(f32::MIN, f32::max) > 0.0,
            "channel {c} is empty"
        );
    }
}

/// Build a 4D cube with NAXIS4 = 2 (two Stokes planes) and a single header beam.
fn make_4d_cube(path: &std::path::Path) {
    const NSTOKES: usize = 2;
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type: fitsio::images::ImageType::Float,
            dimensions: &[NSTOKES, NFREQ, NY, NX],
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    hdu.write_image(&mut f, &vec![1.0f32; NSTOKES * NFREQ * NY * NX])
        .unwrap();
    hdu.write_key(&mut f, "CDELT1", -0.0005f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 0.0005f64).unwrap();
    hdu.write_key(&mut f, "CRPIX3", 1i64).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "BMAJ", 0.005f64).unwrap();
    hdu.write_key(&mut f, "BMIN", 0.004f64).unwrap();
    hdu.write_key(&mut f, "BPA", 0.0f64).unwrap();
}

/// A multi-Stokes (NAXIS4 > 1) cube must be rejected rather than silently
/// convolving only Stokes 0 and zero-filling the rest.
#[test]
fn cli_multi_stokes_cube_is_rejected() {
    let dir = workdir("cli_4d");
    let path = dir.join("in.fits");
    make_4d_cube(&path);

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "total"]);
    assert!(!ok, "multi-Stokes cube should be rejected:\n{log}");
    assert!(
        log.contains("Stokes"),
        "error should mention Stokes:\n{log}"
    );
    assert!(
        !dir.join("in.sm.fits").exists(),
        "no output should be written for a rejected cube"
    );
}

#[test]
fn cli_verbose_logs_per_channel_beams() {
    let dir = workdir("cli_verbose");
    let path = dir.join("in.fits");
    make_varied_cube(&path);

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "total", "-v"]);
    assert!(ok, "binary failed:\n{log}");

    // -v (DEBUG) must report current/target/kernel for each channel.
    for c in 0..NFREQ {
        assert!(
            log.contains(&format!("Channel {c}: current")),
            "missing per-channel beam log for channel {c}:\n{log}"
        );
    }
    assert!(
        log.contains("Initialising output cube"),
        "missing init log:\n{log}"
    );
}

/// Build a 4D cube in CASA axis order (RA, DEC, STOKES, FREQ) with one Stokes
/// plane, a varying beam per channel (via BEAMS), and the reference channel on
/// axis 4.
fn make_casa_order_cube(path: &std::path::Path) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type: fitsio::images::ImageType::Float,
            dimensions: &[NFREQ, 1, NY, NX],
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let mut data = vec![0.0f32; NX * NY * NFREQ];
    for c in 0..NFREQ {
        data[c * NX * NY + (NY / 2) * NX + NX / 2] = 1.0 + c as f32;
    }
    hdu.write_image(&mut f, &data).unwrap();
    hdu.write_key(&mut f, "CDELT1", -0.0005f64).unwrap();
    hdu.write_key(&mut f, "CDELT2", 0.0005f64).unwrap();
    hdu.write_key(&mut f, "CTYPE3", "STOKES").unwrap();
    hdu.write_key(&mut f, "CRPIX3", 1.0f64).unwrap();
    hdu.write_key(&mut f, "CTYPE4", "FREQ").unwrap();
    hdu.write_key(&mut f, "CRPIX4", 2.0f64).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
    hdu.write_key(&mut f, "BMAJ", 0.005f64).unwrap();
    hdu.write_key(&mut f, "BMIN", 0.004f64).unwrap();
    hdu.write_key(&mut f, "BPA", 0.0f64).unwrap();
}

/// A CASA-order cube (Stokes on axis 3, frequency on axis 4) is read by its
/// CTYPEs: the channels come from NAXIS4 and the reference channel from CRPIX4,
/// rather than a single Stokes plane being mistaken for multiple Stokes and
/// the cube rejected.
#[test]
fn casa_axis_order_cube_is_read_and_smoothed() {
    let dir = workdir("casa_order");
    let path = dir.join("in.fits");
    make_casa_order_cube(&path);

    let meta = cube_io::read_cube_meta(&path).unwrap();
    assert_eq!((meta.nfreq, meta.nstokes, meta.crpix_freq), (NFREQ, 1, 2.0));
    assert!(meta.is_4d);

    let (ok, log) = run_cli(&[
        "3d",
        path.to_str().unwrap(),
        "--mode",
        "total",
        "--bmaj",
        "30",
        "--bmin",
        "30",
        "--bpa",
        "0",
    ]);
    assert!(ok, "binary failed:\n{log}");
    let out = dir.join("in.sm.fits");
    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    assert_eq!((out_meta.nfreq, out_meta.nstokes), (NFREQ, 1));
    for c in 0..NFREQ {
        let plane = read_channel::<f32>(&out, c, &out_meta);
        // Channel c's point source had amplitude 1 + c, so the smoothed peaks
        // keep that order: the channels were not shuffled or mixed up.
        let peak = plane.iter().cloned().fold(f32::MIN, f32::max);
        let first = read_channel::<f32>(&out, 0, &out_meta);
        let first_peak = first.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (peak / first_peak - (1.0 + c as f32)).abs() < 1e-3,
            "channel {c}: peak ratio {}",
            peak / first_peak
        );
    }
}

/// A reference pixel before the first channel (negative CRPIX, as a WCS
/// referenced to an off-axis frequency can have) attaches the *first*
/// channel's beam to the primary header, not the last one's.
#[test]
fn negative_crpix_uses_first_channel_beam() {
    let dir = workdir("neg_crpix");
    let path = dir.join("in.fits");
    make_varied_cube(&path);
    {
        let mut f = FitsFile::edit(&path).unwrap();
        let hdu = f.primary_hdu().unwrap();
        hdu.write_key(&mut f, "CRPIX3", -5.0f64).unwrap();
    }

    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "natural"]);
    assert!(ok, "binary failed:\n{log}");
    let out = dir.join("in.sm.fits");
    let meta = cube_io::read_cube_meta(&out).unwrap();
    let (bmaj, ..) = read_primary_beam_keys(&out);
    let first = meta.beams[0].unwrap();
    let last = meta.beams[NFREQ - 1].unwrap();
    assert!(
        (bmaj - first.major_deg()).abs() < 1e-9 && (bmaj - last.major_deg()).abs() > 1e-6,
        "primary BMAJ {bmaj} should be channel 0's {} (not channel {}'s {})",
        first.major_deg(),
        NFREQ - 1,
        last.major_deg()
    );
}

/// A NaN BMIN masks the channel instead of becoming a near-zero axis, and a
/// row with BMIN > BMAJ is read with the axes swapped.
#[test]
fn beams_table_nan_and_swapped_axes() {
    let dir = workdir("beams_nan_swapped");
    let path = dir.join("in.fits");
    make_cube_with_beams(
        &path,
        &[16.0, 17.0, 12.0],
        &[12.0, f32::NAN, 18.0],
        &[0.0, 0.0, 30.0],
    );
    let meta = cube_io::read_cube_meta(&path).unwrap();
    assert!(meta.beams[0].is_some());
    assert!(meta.beams[1].is_none(), "NaN BMIN must mask the channel");
    let swapped = meta.beams[2].expect("swapped row is kept");
    assert!((swapped.major_arcsec() - 18.0).abs() < 1e-4);
    assert!((swapped.minor_arcsec() - 12.0).abs() < 1e-4);
    assert!((swapped.pa_deg() - -60.0).abs() < 1e-4);

    std::fs::write(
        dir.join("beamlog.log.txt"),
        "#Channel BMAJ[arcsec] BMIN[arcsec] BPA[deg]\n0 16 12 0\n1 17 nan 0\n2 12 18 30\n",
    )
    .unwrap();
    let beams = cube_io::read_beamlog(&dir.join("beamlog.log.txt")).unwrap();
    assert!(beams[1].is_zero());
    assert!((beams[2].major_arcsec() - 18.0).abs() < 1e-9);
}

/// Pixel sizes come from the CD matrix when the header has no CDELT.
#[test]
fn cd_matrix_header() {
    let dir = workdir("cd_matrix");
    let path = dir.join("in.fits");
    let rot = 30f64.to_radians();
    // Unrotated, then rotated by 30 deg with 1" x 2" pixels.
    let cases = [
        ([-0.001, 0.0, 0.0, 0.001], (0.001, 0.001)),
        (
            [
                -0.001 * rot.cos(),
                -0.002 * rot.sin(),
                -0.001 * rot.sin(),
                0.002 * rot.cos(),
            ],
            (0.001, 0.002),
        ),
    ];
    for (cd, (dx, dy)) in cases {
        make_cube(&path);
        {
            let mut f = FitsFile::edit(&path).unwrap();
            let hdu = f.primary_hdu().unwrap();
            atfits_rs::delete_key(&mut f, "CDELT1").unwrap();
            atfits_rs::delete_key(&mut f, "CDELT2").unwrap();
            for (key, v) in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"].into_iter().zip(cd) {
                hdu.write_key(&mut f, key, v).unwrap();
            }
        }
        let meta = cube_io::read_cube_meta(&path).unwrap();
        assert!((meta.dx_deg - dx).abs() < 1e-12, "{}", meta.dx_deg);
        assert!((meta.dy_deg - dy).abs() < 1e-12, "{}", meta.dy_deg);
    }
}

/// An explicit target is checked only against channels within the cutoff.
#[test]
fn cli_target_ignores_cutoff_channels() {
    let dir = workdir("cli_target_cutoff");
    let path = dir.join("in.fits");
    make_cube_with_beams(
        &path,
        &[16.0, 30.0, 18.0],
        &[12.0, 30.0, 14.0],
        &[0.0, 0.0, 10.0],
    );
    let (ok, log) = run_cli(&[
        "3d",
        path.to_str().unwrap(),
        "--cutoff",
        "20",
        "--bmaj",
        "20",
        "--bmin",
        "20",
        "--bpa",
        "0",
    ]);
    assert!(ok, "{log}");
    let out = dir.join("in.sm.fits");
    let meta = cube_io::read_cube_meta(&out).unwrap();
    assert!(
        read_channel::<f32>(&out, 1, &meta)
            .iter()
            .all(|v| v.is_nan())
    );
    assert!(
        read_channel::<f32>(&out, 0, &meta)
            .iter()
            .all(|v| v.is_finite())
    );
}

/// An integer cube is written out as floats, so blanked channels can hold NaN
/// and smoothed values are not rounded.
#[test]
fn cli_integer_cube_writes_floats() {
    let dir = workdir("cli_integer");
    let path = dir.join("in.fits");
    make_cube_typed(
        &path,
        fitsio::images::ImageType::Short,
        &[16.0, 0.0, 18.0],
        &[12.0, 0.0, 14.0],
        &[0.0, 0.0, 10.0],
    );
    let (ok, log) = run_cli(&["3d", path.to_str().unwrap(), "--mode", "total"]);
    assert!(ok, "{log}");
    let out = dir.join("in.sm.fits");
    let mut f = FitsFile::open(&out).unwrap();
    let bitpix: i64 = f.primary_hdu().unwrap().read_key(&mut f, "BITPIX").unwrap();
    assert_eq!(bitpix, -32);
    let meta = cube_io::read_cube_meta(&out).unwrap();
    assert!(
        read_channel::<f32>(&out, 1, &meta)
            .iter()
            .all(|v| v.is_nan())
    );
    let plane = read_channel::<f32>(&out, 0, &meta);
    assert!(
        plane.iter().any(|v| v.fract() != 0.0),
        "values were rounded"
    );
}

/// A channel over the cutoff is blanked even when the target could contain it.
#[test]
fn cli_cutoff_blanks_even_when_target_contains() {
    let dir = workdir("cli_cutoff_contained");
    let path = dir.join("in.fits");
    make_cube_with_beams(
        &path,
        &[16.0, 30.0, 18.0],
        &[12.0, 30.0, 14.0],
        &[0.0, 0.0, 10.0],
    );
    let (ok, log) = run_cli(&[
        "3d",
        path.to_str().unwrap(),
        "--cutoff",
        "20",
        "--bmaj",
        "40",
        "--bmin",
        "40",
        "--bpa",
        "0",
    ]);
    assert!(ok, "{log}");
    let out = dir.join("in.sm.fits");
    let meta = cube_io::read_cube_meta(&out).unwrap();
    assert!(
        read_channel::<f32>(&out, 1, &meta)
            .iter()
            .all(|v| v.is_nan())
    );
}
