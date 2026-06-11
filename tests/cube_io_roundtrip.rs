//! Regression tests for cube output initialisation and per-channel writes.
//!
//! Guards against a bug where `init_output_cube` produced an output whose
//! primary HDU had NAXIS=0 (the header was never copied), so the first
//! `write_channel` failed with the misleading cfitsio error 302
//! ("column number < 1 or > tfields").

use convolve_rs::beam::Beam;
use convolve_rs::cube_io::{self, CubeMode};
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
        ColumnDescription::new("BMAJ").with_type(ColumnDataType::Float).create().unwrap(),
        ColumnDescription::new("BMIN").with_type(ColumnDataType::Float).create().unwrap(),
        ColumnDescription::new("BPA").with_type(ColumnDataType::Float).create().unwrap(),
        ColumnDescription::new("CHAN").with_type(ColumnDataType::Int).create().unwrap(),
        ColumnDescription::new("POL").with_type(ColumnDataType::Int).create().unwrap(),
    ];
    let t = f.create_table("BEAMS", &cols).unwrap();
    t.write_col(&mut f, "BMAJ", &[20.0f32; NFREQ]).unwrap();
    t.write_col(&mut f, "BMIN", &[15.0f32; NFREQ]).unwrap();
    t.write_col(&mut f, "BPA", &[0.0f32; NFREQ]).unwrap();
    t.write_col(&mut f, "CHAN", &(0..NFREQ as i32).collect::<Vec<_>>()).unwrap();
    t.write_col(&mut f, "POL", &[0i32; NFREQ]).unwrap();
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
    cube_io::init_output_cube(&path, &out, &target, CubeMode::Total, &meta).unwrap();

    // The output primary HDU must carry the copied 3D image structure.
    {
        let mut f = FitsFile::edit(out.to_string_lossy().to_string()).unwrap();
        let hdu = f.primary_hdu().unwrap();
        let naxis: i64 = hdu.read_key(&mut f, "NAXIS").unwrap();
        assert_eq!(naxis, 3, "output primary HDU lost its NAXIS=3 structure");
    }

    // Write each channel a distinct constant, then read it back.
    for c in 0..NFREQ {
        let plane = Array2::from_elem((NY, NX), c as f32 + 0.5);
        cube_io::write_channel(&out, c, &plane, &meta).unwrap();
    }
    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    for c in 0..NFREQ {
        let plane = cube_io::read_channel(&out, c, &out_meta).unwrap();
        assert!(plane.iter().all(|&v| (v - (c as f32 + 0.5)).abs() < 1e-6));
    }
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
    cube_io::init_output_cube(&path, &out, &target, CubeMode::Natural, &meta).unwrap();

    for c in 0..NFREQ {
        let plane = Array2::from_elem((NY, NX), c as f32 + 0.5);
        cube_io::write_channel(&out, c, &plane, &meta).unwrap();
    }

    // BEAMS extension must round-trip the per-channel target beams.
    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    for c in 0..NFREQ {
        let b = out_meta.beams[c].expect("channel beam present");
        assert!((b.major_arcsec() - (25.0 + c as f64)).abs() < 1e-3);
    }
}

// ── End-to-end CLI smoke tests ─────────────────────────────────────────────────
//
// Drive the actual `convolvers` binary on a synthetic cube — the same path that
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
    // A single bright central pixel per channel — well-defined to convolve.
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
        ColumnDescription::new("BMAJ").with_type(ColumnDataType::Float).create().unwrap(),
        ColumnDescription::new("BMIN").with_type(ColumnDataType::Float).create().unwrap(),
        ColumnDescription::new("BPA").with_type(ColumnDataType::Float).create().unwrap(),
        ColumnDescription::new("CHAN").with_type(ColumnDataType::Int).create().unwrap(),
        ColumnDescription::new("POL").with_type(ColumnDataType::Int).create().unwrap(),
    ];
    let t = f.create_table("BEAMS", &cols).unwrap();
    t.write_col(&mut f, "BMAJ", &bmaj).unwrap();
    t.write_col(&mut f, "BMIN", &bmin).unwrap();
    t.write_col(&mut f, "BPA", &bpa).unwrap();
    t.write_col(&mut f, "CHAN", &(0..NFREQ as i32).collect::<Vec<_>>()).unwrap();
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

    // Beamlog should be written alongside the output.
    assert!(dir.join("beamlog.in.sm.txt").exists(), "beamlog missing:\n{log}");
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
    assert!(log.contains("Initialising output cube"), "missing init log:\n{log}");
}
