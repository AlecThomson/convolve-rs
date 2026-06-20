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
    cube_io::init_output_cube(&path, &out, &target, CubeMode::Natural, &meta).unwrap();

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
    let res = cube_io::init_output_cube(&path, &path, &target, CubeMode::Total, &meta);
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

/// CASAMBM must be a FITS *logical* (readable as `bool`), not a quoted string —
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
    cube_io::init_output_cube(&path, &total, &target, CubeMode::Total, &meta).unwrap();
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
    cube_io::init_output_cube(&path, &nat, &target, CubeMode::Natural, &meta).unwrap();
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
    // (a point source convolved to a Gaussian) — not zeros or NaNs.
    for c in 0..NFREQ {
        let plane = cube_io::read_channel(&out, c, &meta).unwrap();
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
    cube_io::init_output_cube(&path, &out, &target, CubeMode::Total, &meta).unwrap();

    // 1 + 1e-10 is not representable in f32, so an f32 round-trip would lose it.
    let val = 1.0 + 1e-10_f64;
    for c in 0..NFREQ {
        let plane = Array2::from_elem((NY, NX), val);
        cube_io::write_channel_as::<f64>(&out, c, &plane, &meta).unwrap();
    }

    let out_meta = cube_io::read_cube_meta(&out).unwrap();
    assert_eq!(out_meta.dtype, PixelType::F64, "output kept BITPIX -64");
    for c in 0..NFREQ {
        let plane = cube_io::read_channel_as::<f64>(&out, c, &out_meta).unwrap();
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
        let plane = cube_io::read_channel_as::<f64>(&out, c, &meta).unwrap();
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

/// The streaming `CubeWriter::create`/`finish` path (handle held open, data unit
/// written once) must produce output identical to the legacy create-close-reopen
/// `init_output_cube` + `write_channel` path: same primary beam keywords, same
/// CASAMBM logical, same per-channel BEAMS table, and the same channel pixels.
/// This pins the performance refactor to byte-for-byte behaviour preservation.
#[test]
fn streaming_writer_matches_init_output_cube() {
    for mode in [CubeMode::Total, CubeMode::Natural] {
        let tag = if mode == CubeMode::Natural {
            "stream_nat"
        } else {
            "stream_tot"
        };
        let dir = workdir(tag);
        let path = dir.join("in.fits");
        make_cube(&path);
        let meta = cube_io::read_cube_meta(&path).unwrap();

        // Distinct per-channel target beams so the BEAMS table is non-trivial.
        let target: Vec<Option<Beam>> = (0..NFREQ)
            .map(|c| Some(Beam::from_arcsec(25.0 + c as f64, 20.0, 5.0).unwrap()))
            .collect();
        let planes: Vec<Array2<f32>> = (0..NFREQ)
            .map(|c| Array2::from_elem((NY, NX), c as f32 + 0.5))
            .collect();

        // Legacy path: init (create-close-reopen) then per-channel write.
        let legacy = dir.join("legacy.fits");
        cube_io::init_output_cube(&path, &legacy, &target, mode, &meta).unwrap();
        for (c, plane) in planes.iter().enumerate() {
            cube_io::write_channel(&legacy, c, plane, &meta).unwrap();
        }

        // Streaming path: single open handle from create through finish.
        let stream = dir.join("stream.fits");
        {
            let mut w = cube_io::CubeWriter::create(&path, &stream, &target, mode, &meta).unwrap();
            // Write out of order to mimic the parallel pipeline's arrival order.
            for c in (0..NFREQ).rev() {
                w.write_channel(c, &planes[c], &meta).unwrap();
            }
            w.finish().unwrap();
        }

        // Primary beam keywords + CASAMBM must match exactly.
        assert_eq!(
            read_primary_beam_keys(&legacy),
            read_primary_beam_keys(&stream),
            "{tag}: primary beam keywords differ between paths"
        );

        // BEAMS table (or its absence) and channel pixels must match.
        let lm = cube_io::read_cube_meta(&legacy).unwrap();
        let sm = cube_io::read_cube_meta(&stream).unwrap();
        for c in 0..NFREQ {
            assert_eq!(
                lm.beams[c].map(|b| (b.major_arcsec(), b.minor_arcsec(), b.pa_deg)),
                sm.beams[c].map(|b| (b.major_arcsec(), b.minor_arcsec(), b.pa_deg)),
                "{tag}: channel {c} beam differs"
            );
            let lp = cube_io::read_channel(&legacy, c, &lm).unwrap();
            let sp = cube_io::read_channel(&stream, c, &sm).unwrap();
            assert_eq!(lp, sp, "{tag}: channel {c} pixels differ");
        }
    }
}

/// Build a 3D cube like `make_varied_cube` but with channel 1's beam masked
/// (BMAJ = 0), so the streaming pipeline must blank that channel.
fn make_cube_masked_chan1(path: &std::path::Path) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&fitsio::images::ImageDescription {
            data_type: fitsio::images::ImageType::Float,
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

    // Channel 1 masked (BMAJ = 0); channels 0 and 2 are valid.
    let bmaj = [16.0f32, 0.0, 18.0];
    let bmin = [12.0f32, 0.0, 14.0];
    let bpa = [0.0f32, 0.0, 10.0];
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

/// A masked-beam channel must be written as NaN (explicit no-data), while the
/// valid channels stay finite — never zero-filled or corrupted to infinity by a
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
    let masked = cube_io::read_channel(&out, 1, &meta).unwrap();
    assert!(
        masked.iter().all(|v| v.is_nan()),
        "masked channel should be all-NaN, got finite/zero pixels"
    );

    // Valid channels: finite, non-empty, and crucially not infinite.
    for c in [0, 2] {
        let plane = cube_io::read_channel(&out, c, &meta).unwrap();
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
