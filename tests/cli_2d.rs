//! End-to-end tests of `convolvers 2d`.

use std::path::{Path, PathBuf};

use convolve_rs::{Beam, BrightnessUnit, read_fits, read_fits_header, smooth};
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};
use ndarray::Array2;

const N: usize = 32;
const PIX_DEG: f64 = 2.0 / 3600.0;

fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("convolve_rs_2d_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn pixels(nplanes: usize) -> Vec<f64> {
    (0..nplanes * N * N)
        .map(|k| ((k * 37) % 101) as f64 / 101.0 + (k / (N * N)) as f64)
        .collect()
}

/// Write an image of shape `dims` (FITS order) with the given beam in arcsec.
fn make_image(path: &Path, dims: &[usize], data_type: ImageType, beam: (f64, f64, f64)) {
    let mut f = FitsFile::create(path)
        .with_custom_primary(&ImageDescription {
            data_type,
            dimensions: &dims.iter().rev().copied().collect::<Vec<_>>(),
        })
        .overwrite()
        .open()
        .unwrap();
    let hdu = f.primary_hdu().unwrap();
    let nplanes = dims[2..].iter().product();
    hdu.write_image(&mut f, &pixels(nplanes)).unwrap();
    hdu.write_key(&mut f, "CDELT1", -PIX_DEG).unwrap();
    hdu.write_key(&mut f, "CDELT2", PIX_DEG).unwrap();
    hdu.write_key(&mut f, "BMAJ", beam.0 / 3600.0).unwrap();
    hdu.write_key(&mut f, "BMIN", beam.1 / 3600.0).unwrap();
    hdu.write_key(&mut f, "BPA", beam.2).unwrap();
    hdu.write_key(&mut f, "BUNIT", "Jy/beam").unwrap();
}

fn make_2d(path: &Path, beam: (f64, f64, f64)) {
    make_image(path, &[N, N], ImageType::Float, beam);
}

fn run(args: &[&str]) -> (bool, String) {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_convolvers"))
        .arg("2d")
        .args(args)
        .output()
        .unwrap();
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), log)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn header_beam(path: &Path) -> Beam {
    read_fits_header(path).unwrap().beam
}

/// An image over the cutoff is written as NaN; the run does not abort.
#[test]
fn cutoff_blanks_image() {
    let dir = workdir("cutoff");
    let (a, b) = (dir.join("a.fits"), dir.join("b.fits"));
    make_2d(&a, (10.0, 10.0, 0.0));
    make_2d(&b, (30.0, 30.0, 0.0));
    let (ok, log) = run(&[s(&a), s(&b), "--cutoff", "20"]);
    assert!(ok, "{log}");
    let out_a = read_fits::<f32>(&dir.join("a.sm.fits")).unwrap();
    let out_b = read_fits::<f32>(&dir.join("b.sm.fits")).unwrap();
    assert!(out_a.image.iter().all(|v| v.is_finite()));
    assert!(out_b.image.iter().all(|v| v.is_nan()));
}

/// An explicit target is checked only against images that will be smoothed.
#[test]
fn target_ignores_cutoff_images() {
    let dir = workdir("target_cutoff");
    let (a, b) = (dir.join("a.fits"), dir.join("b.fits"));
    make_2d(&a, (10.0, 10.0, 0.0));
    make_2d(&b, (30.0, 30.0, 0.0));
    let (ok, log) = run(&[
        s(&a),
        s(&b),
        "--cutoff",
        "20",
        "--bmaj",
        "15",
        "--bmin",
        "15",
        "--bpa",
        "0",
    ]);
    assert!(ok, "{log}");
    let (ok, log) = run(&[s(&a), s(&b), "--bmaj", "15", "--bmin", "15", "--bpa", "0"]);
    assert!(
        !ok,
        "a target smaller than an unflagged image must fail:\n{log}"
    );
}

/// A zero-size header beam blanks the image rather than dividing by zero.
#[test]
fn zero_beam_is_blanked() {
    let dir = workdir("zero_beam");
    let (a, b) = (dir.join("a.fits"), dir.join("b.fits"));
    make_2d(&a, (10.0, 10.0, 0.0));
    make_2d(&b, (0.0, 0.0, 0.0));
    let (ok, log) = run(&[s(&a), s(&b)]);
    assert!(ok, "{log}");
    let out_b = read_fits::<f32>(&dir.join("b.sm.fits")).unwrap();
    assert!(out_b.image.iter().all(|v| v.is_nan()));
}

/// The output header carries the new beam, not the input's.
#[test]
fn output_header_has_new_beam() {
    let dir = workdir("header_beam");
    let a = dir.join("a.fits");
    make_2d(&a, (10.0, 8.0, 0.0));
    let (ok, log) = run(&[s(&a), "--bmaj", "20", "--bmin", "15", "--bpa", "30"]);
    assert!(ok, "{log}");
    let got = header_beam(&dir.join("a.sm.fits"));
    assert!(
        got.approx_eq(&Beam::from_arcsec(20.0, 15.0, 30.0).unwrap()),
        "{got}"
    );
}

/// BITPIX = -64 images are read and written in f64.
#[test]
fn f64_image_keeps_precision() {
    let dir = workdir("f64");
    let a = dir.join("a.fits");
    make_image(&a, &[N, N], ImageType::Double, (10.0, 10.0, 0.0));
    let (ok, log) = run(&[s(&a), "--bmaj", "20", "--bmin", "20", "--bpa", "0"]);
    assert!(ok, "{log}");
    let input = Array2::from_shape_vec((N, N), pixels(1)).unwrap();
    let want = smooth(
        &input,
        &Beam::from_arcsec(10.0, 10.0, 0.0).unwrap(),
        &Beam::from_arcsec(20.0, 20.0, 0.0).unwrap(),
        PIX_DEG,
        PIX_DEG,
        None,
        BrightnessUnit::JyPerBeam,
    )
    .unwrap();
    let out = dir.join("a.sm.fits");
    let got = read_fits::<f64>(&out).unwrap().image;
    // Far below f32 resolution; not exact because BMAJ goes through the header.
    let diff = (&got - &want)
        .mapv(f64::abs)
        .fold(0.0, |a: f64, &b| a.max(b));
    assert!(diff < 1e-10, "max diff {diff}");
    let mut f = FitsFile::open(&out).unwrap();
    let bitpix: i64 = f.primary_hdu().unwrap().read_key(&mut f, "BITPIX").unwrap();
    assert_eq!(bitpix, -64);
}

/// A 4D image with several planes: the first plane is smoothed and the output
/// has degenerate leading axes, as racs_tools writes it.
#[test]
fn four_d_image_uses_first_plane() {
    let dir = workdir("four_d");
    let a = dir.join("a.fits");
    make_image(&a, &[N, N, 3, 1], ImageType::Float, (10.0, 10.0, 0.0));
    let (ok, log) = run(&[s(&a), "--bmaj", "20", "--bmin", "20", "--bpa", "0"]);
    assert!(ok, "{log}");
    let input = Array2::from_shape_vec((N, N), pixels(1))
        .unwrap()
        .mapv(|x| x as f32);
    let want = smooth(
        &input,
        &Beam::from_arcsec(10.0, 10.0, 0.0).unwrap(),
        &Beam::from_arcsec(20.0, 20.0, 0.0).unwrap(),
        PIX_DEG,
        PIX_DEG,
        None,
        BrightnessUnit::JyPerBeam,
    )
    .unwrap();
    let out = dir.join("a.sm.fits");
    let header = read_fits_header(&out).unwrap();
    assert_eq!((header.nx, header.ny, header.nplanes), (N, N, 1));
    assert_eq!(read_fits::<f32>(&out).unwrap().image, want);
}

fn dryrun(args: &[&str]) -> String {
    let (ok, log) = run(args);
    assert!(ok, "{log}");
    log.lines()
        .find(|l| l.starts_with("BMAJ="))
        .unwrap()
        .to_string()
}

/// A computed beam on an exact tenth stays there; a user target is not rounded.
#[test]
fn beam_rounding() {
    let dir = workdir("rounding");
    let a = dir.join("a.fits");
    make_2d(&a, (30.5, 20.0, 0.0));
    assert!(dryrun(&[s(&a), "--dryrun"]).starts_with("BMAJ=30.5000\" BMIN=20.0000\""));
    let line = dryrun(&[
        s(&a),
        "--dryrun",
        "--bmaj",
        "30.54",
        "--bmin",
        "30.54",
        "--bpa",
        "0",
    ]);
    assert!(line.starts_with("BMAJ=30.5400\""), "{line}");
}

/// An integer image is written out as floats.
#[test]
fn integer_image_writes_floats() {
    let dir = workdir("integer");
    let a = dir.join("a.fits");
    make_image(&a, &[N, N], ImageType::Long, (10.0, 10.0, 0.0));
    {
        let mut f = FitsFile::edit(&a).unwrap();
        let data: Vec<i32> = (0..N * N).map(|k| (k % 7) as i32).collect();
        f.primary_hdu().unwrap().write_image(&mut f, &data).unwrap();
    }
    let (ok, log) = run(&[s(&a), "--bmaj", "20", "--bmin", "20", "--bpa", "0"]);
    assert!(ok, "{log}");
    let out = dir.join("a.sm.fits");
    let mut f = FitsFile::open(&out).unwrap();
    let bitpix: i64 = f.primary_hdu().unwrap().read_key(&mut f, "BITPIX").unwrap();
    assert_eq!(bitpix, -32);
    let image = read_fits::<f32>(&out).unwrap().image;
    assert!(
        image.iter().any(|v| v.fract() != 0.0),
        "values were rounded"
    );
}
