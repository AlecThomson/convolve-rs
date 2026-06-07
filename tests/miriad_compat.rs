/// Integration tests for convolve-rs, validating against MIRIAD.
///
/// The MIRIAD test (`test_smooth_matches_miriad`) mirrors the Python `test_2d.py`
/// fixture: a 100×100 Gaussian image with BMAJ=20", BMIN=10", BPA=10°, pixel
/// scale 2.5"/pix, smoothed to 40"×40"@0°.
///
/// All other tests are pure-Rust algebraic invariant checks that do not require
/// any external binaries.

use convolve_rs::{Beam, gaussft, fftfreq, smooth};
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Test constants ────────────────────────────────────────────────────────────

/// Path to MIRIAD binaries.
const MIRIAD_BIN: &str = "/Users/alec.thomson/bin/miriad/darwin_arm64/bin";

const PIX_ARCSEC: f64 = 2.5;
const OLD_BMAJ: f64 = 20.0;
const OLD_BMIN: f64 = 10.0;
const OLD_BPA: f64 = 10.0;
const TARGET_BMAJ: f64 = 40.0;
const TARGET_BMIN: f64 = 40.0;
const TARGET_BPA: f64 = 0.0;
const NROW: usize = 100;
const NCOL: usize = 100;

// ── Image generation ──────────────────────────────────────────────────────────

/// Generate a normalised 2D elliptical Gaussian image on an (nrow × ncol) grid.
///
/// PA follows the astronomical convention (degrees East of North):
///   * u = Δcol·sin(PA) + Δrow·cos(PA)  → along major axis
///   * v = Δcol·cos(PA) − Δrow·sin(PA)  → along minor axis
///
/// This matches the convention used by MIRIAD and astropy's Gaussian2DKernel
/// via `theta = (90 − PA)°`.
fn make_gaussian_image(nrow: usize, ncol: usize,
    bmaj_pix: f64, bmin_pix: f64, pa_deg: f64) -> Array2<f32>
{
    let fwhm_to_sigma: f64 = 2.0 * (2.0_f64 * 2.0_f64.ln()).sqrt();
    let sigma_maj = bmaj_pix / fwhm_to_sigma;
    let sigma_min = bmin_pix / fwhm_to_sigma;
    let pa_rad = pa_deg.to_radians();
    let cy = (nrow as f64 - 1.0) / 2.0;
    let cx = (ncol as f64 - 1.0) / 2.0;

    let mut data = Array2::<f64>::zeros((nrow, ncol));
    for r in 0..nrow {
        for c in 0..ncol {
            let dr = r as f64 - cy;
            let dc = c as f64 - cx;
            let u = dc * pa_rad.sin() + dr * pa_rad.cos();
            let v = dc * pa_rad.cos() - dr * pa_rad.sin();
            data[[r, c]] = (-0.5 * ((u / sigma_maj).powi(2) + (v / sigma_min).powi(2))).exp();
        }
    }

    let peak = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    data.mapv(|x| (x / peak) as f32)
}

// ── FITS helpers ──────────────────────────────────────────────────────────────

/// Write a 2D float image as a minimal FITS file with WCS + beam headers.
fn write_test_fits(
    path: &Path,
    image: &Array2<f32>,
    beam: &Beam,
    pix_arcsec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let (nrows, ncols) = image.dim();
    let pix_deg = pix_arcsec / 3600.0;

    let description = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &[nrows, ncols],   // C row-major: NAXIS1=ncols, NAXIS2=nrows
    };
    let mut fptr = FitsFile::create(path.to_str().ok_or("non-UTF-8 path")?)
        .with_custom_primary(&description)
        .open()?;

    let hdu = fptr.primary_hdu()?;

    // Beam keywords (degrees, FITS convention)
    hdu.write_key(&mut fptr, "BMAJ", beam.major_deg)?;
    hdu.write_key(&mut fptr, "BMIN", beam.minor_deg)?;
    hdu.write_key(&mut fptr, "BPA",  beam.pa_deg)?;

    // Minimal WCS
    hdu.write_key(&mut fptr, "CDELT1", -pix_deg)?;     // RA: negative
    hdu.write_key(&mut fptr, "CDELT2",  pix_deg)?;
    hdu.write_key(&mut fptr, "CRPIX1", (ncols / 2 + 1) as f64)?;
    hdu.write_key(&mut fptr, "CRPIX2", (nrows / 2 + 1) as f64)?;
    hdu.write_key(&mut fptr, "CRVAL1", 0.0_f64)?;
    hdu.write_key(&mut fptr, "CRVAL2", 0.0_f64)?;
    hdu.write_key(&mut fptr, "CTYPE1", "RA---SIN")?;
    hdu.write_key(&mut fptr, "CTYPE2", "DEC--SIN")?;
    hdu.write_key(&mut fptr, "EQUINOX", 2000.0_f64)?;
    hdu.write_key(&mut fptr, "BUNIT",  "Jy/beam")?;

    let flat: Vec<f32> = image.iter().copied().collect();
    hdu.write_image(&mut fptr, &flat)?;

    Ok(())
}

/// Read image data from a FITS file.
fn read_fits_pixels(path: &Path) -> Vec<f32> {
    let data = convolve_rs::read_fits(path).expect("read_fits_pixels failed");
    data.image.into_raw_vec_and_offset().0
}

// ── MIRIAD helpers ────────────────────────────────────────────────────────────

fn miriad_bin(name: &str) -> PathBuf {
    Path::new(MIRIAD_BIN).join(name)
}

fn miriad_available() -> bool {
    miriad_bin("fits").exists() && miriad_bin("convol").exists()
}

/// Invoke MIRIAD `fits op=xyin` + `convol options=final` + `fits op=xyout` to
/// smooth `input_fits` to `target_beam`, writing the result to `output_fits`.
fn miriad_smooth(
    input_fits: &Path,
    output_fits: &Path,
    target_beam: &Beam,
) -> Result<(), Box<dyn std::error::Error>> {
    let mir_root = Path::new(MIRIAD_BIN).parent().unwrap();

    let tmpdir = output_fits.parent().unwrap();
    let mir_in  = tmpdir.join("in.im");
    let mir_out = tmpdir.join("sm.im");

    // FITS → MIRIAD
    let status = Command::new(miriad_bin("fits"))
        .env("MIR", mir_root)
        .arg(format!("op=xyin"))
        .arg(format!("in={}", input_fits.display()))
        .arg(format!("out={}", mir_in.display()))
        .status()?;
    if !status.success() { return Err("fits op=xyin failed".into()); }

    // Smooth
    let fwhm = format!(
        "fwhm={},{}",
        target_beam.major_arcsec(),
        target_beam.minor_arcsec(),
    );
    let pa = format!("pa={}", target_beam.pa_deg);
    let status = Command::new(miriad_bin("convol"))
        .env("MIR", mir_root)
        .arg(format!("map={}", mir_in.display()))
        .arg(&fwhm)
        .arg(&pa)
        .arg("options=final")
        .arg(format!("out={}", mir_out.display()))
        .status()?;
    if !status.success() { return Err("convol failed".into()); }

    // MIRIAD → FITS
    let status = Command::new(miriad_bin("fits"))
        .env("MIR", mir_root)
        .arg("op=xyout")
        .arg(format!("in={}", mir_out.display()))
        .arg(format!("out={}", output_fits.display()))
        .status()?;
    if !status.success() { return Err("fits op=xyout failed".into()); }

    Ok(())
}

/// Create a unique temp directory for a test.
fn test_tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("convolve_rs_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).expect("failed to create tmpdir");
    dir
}

// ── Pure-Rust algebraic tests ─────────────────────────────────────────────────

/// Convolve(A, B).deconvolve(A) == B  for any valid A, B.
#[test]
fn test_beam_conv_deconv_roundtrip() {
    let test_cases = [
        ((20.0, 10.0, 10.0), (5.0, 4.0, 30.0)),
        ((15.0, 15.0,  0.0), (8.0, 6.0, 45.0)),
        ((12.0,  8.0, 90.0), (3.0, 3.0,  0.0)),
    ];
    for ((maj_a, min_a, pa_a), (maj_b, min_b, pa_b)) in test_cases {
        let a = Beam::from_arcsec(maj_a, min_a, pa_a).unwrap();
        let b = Beam::from_arcsec(maj_b, min_b, pa_b).unwrap();
        let convolved = a.convolve(&b);
        let recovered = convolved.deconvolve(&a).unwrap();
        let tol = 1e-9;
        assert!(
            (recovered.major_deg - b.major_deg).abs() < tol,
            "({maj_a}×{min_a}@{pa_a}) + ({maj_b}×{min_b}@{pa_b}): \
             major {:.10e} vs {:.10e}",
            recovered.major_deg, b.major_deg,
        );
        assert!(
            (recovered.minor_deg - b.minor_deg).abs() < tol,
            "minor mismatch: {:.10e} vs {:.10e}",
            recovered.minor_deg, b.minor_deg,
        );
    }
}

/// Test parameters from Python test_2d.py: old=20×10@10, target=40×40@0.
#[test]
fn test_beam_deconvolve_test_params() {
    let old    = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();

    let conv_beam = target.deconvolve(&old).unwrap();

    // Verify by re-convolving: conv_beam ⊛ old ≈ target
    let reconv = old.convolve(&conv_beam);
    let tol = 1e-9;
    assert!(
        (reconv.major_deg - target.major_deg).abs() < tol,
        "reconv.major={:.10e} target.major={:.10e}",
        reconv.major_deg, target.major_deg,
    );
    assert!(
        (reconv.minor_deg - target.minor_deg).abs() < tol,
        "reconv.minor={:.10e} target.minor={:.10e}",
        reconv.minor_deg, target.minor_deg,
    );

    // Sanity: convolving beam must fit inside the target beam
    assert!(
        conv_beam.major_deg <= target.major_deg + 1e-12,
        "conv beam larger than target"
    );
    assert!(conv_beam.major_deg > 0.0, "conv beam is zero");
}

/// At DC (u=0, v=0) the gaussft filter equals g_ratio exactly.
///
/// Note: g_ratio = sqrt(Ω_new / Ω_old).  For smoothing to a *larger* beam,
/// Ω_new > Ω_old, so g_ratio > 1.  The final Jy/beam scaling is g_ratio²
/// (applied as g_ratio once inside g_final, and once more in smooth()), which
/// equals Ω_new / Ω_old — the correct ratio for Jy/beam units.
#[test]
fn test_gaussft_dc_equals_g_ratio() {
    let old    = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let u_freqs = fftfreq(NROW, pix_deg.to_radians());
    let v_freqs = fftfreq(NCOL, pix_deg.to_radians());

    let (g_final, g_ratio) = gaussft(&old, &target, &u_freqs, &v_freqs);

    // DC component is at index [0, 0] in row-major layout
    let dc = g_final[0].re;
    assert!(
        (dc - g_ratio).abs() < 1e-12,
        "g_final[DC] = {dc:.10e}, g_ratio = {g_ratio:.10e}",
    );

    // g_ratio = sqrt(Ω_new / Ω_old); must be positive and > 1 when smoothing to larger beam
    assert!(g_ratio > 0.0, "g_ratio must be positive");
    let omega_ratio = (TARGET_BMAJ * TARGET_BMIN) / (OLD_BMAJ * OLD_BMIN);  // = 8.0
    let expected_g_ratio = omega_ratio.sqrt();  // = sqrt(8)
    assert!(
        (g_ratio - expected_g_ratio).abs() < 1e-10,
        "g_ratio = {g_ratio:.10e}, expected sqrt(Ω_new/Ω_old) = {expected_g_ratio:.10e}",
    );

    // Filter must be real-valued (Gaussian FT has no imaginary component)
    let max_imag = g_final.iter().map(|c| c.im.abs()).fold(0.0_f64, f64::max);
    assert!(max_imag < 1e-15, "g_final has non-zero imaginary parts: max|im|={max_imag:.2e}");
}

/// gaussft values decrease monotonically from DC (Gaussian must attenuate high frequencies).
#[test]
fn test_gaussft_attenuates_high_freq() {
    let old    = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    // 1-D slice along u (v=0): check g_final[i, 0] decreases for positive u
    let n = 32usize;
    let u_freqs = fftfreq(n, pix_deg.to_radians());
    let v_freqs = fftfreq(1, pix_deg.to_radians());  // single-element v

    let (g_final, g_ratio) = gaussft(&old, &target, &u_freqs, &v_freqs);

    let dc = g_final[0].re;
    assert!((dc - g_ratio).abs() < 1e-12);

    // Positive-frequency bins (indices 1..n/2) should be smaller than DC
    let m = (n + 1) / 2;
    for i in 1..m {
        let val = g_final[i].re;
        assert!(
            val < dc,
            "g_final[{i}] = {val:.6e} should be < DC = {dc:.6e}",
        );
    }
}

/// For a normalised Gaussian PSF image (peak=1), smoothing to a larger beam scales
/// the sum by Ω_new / Ω_old:
///
///   Σ(out) / Σ(in) ≈ Ω_new / Ω_old   (= 8 for our test params)
///
/// This is because the image represents 1 Jy/beam and:
///   Σ(PSF with FWHM f) ≈ Ω_beam / Ω_pix
/// so the ratio of sums equals the ratio of beam areas.
/// The integrated flux density S = Σ * Ω_pix / Ω_beam is conserved.
#[test]
fn test_smooth_beam_area_ratio() {
    let old    = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let bmaj_pix = OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = OLD_BMIN / PIX_ARCSEC;
    let image = make_gaussian_image(NROW, NCOL, bmaj_pix, bmin_pix, OLD_BPA);

    let smoothed = smooth(&image, &old, &target, pix_deg, pix_deg, None).unwrap();

    let sum_in: f64  = image.iter().map(|&x| x as f64).sum();
    let sum_out: f64 = smoothed.iter().map(|&x| x as f64).sum();

    // Expected: Ω_new / Ω_old = (40*40) / (20*10) = 8
    let area_old = old.area_sr();
    let area_new = target.area_sr();
    let expected_ratio = area_new / area_old;
    let actual_ratio   = sum_out / sum_in;

    // Allow 1% tolerance (image is truncated, not an infinite Gaussian)
    assert!(
        (actual_ratio - expected_ratio).abs() / expected_ratio < 0.01,
        "sum ratio {actual_ratio:.6} vs expected {expected_ratio:.6}",
    );
}

/// NaN regions must propagate through UV-plane convolution.
///
/// The NaN mask is convolved with the same UV filter (g_final).  A pixel is
/// blanked when `mask_conv >= 1.0` (matching the Python `~(mask_conv + 1 < 2)`
/// condition in `convolve_uv.py`).
///
/// For reliable propagation the kernel must cover most of the NaN region so
/// the integrated kernel weight over NaN pixels reaches 1.0.  We therefore use
/// a large NaN slab (the entire top half of a 100×100 image) and test that:
///   • pixels well inside the NaN slab are NaN in the output
///   • pixels well outside it (far from the NaN boundary) remain finite
#[test]
fn test_smooth_nan_propagation() {
    let old    = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let bmaj_pix = OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = OLD_BMIN / PIX_ARCSEC;
    let mut image = make_gaussian_image(NROW, NCOL, bmaj_pix, bmin_pix, OLD_BPA);

    // Blank the entire top half of the image.  Well inside this NaN slab, the
    // integrated g_final weight over the NaN area easily exceeds the threshold 1.0.
    for r in 0..(NROW / 2) {
        for c in 0..NCOL {
            image[[r, c]] = f32::NAN;
        }
    }

    let smoothed = smooth(&image, &old, &target, pix_deg, pix_deg, None).unwrap();

    // Pixels deep inside the NaN region (far from the valid boundary) must be NaN.
    // We check row 10 (40 pixels from the NaN/valid boundary at row 50).
    for c in 0..NCOL {
        assert!(
            smoothed[[10, c]].is_nan(),
            "expected NaN at [10,{c}], got {}",
            smoothed[[10, c]],
        );
    }

    // Pixels well into the valid region (far from the NaN boundary) must be finite.
    // Row 90 is 40 pixels below the NaN boundary.
    for c in 0..NCOL {
        assert!(
            smoothed[[90, c]].is_finite(),
            "expected finite at [90,{c}], got {}",
            smoothed[[90, c]],
        );
    }
}

/// Deconvolving a beam from itself should fail (result is a point source, not valid).
#[test]
fn test_beam_self_deconvolve_fails() {
    let b = Beam::from_arcsec(20.0, 10.0, 0.0).unwrap();
    assert!(b.deconvolve(&b).is_err(), "self-deconvolve should fail");
}

/// Deconvolving a larger beam from a smaller one must fail.
#[test]
fn test_beam_deconvolve_too_large_fails() {
    let small = Beam::from_arcsec(10.0, 8.0, 0.0).unwrap();
    let large = Beam::from_arcsec(20.0, 20.0, 0.0).unwrap();
    assert!(small.deconvolve(&large).is_err());
}

/// Two identical beams → common beam is equal to that beam.
#[test]
fn test_common_beam_identical() {
    use convolve_rs::common_beam;
    let b = Beam::from_arcsec(20.0, 10.0, 10.0).unwrap();
    let result = common_beam(&[b, b], 0.001, 200, 1e-5).unwrap();
    let tol = 0.05 / 3600.0; // 0.05 arcsec in degrees
    assert!(
        (result.major_deg - b.major_deg).abs() < tol,
        "major: {:.6} vs {:.6}",
        result.major_arcsec(), b.major_arcsec(),
    );
}

/// The common beam of two different circular beams is the larger one.
#[test]
fn test_common_beam_two_circular() {
    use convolve_rs::common_beam;
    let small = Beam::from_arcsec(10.0, 10.0, 0.0).unwrap();
    let large = Beam::from_arcsec(20.0, 20.0, 0.0).unwrap();
    let result = common_beam(&[small, large], 0.001, 200, 1e-5).unwrap();
    let tol = 0.1 / 3600.0; // 0.1 arcsec
    assert!(
        (result.major_deg - large.major_deg).abs() < tol,
        "common beam should equal the larger beam; got {:.4}\" expected {:.4}\"",
        result.major_arcsec(), large.major_arcsec(),
    );
}

// ── MIRIAD integration test ───────────────────────────────────────────────────

/// Compare our FFT convolution against MIRIAD `convol options=final`.
///
/// Skips automatically if the MIRIAD binaries are not present on the system.
#[test]
fn test_smooth_matches_miriad() {
    if !miriad_available() {
        eprintln!("MIRIAD not found at {MIRIAD_BIN}, skipping test_smooth_matches_miriad");
        return;
    }

    let tmpdir = test_tmpdir("miriad");
    let input_fits  = tmpdir.join("input.fits");
    let miriad_fits = tmpdir.join("miriad.fits");

    let old    = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    // Build test image
    let bmaj_pix = OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = OLD_BMIN / PIX_ARCSEC;
    let image = make_gaussian_image(NROW, NCOL, bmaj_pix, bmin_pix, OLD_BPA);

    // Write FITS
    write_test_fits(&input_fits, &image, &old, PIX_ARCSEC)
        .expect("write_test_fits failed");

    // Run MIRIAD
    miriad_smooth(&input_fits, &miriad_fits, &target)
        .expect("MIRIAD smooth failed");

    // Run Rust
    let rust_result = smooth(&image, &old, &target, pix_deg, pix_deg, None)
        .expect("Rust smooth failed");

    // Read MIRIAD pixels
    let mir_pixels = read_fits_pixels(&miriad_fits);
    let rust_pixels: Vec<f32> = rust_result.into_raw_vec_and_offset().0;

    assert_eq!(
        mir_pixels.len(), rust_pixels.len(),
        "pixel count mismatch: MIRIAD {} vs Rust {}",
        mir_pixels.len(), rust_pixels.len(),
    );

    // Compare pixel-by-pixel
    let atol = 1e-3_f32;
    let mismatches: Vec<(usize, f32, f32)> = mir_pixels.iter()
        .zip(rust_pixels.iter())
        .enumerate()
        .filter(|&(_, (m, r))| (m - r).abs() > atol)
        .map(|(i, (m, r))| (i, *m, *r))
        .collect();

    if !mismatches.is_empty() {
        let n = mismatches.len();
        let (idx0, m0, r0) = mismatches[0];
        panic!(
            "{n} pixel(s) differ by > {atol}: first at index {idx0}: \
             MIRIAD={m0:.6e} Rust={r0:.6e} diff={:.6e}",
            (m0 - r0).abs(),
        );
    }

    // Clean up
    let _ = std::fs::remove_dir_all(&tmpdir);
}
