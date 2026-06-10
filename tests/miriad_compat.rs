use convolve_rs::cube_io;
/// Integration tests for convolve-rs, validating against MIRIAD.
///
/// The MIRIAD test (`test_smooth_matches_miriad`) mirrors the Python `test_2d.py`
/// fixture: a 100×100 Gaussian image with BMAJ=20", BMIN=10", BPA=10°, pixel
/// scale 2.5"/pix, smoothed to 40"×40"@0°.
///
/// All other tests are pure-Rust algebraic invariant checks that do not require
/// any external binaries.
use convolve_rs::{Beam, BrightnessUnit, common_beam, fftfreq, gaussft, smooth};
use fitsio::FitsFile;
use fitsio::images::{ImageDescription, ImageType};
use fitsio::tables::{ColumnDataType, ColumnDescription};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Test constants ────────────────────────────────────────────────────────────

/// Find the MIRIAD binary directory.
///
/// Checks the `MIRIAD_BIN` environment variable first, then searches `PATH`
/// for a directory that contains both `fits` and `convol`.
fn find_miriad_bin_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MIRIAD_BIN") {
        let p = PathBuf::from(dir);
        if p.join("fits").exists() && p.join("convol").exists() {
            return Some(p);
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            if dir.join("fits").exists() && dir.join("convol").exists() {
                return Some(dir);
            }
        }
    }
    None
}

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
fn make_gaussian_image(
    nrow: usize,
    ncol: usize,
    bmaj_pix: f64,
    bmin_pix: f64,
    pa_deg: f64,
) -> Array2<f32> {
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
    bunit: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (nrows, ncols) = image.dim();
    let pix_deg = pix_arcsec / 3600.0;

    let description = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &[nrows, ncols], // C row-major: NAXIS1=ncols, NAXIS2=nrows
    };
    let mut fptr = FitsFile::create(path.to_str().ok_or("non-UTF-8 path")?)
        .with_custom_primary(&description)
        .open()?;

    let hdu = fptr.primary_hdu()?;

    // Beam keywords (degrees, FITS convention)
    hdu.write_key(&mut fptr, "BMAJ", beam.major_deg)?;
    hdu.write_key(&mut fptr, "BMIN", beam.minor_deg)?;
    hdu.write_key(&mut fptr, "BPA", beam.pa_deg)?;

    // Minimal WCS
    hdu.write_key(&mut fptr, "CDELT1", -pix_deg)?; // RA: negative
    hdu.write_key(&mut fptr, "CDELT2", pix_deg)?;
    hdu.write_key(&mut fptr, "CRPIX1", (ncols / 2 + 1) as f64)?;
    hdu.write_key(&mut fptr, "CRPIX2", (nrows / 2 + 1) as f64)?;
    hdu.write_key(&mut fptr, "CRVAL1", 0.0_f64)?;
    hdu.write_key(&mut fptr, "CRVAL2", 0.0_f64)?;
    hdu.write_key(&mut fptr, "CTYPE1", "RA---SIN")?;
    hdu.write_key(&mut fptr, "CTYPE2", "DEC--SIN")?;
    hdu.write_key(&mut fptr, "EQUINOX", 2000.0_f64)?;
    hdu.write_key(&mut fptr, "BUNIT", bunit)?;

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

fn miriad_available() -> bool {
    find_miriad_bin_dir().is_some()
}

/// Invoke MIRIAD `fits op=xyin` + `convol options=final` + `fits op=xyout` to
/// smooth `input_fits` to `target_beam`, writing the result to `output_fits`.
fn miriad_smooth(
    input_fits: &Path,
    output_fits: &Path,
    target_beam: &Beam,
) -> Result<(), Box<dyn std::error::Error>> {
    let bin_dir = find_miriad_bin_dir().ok_or("MIRIAD not found")?;
    let mir_root = bin_dir.parent().unwrap_or(&bin_dir).to_path_buf();

    let tmpdir = output_fits.parent().unwrap();
    let mir_in = tmpdir.join("in.im");
    let mir_out = tmpdir.join("sm.im");

    // FITS → MIRIAD
    let status = Command::new(bin_dir.join("fits"))
        .env("MIR", &mir_root)
        .arg("op=xyin")
        .arg(format!("in={}", input_fits.display()))
        .arg(format!("out={}", mir_in.display()))
        .status()?;
    if !status.success() {
        return Err("fits op=xyin failed".into());
    }

    // Smooth
    let fwhm = format!(
        "fwhm={},{}",
        target_beam.major_arcsec(),
        target_beam.minor_arcsec(),
    );
    let pa = format!("pa={}", target_beam.pa_deg);
    let status = Command::new(bin_dir.join("convol"))
        .env("MIR", &mir_root)
        .arg(format!("map={}", mir_in.display()))
        .arg(&fwhm)
        .arg(&pa)
        .arg("options=final")
        .arg(format!("out={}", mir_out.display()))
        .status()?;
    if !status.success() {
        return Err("convol failed".into());
    }

    // MIRIAD → FITS
    let status = Command::new(bin_dir.join("fits"))
        .env("MIR", &mir_root)
        .arg("op=xyout")
        .arg(format!("in={}", mir_out.display()))
        .arg(format!("out={}", output_fits.display()))
        .status()?;
    if !status.success() {
        return Err("fits op=xyout failed".into());
    }

    Ok(())
}

/// Create a unique temp directory for a test.
fn test_tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("convolve_rs_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).expect("failed to create tmpdir");
    dir
}

// ── Pure-Rust algebraic tests ─────────────────────────────────────────────────

/// Convolve(A, B).deconvolve(A) == B  for any valid A, B.
#[test]
fn test_beam_conv_deconv_roundtrip() {
    let test_cases = [
        ((20.0, 10.0, 10.0), (5.0, 4.0, 30.0)),
        ((15.0, 15.0, 0.0), (8.0, 6.0, 45.0)),
        ((12.0, 8.0, 90.0), (3.0, 3.0, 0.0)),
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
            recovered.major_deg,
            b.major_deg,
        );
        assert!(
            (recovered.minor_deg - b.minor_deg).abs() < tol,
            "minor mismatch: {:.10e} vs {:.10e}",
            recovered.minor_deg,
            b.minor_deg,
        );
    }
}

/// Test parameters from Python test_2d.py: old=20×10@10, target=40×40@0.
#[test]
fn test_beam_deconvolve_test_params() {
    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();

    let conv_beam = target.deconvolve(&old).unwrap();

    // Verify by re-convolving: conv_beam ⊛ old ≈ target
    let reconv = old.convolve(&conv_beam);
    let tol = 1e-9;
    assert!(
        (reconv.major_deg - target.major_deg).abs() < tol,
        "reconv.major={:.10e} target.major={:.10e}",
        reconv.major_deg,
        target.major_deg,
    );
    assert!(
        (reconv.minor_deg - target.minor_deg).abs() < tol,
        "reconv.minor={:.10e} target.minor={:.10e}",
        reconv.minor_deg,
        target.minor_deg,
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
    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
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
    let omega_ratio = (TARGET_BMAJ * TARGET_BMIN) / (OLD_BMAJ * OLD_BMIN); // = 8.0
    let expected_g_ratio = omega_ratio.sqrt(); // = sqrt(8)
    assert!(
        (g_ratio - expected_g_ratio).abs() < 1e-10,
        "g_ratio = {g_ratio:.10e}, expected sqrt(Ω_new/Ω_old) = {expected_g_ratio:.10e}",
    );

    // Filter must be real-valued (Gaussian FT has no imaginary component)
    let max_imag = g_final.iter().map(|c| c.im.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_imag < 1e-15,
        "g_final has non-zero imaginary parts: max|im|={max_imag:.2e}"
    );
}

/// gaussft values decrease monotonically from DC (Gaussian must attenuate high frequencies).
#[test]
fn test_gaussft_attenuates_high_freq() {
    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    // 1-D slice along u (v=0): check g_final[i, 0] decreases for positive u
    let n = 32usize;
    let u_freqs = fftfreq(n, pix_deg.to_radians());
    let v_freqs = fftfreq(1, pix_deg.to_radians()); // single-element v

    let (g_final, g_ratio) = gaussft(&old, &target, &u_freqs, &v_freqs);

    let dc = g_final[0].re;
    assert!((dc - g_ratio).abs() < 1e-12);

    // Positive-frequency bins (indices 1..n/2) should be smaller than DC
    let m = n.div_ceil(2);
    for (i, bin) in g_final.iter().enumerate().take(m).skip(1) {
        let val = bin.re;
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
    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let bmaj_pix = OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = OLD_BMIN / PIX_ARCSEC;
    let image = make_gaussian_image(NROW, NCOL, bmaj_pix, bmin_pix, OLD_BPA);

    let smoothed = smooth(
        &image,
        &old,
        &target,
        pix_deg,
        pix_deg,
        None,
        BrightnessUnit::JyPerBeam,
    )
    .unwrap();

    let sum_in: f64 = image.iter().map(|&x| x as f64).sum();
    let sum_out: f64 = smoothed.iter().map(|&x| x as f64).sum();

    // Expected: Ω_new / Ω_old = (40*40) / (20*10) = 8
    let area_old = old.area_sr();
    let area_new = target.area_sr();
    let expected_ratio = area_new / area_old;
    let actual_ratio = sum_out / sum_in;

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
    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
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

    let smoothed = smooth(
        &image,
        &old,
        &target,
        pix_deg,
        pix_deg,
        None,
        BrightnessUnit::JyPerBeam,
    )
    .unwrap();

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
        result.major_arcsec(),
        b.major_arcsec(),
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
        result.major_arcsec(),
        large.major_arcsec(),
    );
}

// ── MIRIAD integration test ───────────────────────────────────────────────────

/// Compare our FFT convolution against MIRIAD `convol options=final`.
///
/// Skips automatically if the MIRIAD binaries are not present on the system.
#[test]
fn test_smooth_matches_miriad() {
    if !miriad_available() {
        eprintln!("MIRIAD not found on PATH or MIRIAD_BIN, skipping test_smooth_matches_miriad");
        return;
    }

    let tmpdir = test_tmpdir("miriad");
    let input_fits = tmpdir.join("input.fits");
    let miriad_fits = tmpdir.join("miriad.fits");

    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    // Build test image
    let bmaj_pix = OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = OLD_BMIN / PIX_ARCSEC;
    let image = make_gaussian_image(NROW, NCOL, bmaj_pix, bmin_pix, OLD_BPA);

    // Write FITS
    write_test_fits(&input_fits, &image, &old, PIX_ARCSEC, "Jy/beam")
        .expect("write_test_fits failed");

    // Run MIRIAD
    miriad_smooth(&input_fits, &miriad_fits, &target).expect("MIRIAD smooth failed");

    // Run Rust
    let rust_result = smooth(
        &image,
        &old,
        &target,
        pix_deg,
        pix_deg,
        None,
        BrightnessUnit::JyPerBeam,
    )
    .expect("Rust smooth failed");

    // Read MIRIAD pixels
    let mir_pixels = read_fits_pixels(&miriad_fits);
    let rust_pixels: Vec<f32> = rust_result.into_raw_vec_and_offset().0;

    assert_eq!(
        mir_pixels.len(),
        rust_pixels.len(),
        "pixel count mismatch: MIRIAD {} vs Rust {}",
        mir_pixels.len(),
        rust_pixels.len(),
    );

    // Compare pixel-by-pixel
    let atol = 1e-3_f32;
    let mismatches: Vec<(usize, f32, f32)> = mir_pixels
        .iter()
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

/// Compare our Kelvin (brightness-temperature) smoothing against MIRIAD,
/// natively: the input FITS is written with `BUNIT=K`, so MIRIAD `convol`
/// performs a straight (surface-brightness-preserving) convolution with no
/// Jy/beam flux scaling — exactly what [`BrightnessUnit::Kelvin`] does.
///
/// Skips automatically if the MIRIAD binaries are not present on the system.
#[test]
fn test_kelvin_matches_miriad() {
    if !miriad_available() {
        eprintln!("MIRIAD not found on PATH or MIRIAD_BIN, skipping test_kelvin_matches_miriad");
        return;
    }

    let tmpdir = test_tmpdir("miriad_kelvin");
    let input_fits = tmpdir.join("input.fits");
    let miriad_fits = tmpdir.join("miriad.fits");

    let old = Beam::from_arcsec(OLD_BMAJ, OLD_BMIN, OLD_BPA).unwrap();
    let target = Beam::from_arcsec(TARGET_BMAJ, TARGET_BMIN, TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let bmaj_pix = OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = OLD_BMIN / PIX_ARCSEC;
    let image = make_gaussian_image(NROW, NCOL, bmaj_pix, bmin_pix, OLD_BPA);

    // Write the image in Kelvin so MIRIAD skips its Jy/beam scaling.
    write_test_fits(&input_fits, &image, &old, PIX_ARCSEC, "K").expect("write_test_fits failed");
    miriad_smooth(&input_fits, &miriad_fits, &target).expect("MIRIAD smooth failed");

    // Rust Kelvin smoothing: surface-brightness preserving (no flux scaling).
    let rust_result = smooth(
        &image,
        &old,
        &target,
        pix_deg,
        pix_deg,
        None,
        BrightnessUnit::Kelvin,
    )
    .expect("Rust smooth failed");

    let mir_pixels = read_fits_pixels(&miriad_fits);
    let rust_pixels: Vec<f32> = rust_result.into_raw_vec_and_offset().0;

    assert_eq!(
        mir_pixels.len(),
        rust_pixels.len(),
        "pixel count mismatch: MIRIAD {} vs Rust {}",
        mir_pixels.len(),
        rust_pixels.len(),
    );

    let atol = 1e-3_f32;
    let mismatches: Vec<(usize, f32, f32)> = mir_pixels
        .iter()
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
             MIRIAD(K)={m0:.6e} Rust(K)={r0:.6e} diff={:.6e}",
            (m0 - r0).abs(),
        );
    }

    let _ = std::fs::remove_dir_all(&tmpdir);
}

// ── Additional common-beam tests ──────────────────────────────────────────────

/// Three different elliptical beams — common beam must enclose all three.
#[test]
fn test_common_beam_three_different() {
    let b1 = Beam::from_arcsec(20.0, 10.0, 30.0).unwrap();
    let b2 = Beam::from_arcsec(18.0, 12.0, -20.0).unwrap();
    let b3 = Beam::from_arcsec(15.0, 15.0, 0.0).unwrap();
    let result = common_beam(&[b1, b2, b3], 1e-4, 200, 5e-4).unwrap();

    // Must be at least as large as the largest input beam's major axis.
    assert!(
        result.major_arcsec() >= 20.0 - 0.1,
        "common major {:.3}\" < largest input 20.0\"",
        result.major_arcsec(),
    );
    // Every input beam must fit inside the result.
    for b in [b1, b2, b3] {
        assert!(
            result.deconvolve(&b).is_ok(),
            "input beam {b} does not fit inside common beam {result}",
        );
    }
}

// ── 3D cube helpers ────────────────────────────────────────────────────────────

const CUBE_NCHAN: usize = 5;
const CUBE_NY: usize = 100;
const CUBE_NX: usize = 100;
const CUBE_OLD_BMAJ: f64 = 50.0;
const CUBE_OLD_BMIN: f64 = 10.0;
const CUBE_OLD_BPA: f64 = 0.0;
const CUBE_TARGET_BMAJ: f64 = 60.0;
const CUBE_TARGET_BMIN: f64 = 60.0;
const CUBE_TARGET_BPA: f64 = 0.0;

/// Write a 4D FITS cube [nstokes=1, nchan, ny, nx] with a CASAMBM BEAMS table.
///
/// Data layout (C row-major, matching FITS Fortran-order with last index fastest):
/// flat[c * ny * nx + j * nx + i] = channel c, row j, col i.
fn write_test_cube_casambm(
    path: &Path,
    channel_images: &[Array2<f32>],
    beams: &[Beam],
    pix_arcsec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let nchan = channel_images.len();
    assert_eq!(beams.len(), nchan, "beam/channel count mismatch");
    let (ny, nx) = channel_images[0].dim();
    let pix_deg = pix_arcsec / 3600.0;

    // fitsio reverses dimensions → pass [nstokes, nchan, ny, nx] to get
    // NAXIS1=nx, NAXIS2=ny, NAXIS3=nchan, NAXIS4=1.
    let description = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &[1, nchan, ny, nx],
    };
    let mut fptr = FitsFile::create(path.to_str().ok_or("non-UTF-8 path")?)
        .with_custom_primary(&description)
        .open()?;
    let hdu = fptr.primary_hdu()?;

    // WCS
    hdu.write_key(&mut fptr, "CDELT1", -pix_deg)?;
    hdu.write_key(&mut fptr, "CDELT2", pix_deg)?;
    hdu.write_key(&mut fptr, "CDELT3", 1.0e6_f64)?;
    hdu.write_key(&mut fptr, "CDELT4", 1.0_f64)?;
    hdu.write_key(&mut fptr, "CRPIX1", (nx / 2 + 1) as f64)?;
    hdu.write_key(&mut fptr, "CRPIX2", (ny / 2 + 1) as f64)?;
    hdu.write_key(&mut fptr, "CRPIX3", 1.0_f64)?;
    hdu.write_key(&mut fptr, "CRPIX4", 1.0_f64)?;
    hdu.write_key(&mut fptr, "CRVAL1", 0.0_f64)?;
    hdu.write_key(&mut fptr, "CRVAL2", 0.0_f64)?;
    hdu.write_key(&mut fptr, "CRVAL3", 1.4e9_f64)?;
    hdu.write_key(&mut fptr, "CRVAL4", 1.0_f64)?;
    hdu.write_key(&mut fptr, "CTYPE1", "RA---SIN")?;
    hdu.write_key(&mut fptr, "CTYPE2", "DEC--SIN")?;
    hdu.write_key(&mut fptr, "CTYPE3", "FREQ")?;
    hdu.write_key(&mut fptr, "CTYPE4", "STOKES")?;
    hdu.write_key(&mut fptr, "EQUINOX", 2000.0_f64)?;
    hdu.write_key(&mut fptr, "BUNIT", "Jy/beam")?;
    hdu.write_key(&mut fptr, "CASAMBM", "T")?;

    // Reference beam from channel 0 (CRPIX3=1 → 0-based index 0).
    hdu.write_key(&mut fptr, "BMAJ", beams[0].major_deg)?;
    hdu.write_key(&mut fptr, "BMIN", beams[0].minor_deg)?;
    hdu.write_key(&mut fptr, "BPA", beams[0].pa_deg)?;

    // Write all channel data as one flat C-order array.
    let flat: Vec<f32> = channel_images
        .iter()
        .flat_map(|img| img.iter().copied())
        .collect();
    hdu.write_image(&mut fptr, &flat)?;

    // Append BEAMS binary-table extension.
    let bmaj_v: Vec<f32> = beams.iter().map(|b| b.major_arcsec() as f32).collect();
    let bmin_v: Vec<f32> = beams.iter().map(|b| b.minor_arcsec() as f32).collect();
    let bpa_v: Vec<f32> = beams.iter().map(|b| b.pa_deg as f32).collect();
    let chan_v: Vec<i32> = (0..nchan as i32).collect();
    let pol_v: Vec<i32> = vec![0; nchan];

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

    let tbl = fptr.create_table("BEAMS", &[col_bmaj, col_bmin, col_bpa, col_chan, col_pol])?;
    tbl.write_col(&mut fptr, "BMAJ", &bmaj_v)?;
    tbl.write_col(&mut fptr, "BMIN", &bmin_v)?;
    tbl.write_col(&mut fptr, "BPA", &bpa_v)?;
    tbl.write_col(&mut fptr, "CHAN", &chan_v)?;
    tbl.write_col(&mut fptr, "POL", &pol_v)?;
    tbl.write_key(&mut fptr, "NCHAN", nchan as i64)?;
    tbl.write_key(&mut fptr, "NPOL", 1i64)?;

    Ok(())
}

/// Read raw pixel data for every channel of a FITS cube, returned as a flat Vec.
///
/// Does not go through `read_cube_meta`; useful when the output FITS may lack
/// a BEAMS table (e.g. plain MIRIAD output).
fn read_cube_flat(path: &Path, nchan: usize, ny: usize, nx: usize) -> Vec<f32> {
    let path_str = path.to_string_lossy().into_owned();
    let mut fptr = FitsFile::open(&path_str).expect("open fits");
    let hdu = fptr.primary_hdu().expect("primary hdu");
    let plane = ny * nx;
    let mut out = Vec::with_capacity(nchan * plane);
    for c in 0..nchan {
        let start = c * plane;
        let end = start + plane;
        let pixels: Vec<f32> = hdu
            .read_section(&mut fptr, start, end)
            .expect("read_section");
        out.extend_from_slice(&pixels);
    }
    out
}

/// MIRIAD smoothing of a spectral cube: fits op=xyin → convol options=cube → fits op=xyout.
///
/// Uses `options=cube` which reads per-channel beams from the CASAMBM BEAMS table
/// and computes the appropriate convolution kernel for each channel (equivalent to
/// `options=final` applied per channel), matching Python `test_robust_3d`.
fn miriad_smooth_cube(
    input_fits: &Path,
    output_fits: &Path,
    target: &Beam,
) -> Result<(), Box<dyn std::error::Error>> {
    let bin_dir = find_miriad_bin_dir().ok_or("MIRIAD not found")?;
    let mir_root = bin_dir.parent().unwrap_or(&bin_dir).to_path_buf();
    let tmpdir = output_fits.parent().unwrap();
    let mir_in = tmpdir.join("cube_in.im");
    let mir_out = tmpdir.join("cube_sm.im");

    let status = Command::new(bin_dir.join("fits"))
        .env("MIR", &mir_root)
        .arg("op=xyin")
        .arg(format!("in={}", input_fits.display()))
        .arg(format!("out={}", mir_in.display()))
        .status()?;
    if !status.success() {
        return Err("fits op=xyin failed".into());
    }

    let status = Command::new(bin_dir.join("convol"))
        .env("MIR", &mir_root)
        .arg(format!("map={}", mir_in.display()))
        .arg(format!(
            "fwhm={},{}",
            target.major_arcsec(),
            target.minor_arcsec()
        ))
        .arg(format!("pa={}", target.pa_deg))
        .arg("options=cube")
        .arg(format!("out={}", mir_out.display()))
        .status()?;
    if !status.success() {
        return Err("convol failed".into());
    }

    let status = Command::new(bin_dir.join("fits"))
        .env("MIR", &mir_root)
        .arg("op=xyout")
        .arg(format!("in={}", mir_out.display()))
        .arg(format!("out={}", output_fits.display()))
        .status()?;
    if !status.success() {
        return Err("fits op=xyout failed".into());
    }

    Ok(())
}

// ── 3D cube tests ─────────────────────────────────────────────────────────────

/// Write a CASAMBM cube then read back each channel with `read_channel`,
/// verifying the pixel values round-trip exactly.
#[test]
fn test_cube_io_roundtrip() {
    let tmpdir = test_tmpdir("cube_roundtrip");
    let path = tmpdir.join("cube.fits");

    let nchan = 3usize;
    let ny = 20usize;
    let nx = 20usize;

    // Each channel gets a distinct uniform value so we can tell them apart.
    let channels: Vec<Array2<f32>> = (0..nchan)
        .map(|c| Array2::from_elem((ny, nx), (c as f32 + 1.0) * 10.0))
        .collect();
    let beam = Beam::from_arcsec(CUBE_OLD_BMAJ, CUBE_OLD_BMIN, CUBE_OLD_BPA).unwrap();
    let beams: Vec<Beam> = vec![beam; nchan];

    write_test_cube_casambm(&path, &channels, &beams, PIX_ARCSEC)
        .expect("write_test_cube_casambm failed");

    let meta = cube_io::read_cube_meta(&path).expect("read_cube_meta failed");
    assert_eq!(meta.nfreq, nchan);
    assert_eq!(meta.ny, ny);
    assert_eq!(meta.nx, nx);
    assert!(meta.is_4d);

    for c in 0..nchan {
        let plane = cube_io::read_channel(&path, c, &meta)
            .unwrap_or_else(|e| panic!("read_channel({c}) failed: {e}"));
        let expected = (c as f32 + 1.0) * 10.0;
        for &v in plane.iter() {
            assert!(
                (v - expected).abs() < 1e-4,
                "channel {c}: expected {expected}, got {v}",
            );
        }
    }

    let _ = std::fs::remove_dir_all(&tmpdir);
}

/// Write a CASAMBM cube with per-channel beams, parse it with `read_cube_meta`,
/// and verify every beam is recovered within f32 precision (≈0.1 arcsec).
#[test]
fn test_cube_casambm_beams_roundtrip() {
    let tmpdir = test_tmpdir("cube_beams");
    let path = tmpdir.join("cube.fits");

    let nchan = 4usize;
    let ny = 10usize;
    let nx = 10usize;

    let beams_in: Vec<Beam> = (0..nchan)
        .map(|c| Beam::from_arcsec(20.0 + c as f64 * 5.0, 10.0, 0.0).unwrap())
        .collect();
    let channels: Vec<Array2<f32>> = (0..nchan).map(|_| Array2::zeros((ny, nx))).collect();

    write_test_cube_casambm(&path, &channels, &beams_in, PIX_ARCSEC).expect("write failed");

    let meta = cube_io::read_cube_meta(&path).expect("read_cube_meta failed");
    assert_eq!(meta.beams.len(), nchan);

    for (c, (expected, actual)) in beams_in.iter().zip(meta.beams.iter()).enumerate() {
        let actual = actual.unwrap_or_else(|| panic!("channel {c} beam is None"));
        let tol_as = 0.1; // f32 storage precision ~= 0.004 arcsec for ~50"
        assert!(
            (actual.major_arcsec() - expected.major_arcsec()).abs() < tol_as,
            "channel {c}: BMAJ {:.3}\" vs {:.3}\"",
            actual.major_arcsec(),
            expected.major_arcsec(),
        );
        assert!(
            (actual.minor_arcsec() - expected.minor_arcsec()).abs() < tol_as,
            "channel {c}: BMIN {:.3}\" vs {:.3}\"",
            actual.minor_arcsec(),
            expected.minor_arcsec(),
        );
    }

    let _ = std::fs::remove_dir_all(&tmpdir);
}

/// For each channel of a cube, smoothing a normalised PSF image to a larger beam
/// should scale the pixel sum by Ω_new / Ω_old (the beam area ratio).
///
/// This is the per-channel version of `test_smooth_beam_area_ratio`.
#[test]
fn test_cube_channel_beam_area_ratio() {
    let old = Beam::from_arcsec(CUBE_OLD_BMAJ, CUBE_OLD_BMIN, CUBE_OLD_BPA).unwrap();
    let target = Beam::from_arcsec(CUBE_TARGET_BMAJ, CUBE_TARGET_BMIN, CUBE_TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let bmaj_pix = CUBE_OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = CUBE_OLD_BMIN / PIX_ARCSEC;
    let expected_ratio = (target.major_deg * target.minor_deg) / (old.major_deg * old.minor_deg);

    for c in 0..CUBE_NCHAN {
        let image = make_gaussian_image(CUBE_NY, CUBE_NX, bmaj_pix, bmin_pix, CUBE_OLD_BPA);
        let smoothed = smooth(
            &image,
            &old,
            &target,
            pix_deg,
            pix_deg,
            None,
            BrightnessUnit::JyPerBeam,
        )
        .unwrap();

        let sum_in: f64 = image.iter().map(|&x| x as f64).sum();
        let sum_out: f64 = smoothed.iter().map(|&x| x as f64).sum();
        let ratio = sum_out / sum_in;

        assert!(
            (ratio - expected_ratio).abs() / expected_ratio < 0.02,
            "channel {c}: sum ratio {ratio:.6} vs expected {expected_ratio:.6} (2% tol)",
        );
    }
}

/// Compare our per-channel UV convolution against MIRIAD `convol options=cube`.
///
/// Mirrors the Python `test_robust_3d` in RACS-tools/tests/test_3d.py:
///   - 4D CASAMBM cube, all channels BMAJ=50", BMIN=10", BPA=0°, pix=2.5"/pix
///   - Target beam: BMAJ=60", BMIN=60", BPA=0°
///   - MIRIAD: `convol options=cube` (per-channel beam from BEAMS table)
///   - Rust: channel-by-channel `smooth` from old_beam → target_beam
///   - Pixel-by-pixel comparison at atol = 1e-3
#[test]
fn test_cube_smoothed_matches_miriad_3d() {
    if !miriad_available() {
        eprintln!(
            "MIRIAD not found on PATH or MIRIAD_BIN, skipping test_cube_smoothed_matches_miriad_3d"
        );
        return;
    }

    let tmpdir = test_tmpdir("cube_miriad3d");
    let input_fits = tmpdir.join("cube_input.fits");
    let miriad_fits = tmpdir.join("cube_miriad.fits");

    let old = Beam::from_arcsec(CUBE_OLD_BMAJ, CUBE_OLD_BMIN, CUBE_OLD_BPA).unwrap();
    let target = Beam::from_arcsec(CUBE_TARGET_BMAJ, CUBE_TARGET_BMIN, CUBE_TARGET_BPA).unwrap();
    let pix_deg = PIX_ARCSEC / 3600.0;

    let bmaj_pix = CUBE_OLD_BMAJ / PIX_ARCSEC;
    let bmin_pix = CUBE_OLD_BMIN / PIX_ARCSEC;
    let image = make_gaussian_image(CUBE_NY, CUBE_NX, bmaj_pix, bmin_pix, CUBE_OLD_BPA);

    let beams: Vec<Beam> = vec![old; CUBE_NCHAN];
    let channels: Vec<Array2<f32>> = vec![image.clone(); CUBE_NCHAN];

    write_test_cube_casambm(&input_fits, &channels, &beams, PIX_ARCSEC)
        .expect("write_test_cube_casambm failed");

    miriad_smooth_cube(&input_fits, &miriad_fits, &target).expect("MIRIAD cube smooth failed");

    // Read the MIRIAD output pixels directly (may not have CASAMBM).
    let mir_flat = read_cube_flat(&miriad_fits, CUBE_NCHAN, CUBE_NY, CUBE_NX);

    let atol = 1e-3_f32;
    for c in 0..CUBE_NCHAN {
        let rust_plane = smooth(
            &image,
            &old,
            &target,
            pix_deg,
            pix_deg,
            None,
            BrightnessUnit::JyPerBeam,
        )
        .expect("smooth failed");
        let rust_flat: Vec<f32> = rust_plane.into_raw_vec_and_offset().0;

        let plane_start = c * CUBE_NY * CUBE_NX;
        let mir_plane = &mir_flat[plane_start..plane_start + CUBE_NY * CUBE_NX];

        let mismatches: Vec<(usize, f32, f32)> = mir_plane
            .iter()
            .zip(rust_flat.iter())
            .enumerate()
            .filter(|&(_, (m, r))| (m - r).abs() > atol)
            .map(|(i, (m, r))| (i, *m, *r))
            .collect();

        if !mismatches.is_empty() {
            let n = mismatches.len();
            let (idx0, m0, r0) = mismatches[0];
            panic!(
                "channel {c}: {n} pixel(s) differ by > {atol}: \
                 first at local index {idx0}: MIRIAD={m0:.6e} Rust={r0:.6e} diff={:.6e}",
                (m0 - r0).abs(),
            );
        }
    }

    let _ = std::fs::remove_dir_all(&tmpdir);
}
