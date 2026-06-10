/// Common beam algorithms.
///
/// Two algorithms are provided:
/// - `find_commonbeam_between`: analytic CASA algorithm for exactly 2 beams.
/// - `common_manybeams_mve`: Khachiyan minimum-volume-enclosing-ellipsoid for N beams.
/// - `common_beam`: dispatcher that picks the appropriate algorithm.
use crate::beam::{Beam, BeamError, deconvolve_deg};
use thiserror::Error;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Error)]
pub enum CommonBeamError {
    #[error("no beams provided")]
    NoBeans,
    #[error("all beams are invalid or flagged")]
    AllFlagged,
    #[error("Khachiyan algorithm did not converge after {0} iterations")]
    NoConvergence(usize),
    #[error("common beam does not deconvolve all inputs: {0}")]
    DeconvFailed(String),
    #[error("beam error: {0}")]
    Beam(#[from] BeamError),
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Find the smallest beam that all `beams` can be convolved to.
///
/// Uses the 2-beam analytic algorithm when `beams.len() == 2`, otherwise
/// the Khachiyan minimum-volume-enclosing-ellipse algorithm (same as
/// `radio_beam.Beams.common_beam(method='pts')`).
pub fn common_beam(
    beams: &[Beam],
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
) -> Result<Beam, CommonBeamError> {
    if beams.is_empty() {
        return Err(CommonBeamError::NoBeans);
    }
    if beams.len() == 1 {
        return Ok(beams[0]);
    }

    // Fast path: if the largest beam already contains all others.
    let largest = largest_beam(beams);
    if fits_in_beam(beams, &largest) {
        return Ok(largest);
    }

    if beams.len() == 2
        && let Ok(b) = find_commonbeam_between(&beams[0], &beams[1])
    {
        return Ok(b);
    }

    common_manybeams_mve(beams, tolerance, nsamps, epsilon)
}

// ── 2-beam analytic algorithm (CASA ia.commonbeam) ───────────────────────────

/// Find the common beam between exactly 2 beams using the CASA analytic method.
pub fn find_commonbeam_between(beam1: &Beam, beam2: &Beam) -> Result<Beam, CommonBeamError> {
    if beam1.approx_eq(beam2) {
        return Ok(*beam1);
    }

    let (large_beam, small_beam) = if beam1.area_sr() >= beam2.area_sr() {
        (beam1, beam2)
    } else {
        (beam2, beam1)
    };

    // If the large beam already contains the small beam, large beam is the answer.
    let deconv = large_beam.deconvolve_or_zero(small_beam);
    if deconv.is_finite() {
        return Ok(*large_beam);
    }

    let large_major = large_beam.major_arcsec();
    let large_minor = large_beam.minor_arcsec();
    let small_major = small_beam.major_arcsec();

    // If the small beam is circular the minor axis is the circle radius.
    if small_beam.is_circular(1e-6) {
        let beam = Beam::from_arcsec(large_major, small_major, large_beam.pa_deg)?;
        return Ok(beam);
    }

    // Wrap PA difference to [-π/2, π/2].
    let pa_diff_rad = ((small_beam.pa_deg - large_beam.pa_deg) * DEG2RAD
        + std::f64::consts::FRAC_PI_2
        + std::f64::consts::PI)
        .rem_euclid(std::f64::consts::PI)
        - std::f64::consts::FRAC_PI_2;

    // If beams are perpendicular the common beam axes are the two major axes.
    if (pa_diff_rad.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9 {
        let (major, minor) = if large_major >= small_major {
            (large_major, small_major)
        } else {
            (small_major, large_major)
        };
        let pa = if large_major >= small_major {
            large_beam.pa_deg
        } else {
            small_beam.pa_deg
        };
        return Beam::from_arcsec(major, minor, pa).map_err(Into::into);
    }

    // Transform to coordinate frame where large_beam is circular.
    let major_comb = (large_major * small_major).sqrt();
    let p = major_comb / large_major;
    let q = major_comb / large_minor;

    let (trans_maj_sc, _trans_min_sc, trans_pa_sc) =
        transform_ellipse_arcsec(small_major, small_beam.minor_arcsec(), pa_diff_rad, p, q);

    // Override the minor axis per the CASA algorithm.
    let trans_min_sc = major_comb;

    // Transform back.
    let (trans_maj_unsc, trans_min_unsc, trans_pa_unsc) =
        transform_ellipse_arcsec(trans_maj_sc, trans_min_sc, trans_pa_sc, 1.0 / p, 1.0 / q);

    let final_pa_deg = trans_pa_unsc.to_degrees() + large_beam.pa_deg;

    let eps = 100.0 * f64::EPSILON;
    let beam = Beam::from_arcsec(trans_maj_unsc + eps, trans_min_unsc + eps, final_pa_deg)?;

    Ok(beam)
}

/// Transform an ellipse (arcsec axes, PA in radians) by scaling factors (p, q).
/// Returns (major_arcsec, minor_arcsec, pa_rad). Port of radio_beam utils.transform_ellipse.
fn transform_ellipse_arcsec(
    major: f64,
    minor: f64,
    pa: f64,
    x_scale: f64,
    y_scale: f64,
) -> (f64, f64, f64) {
    let cospa = pa.cos();
    let sinpa = pa.sin();
    let cos2pa = cospa * cospa;
    let sin2pa = sinpa * sinpa;
    let major2 = major * major;
    let minor2 = minor * minor;

    let a = cos2pa / major2 + sin2pa / minor2;
    let b = -2.0 * cospa * sinpa * (1.0 / major2 - 1.0 / minor2);
    let c = sin2pa / major2 + cos2pa / minor2;

    let x2 = x_scale * x_scale;
    let y2 = y_scale * y_scale;

    let r = a / x2;
    let s = b * b / (4.0 * x2 * y2);
    let t = c / y2;

    let udiff = r - t;
    let f1 = udiff * udiff + 4.0 * s;
    let f2 = f1.sqrt() * udiff.abs();

    let j1 = (f2 + f1) / (2.0 * f1);
    let j2 = (f1 - f2) / (2.0 * f1);

    let k1 = (j1 * (r + t) - t) / (2.0 * j1 - 1.0);
    let k2 = (j2 * (r + t) - t) / (2.0 * j2 - 1.0);

    let c1 = 1.0 / k1.sqrt();
    let c2 = 1.0 / k2.sqrt();

    let pa_sign = if pa >= 0.0 { 1.0 } else { -1.0 };

    if (c1 - c2).abs() < f64::EPSILON {
        (1.0 / c1, 1.0 / c1, 0.0)
    } else if c1 > c2 {
        (c1, c2, pa_sign * j1.sqrt().acos())
    } else {
        (c2, c1, pa_sign * j2.sqrt().acos())
    }
}

// ── Khachiyan MVE algorithm ───────────────────────────────────────────────────

/// Find common beam using the minimum-volume-enclosing-ellipsoid of sampled beam
/// boundary points (Khachiyan algorithm). Matches `radio_beam` `method='pts'`.
pub fn common_manybeams_mve(
    beams: &[Beam],
    tolerance: f64,
    nsamps: usize,
    epsilon: f64,
) -> Result<Beam, CommonBeamError> {
    let max_iter = 10;
    let max_epsilon = 1e-3_f64;
    let mut eps = epsilon;

    for step in 0..=max_iter {
        let all_pts = collect_ellipse_points(beams, nsamps, eps);
        let hull_pts = convex_hull_2d(&all_pts);

        let (radii, rotation) = min_vol_ellipse(&hull_pts, tolerance)?;

        // Rotation matrix convention from radio_beam:
        // ((sin θ, cos θ), (cos θ, -sin θ))
        let pa = (-rotation[0][0]).atan2(rotation[1][0]);
        let pa = if pa == -std::f64::consts::PI || pa == std::f64::consts::PI {
            0.0
        } else {
            pa
        };

        let r0 = radii[0];
        let r1 = radii[1];
        let (major_deg, minor_deg) = if r0 >= r1 { (r0, r1) } else { (r1, r0) };

        let com_beam = Beam::new(major_deg, minor_deg, pa.to_degrees())?;

        if fits_in_beam(beams, &com_beam) {
            return Ok(com_beam);
        }

        if step == max_iter {
            return Err(CommonBeamError::DeconvFailed(format!(
                "epsilon reached {eps:.2e} without finding valid solution"
            )));
        }

        eps += (step as f64 + 1.0) * (max_epsilon - eps) / max_iter as f64;
    }

    unreachable!()
}

/// Sample `nsamps` points on the edge of each beam's ellipse (scaled by `1 + epsilon`).
fn collect_ellipse_points(beams: &[Beam], nsamps: usize, epsilon: f64) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity(beams.len() * nsamps);
    for beam in beams {
        let bpa = beam.pa_deg * DEG2RAD;
        let major = beam.major_deg * (1.0 + epsilon);
        let minor = beam.minor_deg * (1.0 + epsilon);
        for k in 0..nsamps {
            let phi = 2.0 * std::f64::consts::PI * k as f64 / nsamps as f64;
            let x = major * phi.cos();
            let y = minor * phi.sin();
            let xr = x * bpa.cos() - y * bpa.sin();
            let yr = x * bpa.sin() + y * bpa.cos();
            pts.push([xr, yr]);
        }
    }
    pts
}

/// 2D convex hull via Graham scan. Returns hull vertices.
fn convex_hull_2d(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if pts.len() <= 3 {
        return pts.to_vec();
    }

    // Find lowest-then-leftmost point.
    let pivot = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a[1].partial_cmp(&b[1])
                .unwrap()
                .then(a[0].partial_cmp(&b[0]).unwrap())
        })
        .map(|(i, _)| i)
        .unwrap();

    let pivot_pt = pts[pivot];

    let mut sorted: Vec<[f64; 2]> = pts.iter().filter(|&&p| p != pivot_pt).cloned().collect();

    sorted.sort_by(|a, b| {
        let angle_a = (a[1] - pivot_pt[1]).atan2(a[0] - pivot_pt[0]);
        let angle_b = (b[1] - pivot_pt[1]).atan2(b[0] - pivot_pt[0]);
        angle_a.partial_cmp(&angle_b).unwrap()
    });

    let mut hull: Vec<[f64; 2]> = vec![pivot_pt];
    for &p in &sorted {
        while hull.len() > 1 {
            let n = hull.len();
            let cross = cross2d(hull[n - 2], hull[n - 1], p);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

fn cross2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Khachiyan algorithm: minimum-volume enclosing ellipsoid of 2D points.
///
/// Returns `(radii, rotation_2x2)` where `radii` are the semi-axes of the ellipse
/// and `rotation` is the 2×2 matrix whose columns are the eigenvectors.
///
/// Port of `radio_beam.commonbeam.getMinVolEllipse`.
fn min_vol_ellipse(
    pts: &[[f64; 2]],
    tolerance: f64,
) -> Result<([f64; 2], [[f64; 2]; 2]), CommonBeamError> {
    let n = pts.len();
    let d = 2_usize;

    // Build Q = [[pts.T], [1...1]] shape (3, N)
    let q: Vec<[f64; 3]> = pts.iter().map(|p| [p[0], p[1], 1.0]).collect(); // (N, 3)

    let mut u = vec![1.0 / n as f64; n];

    let max_iter = 100_000;
    let mut err = 1.0_f64;
    let mut iter = 0;

    while err > tolerance {
        // V = Q.T * diag(u) * Q  (3x3 matrix)
        let v = matmul_qt_diag_q(&q, &u); // 3x3
        let v_inv = mat3_inv(v)?;

        // M[i] = q[i].T * V_inv * q[i]
        let m: Vec<f64> = q.iter().map(|qi| quadratic_form_3(qi, &v_inv)).collect();

        let j = m
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let maximum = m[j];

        let step = (maximum - d as f64 - 1.0) / ((d as f64 + 1.0) * (maximum - 1.0));
        let new_u: Vec<f64> = u.iter().map(|&ui| (1.0 - step) * ui).collect();
        err = new_u
            .iter()
            .zip(u.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        u = new_u;
        u[j] += step;

        iter += 1;
        if iter >= max_iter {
            return Err(CommonBeamError::NoConvergence(max_iter));
        }
    }

    // Center of ellipse
    let center = [
        pts.iter()
            .zip(u.iter())
            .map(|(p, &ui)| p[0] * ui)
            .sum::<f64>(),
        pts.iter()
            .zip(u.iter())
            .map(|(p, &ui)| p[1] * ui)
            .sum::<f64>(),
    ];

    // A = inv(P.T * diag(u) * P - center*center.T) / d  (2x2)
    let ptdp = matmul_pt_diag_p(pts, &u); // 2x2
    let cc = [
        [center[0] * center[0], center[0] * center[1]],
        [center[1] * center[0], center[1] * center[1]],
    ];
    let inner = [
        [ptdp[0][0] - cc[0][0], ptdp[0][1] - cc[0][1]],
        [ptdp[1][0] - cc[1][0], ptdp[1][1] - cc[1][1]],
    ];
    let a = mat2_scale(mat2_inv(inner)?, 1.0 / d as f64);

    // SVD of a (symmetric 2x2): A = U * diag(s) * V^T
    // For symmetric PSD A, eigendecomposition gives radii = 1/sqrt(eigenvalues)
    let (eigenvalues, eigenvectors) = symmetric_2x2_eig(a);

    let radii = [
        (1.0 / eigenvalues[0].abs().sqrt()) * (1.0 + tolerance),
        (1.0 / eigenvalues[1].abs().sqrt()) * (1.0 + tolerance),
    ];

    Ok((radii, eigenvectors))
}

// ── Small matrix helpers ──────────────────────────────────────────────────────

/// Q.T * diag(u) * Q where Q is (N, 3). Returns 3x3.
fn matmul_qt_diag_q(q: &[[f64; 3]], u: &[f64]) -> [[f64; 3]; 3] {
    let mut v = [[0.0_f64; 3]; 3];
    for (qi, &ui) in q.iter().zip(u.iter()) {
        for r in 0..3 {
            for c in 0..3 {
                v[r][c] += qi[r] * ui * qi[c];
            }
        }
    }
    v
}

/// P.T * diag(u) * P where P is (N, 2). Returns 2x2.
fn matmul_pt_diag_p(p: &[[f64; 2]], u: &[f64]) -> [[f64; 2]; 2] {
    let mut m = [[0.0_f64; 2]; 2];
    for (pi, &ui) in p.iter().zip(u.iter()) {
        for r in 0..2 {
            for c in 0..2 {
                m[r][c] += pi[r] * ui * pi[c];
            }
        }
    }
    m
}

/// Quadratic form x.T * M * x for 3-vector x and 3x3 matrix M.
fn quadratic_form_3(x: &[f64; 3], m: &[[f64; 3]; 3]) -> f64 {
    let mut acc = 0.0_f64;
    for r in 0..3 {
        for c in 0..3 {
            acc += x[r] * m[r][c] * x[c];
        }
    }
    acc
}

/// Invert a 3x3 matrix (Cramer's rule).
fn mat3_inv(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], CommonBeamError> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < f64::EPSILON {
        return Err(CommonBeamError::DeconvFailed("singular 3x3 matrix".into()));
    }
    let inv_det = 1.0 / det;

    Ok([
        [
            inv_det * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
            inv_det * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
            inv_det * (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        ],
        [
            inv_det * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
            inv_det * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
            inv_det * (m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        ],
        [
            inv_det * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
            inv_det * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
            inv_det * (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
        ],
    ])
}

/// Invert a symmetric 2x2 matrix.
fn mat2_inv(m: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], CommonBeamError> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < f64::EPSILON {
        return Err(CommonBeamError::DeconvFailed("singular 2x2 matrix".into()));
    }
    let inv_det = 1.0 / det;
    Ok([
        [inv_det * m[1][1], -inv_det * m[0][1]],
        [-inv_det * m[1][0], inv_det * m[0][0]],
    ])
}

fn mat2_scale(m: [[f64; 2]; 2], s: f64) -> [[f64; 2]; 2] {
    [[m[0][0] * s, m[0][1] * s], [m[1][0] * s, m[1][1] * s]]
}

/// Eigendecomposition of a symmetric 2x2 matrix. Returns `(eigenvalues, eigenvectors)`.
/// `eigenvectors[i]` is the i-th column.
fn symmetric_2x2_eig(m: [[f64; 2]; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let a = m[0][0];
    let b = m[0][1]; // = m[1][0] since symmetric
    let c = m[1][1];

    let trace = a + c;
    let det = a * c - b * b;
    let disc = ((trace / 2.0).powi(2) - det).max(0.0).sqrt();

    let lam1 = trace / 2.0 + disc;
    let lam2 = trace / 2.0 - disc;

    // Eigenvectors
    let (v1, v2) = if b.abs() > f64::EPSILON {
        let v1 = normalise([lam1 - c, b]);
        let v2 = normalise([lam2 - c, b]);
        (v1, v2)
    } else if a >= c {
        ([1.0, 0.0], [0.0, 1.0])
    } else {
        ([0.0, 1.0], [1.0, 0.0])
    };

    // Rotation matrix columns are eigenvectors.
    // radio_beam convention: SVD gives rotation such that
    //   PA = atan2(-rotation[0,0], rotation[1,0])
    // Match that: columns = eigenvectors, so rotation[row][col].
    let rotation = [[v1[0], v2[0]], [v1[1], v2[1]]];

    ([lam1, lam2], rotation)
}

fn normalise(v: [f64; 2]) -> [f64; 2] {
    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if len < f64::EPSILON {
        return v;
    }
    [v[0] / len, v[1] / len]
}

// ── Utility ───────────────────────────────────────────────────────────────────

pub fn largest_beam(beams: &[Beam]) -> Beam {
    beams
        .iter()
        .max_by(|a, b| a.area_sr().partial_cmp(&b.area_sr()).unwrap())
        .copied()
        .unwrap()
}

/// True if all beams can be deconvolved from `large_beam`.
pub fn fits_in_beam(beams: &[Beam], large_beam: &Beam) -> bool {
    beams.iter().all(|b| {
        if b.approx_eq(large_beam) {
            return true;
        }
        let result = deconvolve_deg(
            large_beam.major_deg,
            large_beam.minor_deg,
            large_beam.pa_deg,
            b.major_deg,
            b.minor_deg,
            b.pa_deg,
            true,
        );
        match result {
            Ok((maj, min, _)) => maj > 0.0 && min > 0.0,
            Err(_) => false,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_beam_identical() {
        let b = Beam::new(10.0 / 3600.0, 8.0 / 3600.0, 30.0).unwrap();
        let result = common_beam(&[b, b], 1e-4, 200, 5e-4).unwrap();
        assert!(result.approx_eq(&b) || result.major_deg >= b.major_deg);
    }

    #[test]
    fn test_common_beam_two_different() {
        let b1 = Beam::new(10.0 / 3600.0, 8.0 / 3600.0, 30.0).unwrap();
        let b2 = Beam::new(12.0 / 3600.0, 6.0 / 3600.0, 60.0).unwrap();
        let result = common_beam(&[b1, b2], 1e-4, 200, 5e-4).unwrap();
        assert!(
            result.major_deg >= b1.major_deg.max(b2.major_deg) || fits_in_beam(&[b1, b2], &result)
        );
    }
}
