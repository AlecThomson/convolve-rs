//! Microbenchmarks for the UV-plane convolution.
//!
//! Run with `cargo bench`. Three groups:
//!   * `convolve` — single-image `convolve_uv` across image size, precision
//!     (f32/f64) and the clean vs NaN-masked path.
//!   * `convolve_large` — the same at 4096² with a small sample count.
//!   * `cube_throughput` — convolving many same-size planes, comparing per-call
//!     planning (`convolve_uv`) against a shared `FftPlans`
//!     (`convolve_uv_with_plans`); this is the cube-channel hot loop and shows
//!     the plan-reuse win in channels/second.
//!
//! These quantify the precision (Tier 1) and plan-reuse (Tier 0) changes; the
//! end-to-end I/O ceiling (Tier 2) is measured separately by
//! `scripts/profile_cube.sh`.
use std::hint::black_box;

use convolve_rs::{Beam, FftFloat, FftPlans, convolve_uv, convolve_uv_with_plans};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use num_traits::cast;

/// A deterministic, mildly-structured test image in precision `T`, optionally
/// with a blanked (NaN) block to exercise the mask-convolution path.
fn make_image<T: FftFloat>(n: usize, masked: bool) -> Array2<T> {
    let mut img = Array2::from_shape_fn((n, n), |(i, j)| {
        cast::<f64, T>(((i * 13 + j * 7) % 97) as f64 / 97.0).unwrap()
    });
    if masked {
        let blk = (n / 16).max(2);
        for i in 0..blk {
            for j in 0..blk {
                img[(i, j)] = T::nan();
            }
        }
    }
    img
}

fn beams() -> (Beam, Beam) {
    (
        Beam::from_arcsec(10.0, 8.0, 20.0).unwrap(),
        Beam::from_arcsec(16.0, 13.0, 20.0).unwrap(),
    )
}

const DX: f64 = 2.5 / 3600.0;

fn run_one<T: FftFloat>(img: &Array2<T>, old: &Beam, new: &Beam) {
    let res = convolve_uv(img, old, new, DX, DX, None).unwrap();
    black_box(res.image[(0, 0)]);
}

fn bench_convolve(c: &mut Criterion) {
    let (old, new) = beams();
    let mut group = c.benchmark_group("convolve");
    for &n in &[512usize, 1024, 2048] {
        for &masked in &[false, true] {
            let tag = if masked { "masked" } else { "clean" };

            let img32 = make_image::<f32>(n, masked);
            group.bench_with_input(
                BenchmarkId::new(format!("f32/{tag}"), n),
                &img32,
                |b, img| b.iter(|| run_one(img, &old, &new)),
            );

            let img64 = make_image::<f64>(n, masked);
            group.bench_with_input(
                BenchmarkId::new(format!("f64/{tag}"), n),
                &img64,
                |b, img| b.iter(|| run_one(img, &old, &new)),
            );
        }
    }
    group.finish();
}

fn bench_convolve_large(c: &mut Criterion) {
    let (old, new) = beams();
    let mut group = c.benchmark_group("convolve_large");
    group.sample_size(10);
    let n = 4096usize;
    let img32 = make_image::<f32>(n, false);
    group.bench_function(BenchmarkId::new("f32/clean", n), |b| {
        b.iter(|| run_one(&img32, &old, &new))
    });
    let img64 = make_image::<f64>(n, false);
    group.bench_function(BenchmarkId::new("f64/clean", n), |b| {
        b.iter(|| run_one(&img64, &old, &new))
    });
    group.finish();
}

/// Convolve `nchan` same-size planes, the cube-channel hot loop. Compares
/// re-planning every call against reusing one `FftPlans`.
fn bench_cube_throughput(c: &mut Criterion) {
    let (old, new) = beams();
    let (n, nchan) = (512usize, 64u64);
    let planes: Vec<Array2<f32>> = (0..nchan).map(|_| make_image::<f32>(n, false)).collect();

    let mut group = c.benchmark_group("cube_throughput");
    group.throughput(Throughput::Elements(nchan));

    group.bench_function("per_call_planning", |b| {
        b.iter(|| {
            for p in &planes {
                let r = convolve_uv(p, &old, &new, DX, DX, None).unwrap();
                black_box(r.image[(0, 0)]);
            }
        })
    });

    group.bench_function("shared_plans", |b| {
        let plans = FftPlans::<f32>::new(n, n);
        b.iter(|| {
            for p in &planes {
                let r = convolve_uv_with_plans(p, &old, &new, DX, DX, None, &plans).unwrap();
                black_box(r.image[(0, 0)]);
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_convolve,
    bench_convolve_large,
    bench_cube_throughput
);
criterion_main!(benches);
