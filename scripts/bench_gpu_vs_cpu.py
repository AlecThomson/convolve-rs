#!/usr/bin/env python3
"""Compare GPU vs CPU for the UV-plane convolution, to decide whether a native
cuFFT backend is worth building.

The convolution (forward real FFT -> multiply by the analytic Gaussian UV filter
-> inverse real FFT) is written once, parameterised by the array module, and run
on:

  * convolve-rs (Rust, CPU): the production path, via the Python binding, if
    ``convolve_rs`` is importable. This is the absolute number a GPU backend
    must beat. Use a RELEASE build (``pip install convolve-rs``); a
    ``maturin develop`` debug build is ~20x slower and not representative.
  * NumPy (CPU): the same algorithm as the GPU path in pure Python. This is the
    apples-to-apples CPU baseline for the GPU comparison (identical code, same
    FFT library family), independent of the Rust build profile.
  * CuPy (GPU, cuFFT): the same algorithm on the GPU, timed both with and
    without host/device transfers. The "with transfer" figure is the realistic
    one for the streaming pipeline (planes arrive from disk in host memory); the
    "compute only" figure is the optimistic on-device ceiling.

Run it later on a GPU server::

    pip install cupy-cuda12x          # match your CUDA toolkit
    pip install convolve-rs            # for the production CPU reference
    python scripts/bench_gpu_vs_cpu.py --nx 2048 --ny 2048 --nchan 64

It degrades gracefully: if CuPy or a GPU is missing it reports the CPU numbers
and says so, rather than failing.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np


def gauss_uv_filter(xp, ny, nx, pix_deg, old_as, new_as):
    """Analytic Gaussian UV filter on the rfft2 half-spectrum (shape ny by nx//2+1).

    Mirrors ``racs_tools.gaussft`` / ``convolve_uv::gaussft``: deconvolve the old
    beam and re-convolve the new one. ``old_as``/``new_as`` are
    ``(bmaj, bmin, bpa)`` in arcsec/arcsec/deg.
    """
    deg2rad = math.pi / 180.0
    fwhm_to_sigma = 2.0 * math.sqrt(2.0 * math.log(2.0))

    def sigmas(beam_as):
        bmaj, bmin, bpa = beam_as
        return (
            (bmaj / 3600.0 * deg2rad) / fwhm_to_sigma,
            (bmin / 3600.0 * deg2rad) / fwhm_to_sigma,
            bpa * deg2rad,
        )

    sx, sy, bpa = sigmas(new_as)
    sx_in, sy_in, bpa_in = sigmas(old_as)
    g_ratio = math.sqrt(sx * sy) / math.sqrt(sx_in * sy_in)

    d_rad = pix_deg * deg2rad
    u = xp.fft.fftfreq(ny, d=d_rad)[:, None]  # rows (full)
    v = xp.fft.rfftfreq(nx, d=d_rad)[None, :]  # cols (half spectrum)

    ur = u * math.cos(bpa) - v * math.sin(bpa)
    vr = u * math.sin(bpa) + v * math.cos(bpa)
    ur_in = u * math.cos(bpa_in) - v * math.sin(bpa_in)
    vr_in = u * math.sin(bpa_in) + v * math.cos(bpa_in)

    pi2 = math.pi * math.pi
    g_arg = -2.0 * pi2 * ((sx * ur) ** 2 + (sy * vr) ** 2)
    dg_arg = -2.0 * pi2 * ((sx_in * ur_in) ** 2 + (sy_in * vr_in) ** 2)
    return (g_ratio * xp.exp(g_arg - dg_arg)).astype(xp.float64)


def convolve_block(xp, planes, filt, ny, nx, transfer):
    """Convolve every plane with `filt` using array module `xp`.

    When `transfer` is True each plane is copied host->device and the result
    device->host, modelling the streaming pipeline; when False the data is
    assumed already resident (optimistic on-device ceiling).
    """
    for plane in planes:
        arr = xp.asarray(plane) if transfer else plane
        spec = xp.fft.rfft2(arr)
        spec *= filt
        out = xp.fft.irfft2(spec, s=(ny, nx))
        if transfer and xp.__name__ == "cupy":
            _ = xp.asnumpy(out)


def timeit(fn, runs):
    """Best-of-`runs` wall time (seconds) after one warm-up call."""
    fn()  # warm up (cuFFT plan creation, allocator priming)
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def report(name, secs, nchan, ref):
    per = secs / nchan * 1e3
    chps = nchan / secs
    speed = f"{ref / secs:5.2f}x" if ref else "  ref"
    print(f"  {name:<28} {per:8.2f} ms   {chps:8.1f} ch/s   {speed}")
    return secs


def import_optional(name):
    """Import a module by name, returning None if it is not installed."""
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def run_gpu(cp, planes, filt_np, ny, nx, nchan, ref, runs):
    """Run and report the GPU paths; print a skip line on any CUDA error."""
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            print("\n  GPU path skipped: no CUDA device found")
            return
        dev = cp.cuda.Device()
        name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
        print(f"\n  GPU: {name}")

        filt_cp = cp.asarray(filt_np)

        def with_transfer():
            convolve_block(cp, planes, filt_cp, ny, nx, transfer=True)
            cp.cuda.Device().synchronize()

        report("cupy FFT (GPU, with transfer)", timeit(with_transfer, runs), nchan, ref)

        resident = [cp.asarray(pl) for pl in planes]  # pre-loaded on device

        def compute_only():
            convolve_block(cp, resident, filt_cp, ny, nx, transfer=False)
            cp.cuda.Device().synchronize()

        report("cupy FFT (GPU, compute only)", timeit(compute_only, runs), nchan, ref)
    except Exception as e:  # noqa: BLE001  (cupy/CUDA raise many error types)
        print(f"\n  GPU path skipped: {e}")


def run_rust_reference(planes, pix_deg, old_as, new_as, nchan, runs):
    """Time the production Rust convolution if the binding is installed."""
    convolve_rs = import_optional("convolve_rs")
    if convolve_rs is None:
        print("  convolve-rs not installed - skipping production CPU reference")
        return None

    old = convolve_rs.Beam.from_arcsec(*old_as)
    new = convolve_rs.Beam.from_arcsec(*new_as)

    def run():
        for plane in planes:
            convolve_rs.smooth(plane, old, new, pix_deg, pix_deg, bunit="K")

    return report("convolve-rs (Rust, CPU)", timeit(run, runs), nchan, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=2048)
    p.add_argument("--ny", type=int, default=2048)
    p.add_argument("--nchan", type=int, default=64)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    ny, nx, nchan = args.ny, args.nx, args.nchan
    pix_deg = 2.5 / 3600.0
    old_as = (15.0, 12.0, 20.0)
    new_as = (25.0, 20.0, 20.0)

    rng = np.random.default_rng(0)
    planes = [rng.standard_normal((ny, nx)).astype(args.dtype) for _ in range(nchan)]

    print(f"Cube: {nchan} x {ny} x {nx} {args.dtype}  (best of {args.runs} runs)\n")
    print(f"  {'method':<28} {'per-plane':>9}   {'throughput':>10}   speedup")

    # Production CPU reference (real Rust path) and the algorithm-matched numpy
    # baseline that the GPU is compared against.
    ref = run_rust_reference(planes, pix_deg, old_as, new_as, nchan, args.runs)
    filt_np = gauss_uv_filter(np, ny, nx, pix_deg, old_as, new_as)
    cpu = report(
        "numpy FFT (CPU)",
        timeit(
            lambda: convolve_block(np, planes, filt_np, ny, nx, transfer=False),
            args.runs,
        ),
        nchan,
        ref,
    )
    ref = ref if ref is not None else cpu  # fall back to numpy for speedup ratios

    cp = import_optional("cupy")
    if cp is None:
        print("\n  GPU path skipped: cupy not installed (pip install cupy-cuda12x)")
    else:
        run_gpu(cp, planes, filt_np, ny, nx, nchan, ref, args.runs)

    print(
        "\nNotes: the apples-to-apples GPU question is cupy vs numpy (identical code).\n"
        "'with transfer' is the realistic streaming figure; 'compute only' is the\n"
        "on-device ceiling. The convolve-rs (Rust) line is the production target and\n"
        "is only meaningful from a RELEASE build (pip wheel), not maturin develop.\n"
        "A native cuFFT backend is only worth building if the GPU win survives the\n"
        "FITS-I/O bound of a full cube run (scripts/profile_cube.sh)."
    )


if __name__ == "__main__":
    main()
