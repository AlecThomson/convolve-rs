# Decision: defer the GPU (cuFFT) backend — the pipeline is FITS-I/O-bound

**Date:** 2026-06-15
**Status:** Decided — do **not** build the GPU backend yet. Revisit only under the
conditions below.
**Scope:** This is an engineering decision record for the GPU-acceleration
exploration. It is intentionally kept out of the published Sphinx docs
(`docs/`), which describe the shipped package only.

## Question

Is an optional, feature-gated cuFFT backend worth building for the UV-plane
convolution? The convolution is forward real FFT → multiply by the analytic
Gaussian UV filter → inverse real FFT, run per channel.

## Method

- `scripts/bench_gpu_vs_cpu.py` — same algorithm on numpy (CPU), the
  production `convolve_rs` path (CPU), and CuPy/cuFFT (GPU), timing the GPU
  both with and without host↔device transfers.
- `scripts/profile_cube.sh` — the **real** production pipeline end to end
  (release `convolvers` binary: read FITS → convolve across cores → write
  FITS), reporting wall-clock and channels/s including all FITS I/O.

Hardware: NVIDIA A30 (24 GB), 96 logical CPUs, 188 GB RAM, results on
`float32` cubes (the common radio-astronomy case).

## Results

Per-plane compute (best of N), and the end-to-end pipeline throughput:

| Image  | End-to-end pipeline (read→convolve→write) | GPU, with transfer | GPU, compute-only |
| ------ | ----------------------------------------- | ------------------ | ----------------- |
| 1024²  | 83.6 ch/s (12.0 ms/ch)                    | 844 ch/s (1.18 ms) | 9033 ch/s (0.11 ms) |
| 2048²  | — (not run end-to-end)                    | 339 ch/s (2.95 ms) | 1896 ch/s (0.53 ms) |
| 10240² | **0.12 ch/s (8406 ms/ch)**                | 3.6 ch/s (277 ms)  | 71.6 ch/s (14 ms)   |

(10240² end-to-end: 538 s wall for 64 channels, release build, CPU-only.)

## Conclusion: no

FFT compute is a tiny fraction of end-to-end wall-clock at **both** scales, so a
cuFFT backend would accelerate something that is not the bottleneck (Amdahl):

- **10240²**: the entire 64-plane FFT compute is ~0.9 s (GPU compute-only
  ceiling), against a **538 s** end-to-end wall → FFT is **~0.17 %** of the run.
- **1024²**: FFT compute (~0.11 ms/plane) is **~0.9 %** of the 12.0 ms/channel
  pipeline cost.
- The pipeline can't even saturate the *CPU* compute headroom: at 10240² the
  binary ran at only ~206 % CPU (≈2 cores busy), because the rayon convolvers
  sit idle waiting on the single cfitsio writer thread.
- The realistic GPU "with transfer" speedup also *shrinks* as planes grow
  (13.9× → 27.7× → 10.3× at 1024²/2048²/10240²): moving a 419 MB plane each way
  over PCIe becomes its own bottleneck exactly when compute would start to
  matter.

The "compute-only" GPU figure (up to 204×) is the seductive but misleading
number — it assumes data stays resident on the GPU, which it does not in a
streaming, one-FFT-pair-per-plane, write-to-disk workload.

## Revisit the GPU only if BOTH hold

1. A workload is shown to be **FFT-bound** (not I/O-bound) — e.g. after the
   FITS-write bottleneck below is fixed, or for many-channel small-image cubes.
2. Data stays **GPU-resident across many operations**, so transfers amortise
   (e.g. a future all-on-GPU multi-step pipeline), rather than one FFT pair per
   plane.

## The real lever: FITS I/O (the single-writer pipeline)

The streaming pipeline in `src/main.rs::process_cube` convolves planes in
parallel (rayon) and funnels them to **one** writer thread (cfitsio is not
thread-safe on a single handle). Two concrete issues found:

1. **Single-threaded writer is the wall.** All output (26.8 GB at 10240²×64)
   goes through one cfitsio `write_section`. FITS pixels are big-endian on disk,
   so every value is byte-swapped in that one thread — likely CPU-bound, not
   just disk-bound. This sets the ~0.12 ch/s ceiling.

2. **The "bounded" channel does not bound memory.** `cap = 2 ×
   rayon::current_num_threads()` (= 192 on this 96-core box) counts *planes*,
   not bytes. With `nfreq (64) < cap (192)` the channel never backpressures, so
   all 64 convolved planes (26.8 GB) buffer in RAM while the writer drains them;
   add ~96 threads' transient work buffers (read + complex spectrum + output per
   plane) and peak RSS hit **113 GB** for a 27 GB cube.

Suggested directions (not yet implemented):
- Make the channel cap a **byte budget** (`max(4, mem_budget / plane_bytes)`),
  so large planes don't buffer the whole cube.
- Speed up the write path: parallel byte-swap in the producers (hand the writer
  ready-to-write buffers), or disjoint-range `pwrite` from multiple handles,
  or chunked/larger write buffers.
- Re-profile after: if the pipeline becomes FFT-bound, re-open the GPU question.

## Reproduction

The GPU bench needs CuPy *and* the CUDA-12 runtime libraries. On this box the
`.venv` (uv) had `cupy-cuda12x` but no cuFFT; install the toolkit libs via the
`[ctk]` extra (cuda-pathfinder then locates them — no `LD_LIBRARY_PATH` needed):

```sh
uv pip install --python .venv/bin/python "cupy-cuda12x[ctk]"
uv sync --extra benchmark --inexact          # astropy etc. for the cube tools

.venv/bin/python scripts/bench_gpu_vs_cpu.py --nx 2048 --ny 2048 --nchan 64
PYTHON=.venv/bin/python scripts/profile_cube.sh 10240 10240 64 float32 total
```
