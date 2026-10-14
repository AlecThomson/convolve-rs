# Decision: no GPU (cuFFT) backend for now

2026-06-15. Not part of the published docs.

## Measurements

`scripts/bench_gpu_vs_cpu.py` (numpy, convolve-rs and CuPy/cuFFT, with and
without host-device transfers) and `scripts/profile_cube.sh` (the release
binary end to end). NVIDIA A30, 96 logical CPUs, `float32` cubes.

| Image  | End to end          | GPU with transfer  | GPU compute only    |
| ------ | ------------------- | ------------------ | ------------------- |
| 1024²  | 83.6 ch/s (12.0 ms) | 844 ch/s (1.18 ms) | 9033 ch/s (0.11 ms) |
| 2048²  | not run             | 339 ch/s (2.95 ms) | 1896 ch/s (0.53 ms) |
| 10240² | 0.12 ch/s (8406 ms) | 3.6 ch/s (277 ms)  | 71.6 ch/s (14 ms)   |

## Conclusion

The FFT is about 0.2 % of the end-to-end time at 10240² and 1 % at 1024²; the
rest is FITS I/O through the single writer thread, which kept the CPU at about
two busy cores. The GPU speed-up with transfers also shrinks for large planes,
because each 419 MB plane crosses PCIe twice. Revisit only if a workload is
shown to be FFT-bound and its data can stay on the GPU across many operations.

## I/O issues found

- One writer thread byte-swaps and writes everything (26.8 GB at 10240² × 64).
- The streaming queue holds `2 × threads` planes, not a byte budget: with 96
  threads and 64 channels it never fills, and peak RSS reached 113 GB for a
  27 GB cube. A byte budget (`max(4, budget / plane_bytes)`) would fix that.

## Reproduction

```sh
uv pip install --python .venv/bin/python "cupy-cuda12x[ctk]"
uv sync --extra benchmark --inexact
.venv/bin/python scripts/bench_gpu_vs_cpu.py --nx 2048 --ny 2048 --nchan 64
PYTHON=.venv/bin/python scripts/profile_cube.sh 10240 10240 64 float32 total
```
