# Performance & precision

## Native-precision convolution

The convolution runs in the **native precision of the input data**, chosen
automatically from the FITS `BITPIX`:

- `BITPIX = -32` (`float32`, the common radio-astronomy case) — transformed in
  `f32`, about half the memory traffic and compute of a double-precision
  transform.
- `BITPIX = -64` (`float64`) — transformed in `f64`, so no precision is lost.

No flags are needed. The Python {func}`convolve_rs.smooth` likewise accepts
`float32` or `float64` arrays and returns the same dtype it was given. Earlier
releases always computed in `f64` (and read everything as `f32`), so `float32`
data now convolves roughly 1.5× faster, while genuine `float64` cubes are
honoured exactly instead of being truncated to `f32` on read.

## Cube streaming pipeline

Cubes are processed channel-by-channel through a bounded streaming pipeline
rather than materialising the whole cube in memory:

- [rayon](https://docs.rs/rayon) convolves planes in parallel across all CPU
  cores.
- A single writer thread streams finished planes to disk, because cfitsio is
  not thread-safe.
- A bounded channel between them overlaps convolution with disk I/O and caps
  peak memory to the in-flight planes, not the whole output cube.

The FFT plans depend only on the image dimensions, so they are built once per
cube and shared across every channel instead of being re-planned per channel.

For large cubes the bottleneck is usually FITS I/O (the single writer thread)
rather than FFT compute — worth keeping in mind when reasoning about wall-clock.

## Versus RACS-tools `beamcon_2D`

This crate is a Rust port of the Python `beamcon_2D` from
[RACS-tools](https://github.com/AlecThomson/RACS-tools), and produces the same
result: smoothed pixels match `beamcon_2D` to float32 round-off (max relative
difference ~1.6e-7 on a 2048² image), so the speedup is not bought with reduced
accuracy.

Batch wall-clock, smoothing N synthetic images to a common 20″ beam, median of
3 fresh runs on an Apple M4 Pro (12 cores, 24 GB). `convolvers 2d` versus
`beamcon_2D --executor process` (RACS-tools 4.3.1, the fastest beamcon
executor here):

| Workload  | convolvers | beamcon_2D | speedup |
| --------- | ---------- | ---------- | ------- |
| 4 × 2048² | 0.20 s     | 4.09 s     | 20×     |
| 4 × 4096² | 0.86 s     | 5.50 s     | 6.4×    |
| 8 × 4096² | 1.94 s     | 6.65 s     | 3.4×    |
| 4 × 8192² | 5.03 s     | 18.2 s     | 3.6×    |

convolvers is faster at every size. The ratio is largest on small batches,
where beamcon's fixed Python startup dominates, and settles to **~3.5× on the
large, compute-bound end** where that overhead is amortised and the FFT work
dominates. The table uses warm runs for beamcon.

beamcon also pays a one-time **cold-start** penalty that convolvers, a native
binary, simply does not have. The first invocation on a cold environment must
import and byte-compile the whole Python dependency tree (astropy, scipy, numba,
llvmlite, …) and JIT-compile beamcon's numba kernel before any pixels move:

| 4 × 2048², beamcon_2D                     | wall    |
| ----------------------------------------- | ------- |
| cold (fresh `__pycache__` + numba cache)  | ~11 s   |
| warm (caches populated)                   | ~4 s    |

A truly fresh install (cold OS file cache too) is slower still — ~35 s on first
run here. The numba kernel cache itself is a small part of this (~0.4 s); the
bulk is Python import and byte-compilation. A long-lived process pays it once;
batch jobs that re-launch the interpreter per call pay it every time. convolvers
starts in milliseconds regardless.

### Peak memory

Smaller too. Peak resident set, summed across the whole process tree (beamcon's
`process` executor spawns one worker per core, each loading its own Python +
numpy + astropy, so a parent-only `time -l` reading badly under-counts it):

| Workload  | convolvers | beamcon_2D | beamcon / convolvers |
| --------- | ---------- | ---------- | -------------------- |
| 4 × 2048² | 0.36 GB    | 2.6 GB     | 7.4×                 |
| 4 × 4096² | 1.4 GB     | 5.5 GB     | 3.9×                 |
| 4 × 8192² | 5.0 GB     | 10.6 GB    | 2.1×                 |

The absolute gap widens with size: at 8192² convolvers needs ~5 GB where beamcon
needs ~10.6 GB — the difference between fitting and swapping on a 24 GB machine.
convolvers shares one address space across rayon threads and keeps only a
half-spectrum per plane (see below); beamcon pays a full interpreter per worker.

### Where the speed comes from

A single native binary with no per-process
interpreter or JIT startup, native-`f32` transforms on `float32` data (half the
traffic of beamcon's `f64` path), a real-input FFT that keeps only the
half-spectrum (so an 8192² plane needs ~2.4 GB instead of ~6 GB and four planes
fit in RAM without swapping), FFT plans built once and shared across channels,
and a streaming pipeline that overlaps convolution with disk I/O.

Reproduce with the committed harness (set `BEAMCON_BIN` to a `beamcon_2D` whose
env has `numpy < 2.2`, required by its numba):

```sh
scripts/compare_beamcon.sh 4 4096 3      # N=4 images, 4096², median of 3 runs
```

## Profiling and benchmarks

Two committed tools measure this (neither is part of the published package):

End-to-end throughput on a synthetic cube:

```sh
scripts/profile_cube.sh 2048 2048 64 float32 total   # NX NY NCHAN DTYPE MODE
```

Microbenchmarks of the convolution itself (image size × precision × clean vs
NaN-masked), via [criterion](https://docs.rs/criterion):

```sh
cargo bench
```

Indicative single-convolution times (one plane, single-threaded; hardware- and
sample-count-dependent — treat as ballpark):

| Image  | f32 (clean) | f64 (clean) | f32 speedup |
| ------ | ----------- | ----------- | ----------- |
| 512²   | 4.6 ms      | 7.0 ms      | ~1.5×       |
| 1024²  | 22.8 ms     | 37.2 ms     | ~1.6×       |
| 2048²  | 126 ms      | 191 ms      | ~1.5×       |
| 4096²  | 609 ms      | 898 ms      | ~1.5×       |

The NaN-masked path costs roughly 1.6–1.9× the clean path, because it runs a
second FFT pair to propagate the blanking mask.

## Why there is no GPU acceleration

An FFT runs faster on a GPU in a microbenchmark, but that does not speed up a
real cube. The work is dominated by FITS I/O: reading planes from disk and
writing them back, not the FFT. The FFT is a small share of the total time, and
a smaller share the larger the images get. Running it on a GPU also means copying
every plane over PCIe, so the end-to-end time barely moves.

The package is therefore CPU-only, with no CUDA toolkit to match or drivers to
manage, and it installs the same way everywhere (including Apple Silicon). A GPU
backend would only pay off for work that is actually FFT-bound, such as keeping
data on the GPU across many operations. Use the profiling tools above to see
where the time goes for your data.
