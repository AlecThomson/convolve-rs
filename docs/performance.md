# Performance & precision

## Native-precision convolution

The convolution runs in the **native precision of the input data**, chosen
automatically from the FITS `BITPIX`:

- `BITPIX = -32` (`float32`, the common radio-astronomy case) — transformed in
  `f32`, about half the memory traffic and compute of a double-precision
  transform.
- `BITPIX = -64` (`float64`) — transformed in `f64`, so no precision is lost.

No flags are needed. The Python {func}`convolve_rs.smooth` likewise accepts
`float32` or `float64` arrays and returns the same precision it was given. Earlier
releases always computed in `f64` (and read everything as `f32`), so `float32`
data now convolves roughly 1.5× faster, while genuine `float64` cubes are
honoured exactly instead of being truncated to `f32` on read.

## Memory: about one image per plane

A plane is convolved in roughly **one image's worth of working memory** on top
of its input, and in none at all when the input can be consumed (as the CLI
does with each plane it reads):

- The image is real, so only the non-redundant half of its spectrum is kept,
  and the real FFT runs **in place**: the half spectrum of an `ny × nx` image
  fits in the image's own buffer plus two values per row.
- The UV-plane filter is evaluated on the fly during the column transforms and
  never stored. The Jy/beam or Kelvin flux factor is folded into it, so it costs
  no extra pass over the image either.
- A NaN-blanked image has its blanking mask convolved first, in a buffer that
  is freed before the image's own transform, leaving only a one-bit-per-pixel
  record of the pixels to blank.
- The column transforms are done a block of adjacent columns at a time, which
  reads whole cache lines rather than one value per line and is most of the
  speed-up on wide images.

From Python, {func}`convolve_rs.smooth` reads the numpy array in place (any
layout; big-endian FITS data is converted once), hands the result back without
a copy, and releases the GIL while convolving, so a thread pool smoothing
several planes runs them in parallel.

## Compared with racs_tools

RACS-tools' robust mode was itself reworked in 2026 to cut its peak memory
(real-to-complex FFTs in the image's own precision, a real-valued filter).
Measured against that version on one 4-core machine, one plane per call from
Python, with the working memory given as peak RSS above the input array in
units of the image size (re-measure with `scripts/bench_vs_racs_tools.py`, in an
environment with both packages installed):

| Plane (`float32`)       | racs_tools | convolve-rs | Speed-up |
| ----------------------- | ---------- | ----------- | -------- |
| 4500 × 3900             | 1.06 s, 3.8× | 0.27 s, 1.0× | 3.9×   |
| 4500 × 3900, 42 % NaN   | 1.70 s, 4.3× | 0.59 s, 1.1× | 2.9×   |
| 9000 × 7800             | 4.09 s, 3.5× | 1.17 s, 1.0× | 3.5×   |
| 9000 × 7800, 42 % NaN   | 7.14 s, 4.0× | 2.40 s, 1.0× | 3.0×   |

`float64` planes show the same pattern (3.4–4.7× faster, 1.0× vs 3.5–3.6×
memory). End to end, `convolvers 3d` on a 3000 × 2600 × 24 `float32` cube with
a blanked region took 2.1 s at a peak RSS of 0.56× the cube, against 12.5 s and
1.48× for `beamcon_3D` with four threads.

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

## Profiling and benchmarks

Three committed tools measure this (none is part of the published package):

End-to-end throughput on a synthetic cube:

```sh
scripts/profile_cube.sh 2048 2048 64 float32 total   # NX NY NCHAN DTYPE MODE
```

Microbenchmarks of the convolution itself (image size × precision × clean vs
NaN-masked), via [criterion](https://docs.rs/criterion):

```sh
cargo bench
```

Head-to-head against racs_tools' robust mode (time and peak memory per plane,
from Python; see [above](#compared-with-racs-tools)):

```sh
python scripts/bench_vs_racs_tools.py 4500x3900 9000x7800   # NYxNX ...
```

Indicative single-convolution times (one plane, single-threaded; hardware- and
sample-count-dependent — treat as ballpark):

| Image  | f32 (clean) | f64 (clean) | f32 (NaN-masked) |
| ------ | ----------- | ----------- | ---------------- |
| 512²   | 2.7 ms      | 3.5 ms      | 5.9 ms           |
| 1024²  | 10.5 ms     | 14.2 ms     | 23.2 ms          |
| 2048²  | 57 ms       | 72 ms       | 117 ms           |
| 4096²  | 251 ms      | 353 ms      | —                |

The NaN-masked path costs roughly twice the clean path, because it runs a
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
