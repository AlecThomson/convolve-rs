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
