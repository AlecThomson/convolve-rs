# Performance and precision

## Precision

Each plane is transformed in the precision of its data, chosen from the FITS
`BITPIX`:

- `BITPIX = -32`: `f32`. This is the common case and moves half the memory of
  an `f64` transform.
- `BITPIX = -64`: `f64`.
- Integer images: `f32`, written out as `BITPIX = -32`.

The Python {func}`convolve_rs.smooth` takes `float32` or `float64` arrays and
returns the same precision.

## Memory

A plane needs about one image of working memory on top of its input, and none
when the input can be reused, as the CLI does with each plane it reads:

- Only the non-redundant half of the spectrum is kept, and the real FFT runs
  in the image's own buffer plus two values per row.
- The UV filter is evaluated during the column transforms and never stored.
  The Jy/beam or Kelvin flux factor is folded into it.
- A NaN mask is convolved first, in a buffer freed before the image's
  transform, leaving one bit per pixel to blank.

From Python, {func}`convolve_rs.smooth` reads the numpy array without copying
(big-endian FITS data is converted once) and releases the GIL while it works.

## Parallelism

A single plane is split across threads: row FFTs run in parallel, and the
column pass works on blocks of 16 columns in parallel. Cubes run channels in
parallel. Each channel does its row passes on its own thread, and idle threads
can take its column blocks. Finished planes go to one writer thread through a
bounded queue, so memory is capped at the planes in flight. The 2D CLI runs
images in parallel when there are at least as many as threads, and one at a
time otherwise.

## Timings

Measured on an Intel Xeon at 2.10 GHz (4 vCPUs, KVM), 16 GB RAM, Linux, ext4
disk.

`cargo bench`, one plane:

| Image | f32    | f64    | f32 with NaNs |
| ----- | ------ | ------ | ------------- |
| 2048² | 26 ms  | 48 ms  | 48 ms         |
| 4096² | 144 ms | 222 ms | not benched   |

(compared-with-racs-tools)=

### Compared with racs_tools

One plane per call from Python, from `scripts/bench_vs_racs_tools.py`, one run
per case. racs_tools 4.4.1 in `robust` mode runs on one thread (scipy.fft with
its default single worker, numba 0.68 `gaussft` without `parallel`).
convolve-rs uses all 4 threads for the plane. Memory is peak RSS above the
input, in units of the image size.

| Plane (`float32`)     | racs_tools    | convolve-rs   | Speed-up |
| --------------------- | ------------- | ------------- | -------- |
| 4500 × 3900           | 1.16 s, 3.7×  | 0.16 s, 1.05× | 7.2×     |
| 4500 × 3900, 42 % NaN | 1.80 s, 4.3×  | 0.32 s, 1.08× | 5.7×     |
| 9000 × 7800           | 4.52 s, 3.5×  | 0.76 s, 1.02× | 5.9×     |
| 9000 × 7800, 42 % NaN | 10.23 s, 4.0× | 1.38 s, 1.05× | 7.4×     |

`float64` planes were 5.5 to 9.5 times faster, at 1.02 to 1.06× memory against
3.5 to 3.6× for racs_tools.

A 3000 × 2600 × 24 `float32` cube (4D, one Stokes) with a quarter of each
plane blanked, `--mode total`, median of 3 runs under `/usr/bin/time -v`:
`convolvers 3d` took 2.3 s at a peak RSS of 0.68× the cube. `beamcon_3D`
(`--ncores 4 --executor_type thread`) took 11.0 s at 1.41×.

## Profiling

```sh
scripts/profile_cube.sh 2048 2048 64 float32 total   # NX NY NCHAN DTYPE MODE
cargo bench
python scripts/bench_vs_racs_tools.py 4500x3900 9000x7800   # NYxNX ...
```
