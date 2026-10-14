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
parallel instead, each plane on one thread, and stream finished planes to a
single writer thread through a bounded queue, so memory is capped at the
planes in flight. The 2D CLI runs images in parallel when there are at least
as many as threads, and one at a time otherwise.

On 4 cores, `cargo bench` gives:

| Image  | f32     | f64    | f32 with NaNs |
| ------ | ------- | ------ | ------------- |
| 2048²  | 46 ms   | 73 ms  | 54 ms         |
| 4096²  | 148 ms  | 250 ms |               |

(compared-with-racs-tools)=

## Compared with racs_tools

One plane per call from Python on a 4-core machine, with each convolve-rs plane
on a single thread. Memory is peak RSS above the input, in units of the image
size. Re-measure with `scripts/bench_vs_racs_tools.py`.

| Plane (`float32`)       | racs_tools   | convolve-rs  | Speed-up |
| ----------------------- | ------------ | ------------ | -------- |
| 4500 × 3900             | 1.06 s, 3.8× | 0.27 s, 1.0× | 3.9×     |
| 4500 × 3900, 42 % NaN   | 1.70 s, 4.3× | 0.59 s, 1.1× | 2.9×     |
| 9000 × 7800             | 4.09 s, 3.5× | 1.17 s, 1.0× | 3.5×     |
| 9000 × 7800, 42 % NaN   | 7.14 s, 4.0× | 2.40 s, 1.0× | 3.0×     |

`float64` planes behave the same way (3.4 to 4.7 times faster, 1.0× against
3.5× memory). `convolvers 3d` on a 3000 × 2600 × 24 `float32` cube with a
blanked region took 2.1 s at a peak RSS of 0.56× the cube, against 12.5 s and
1.48× for `beamcon_3D` with four threads.

For large cubes the time is mostly FITS I/O, which is also why there is no GPU
backend.

## Profiling

```sh
scripts/profile_cube.sh 2048 2048 64 float32 total   # NX NY NCHAN DTYPE MODE
cargo bench
python scripts/bench_vs_racs_tools.py 4500x3900 9000x7800   # NYxNX ...
```
