---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Python quickstart

This page is a notebook that is re-run on every docs build.

## Beams

A {class}`~convolve_rs.Beam` is a 2D elliptical Gaussian in FITS conventions:
FWHM major/minor axes in degrees, position angle in degrees East of North.

```{code-cell} ipython3
from convolve_rs import Beam

beam = Beam.from_arcsec(20.0, 10.0, 45.0)
print(beam)
```

Beam convolution and deconvolution follow the standard Gaussian algebra (see
[Algorithm background](algorithms.md)):

```{code-cell} ipython3
a = Beam.from_arcsec(3.0, 3.0, 0.0)
b = Beam.from_arcsec(4.0, 4.0, 0.0)

c = a.convolve(b)
print(f"convolved:    {c}")
print(f"deconvolved:  {c.deconvolve(a)}")
```

Deconvolving a beam that is too small raises a `ValueError`:

```{code-cell} ipython3
:tags: [raises-exception]

b.deconvolve(c)
```

## Common beam

{func}`~convolve_rs.common_beam` finds the smallest beam that a whole set of
beams (e.g. the channels of a cube) can be convolved to:

```{code-cell} ipython3
from convolve_rs import common_beam

beams = [
    Beam.from_arcsec(10.0, 8.0, 30.0),
    Beam.from_arcsec(12.0, 6.0, 60.0),
    Beam.from_arcsec(11.0, 7.0, -40.0),
]
target = common_beam(beams)
print(target)
```

## Smoothing an image

{func}`~convolve_rs.smooth` convolves an image from one beam to another in the
UV plane and applies the flux scaling appropriate for its brightness unit.
Here we build a synthetic image: a point source restored with a 10″ beam, plus
noise.

```{code-cell} ipython3
import numpy as np

rng = np.random.default_rng(42)

n = 256
dx_deg = 2.5 / 3600.0  # 2.5 arcsec pixels

old_beam = Beam.from_arcsec(10.0, 10.0, 0.0)
new_beam = Beam.from_arcsec(30.0, 30.0, 0.0)

# Restored point source: a Gaussian with the old beam's FWHM.
y, x = np.mgrid[:n, :n] - n / 2
sigma_pix = (old_beam.major_arcsec / 2.5) / (2 * np.sqrt(2 * np.log(2)))
image = np.exp(-(x**2 + y**2) / (2 * sigma_pix**2)).astype(np.float32)
image += rng.normal(0, 0.01, (n, n)).astype(np.float32)
```

```{code-cell} ipython3
from convolve_rs import smooth

smoothed = smooth(image, old_beam, new_beam, dx_deg, dx_deg, bunit="Jy/beam")
```

For a Jy/beam image, the peak of an unresolved source stays (close to)
constant under convolution, because the flux scaling cancels the smearing:

```{code-cell} ipython3
print(f"input peak:    {image.max():.3f} Jy/beam")
print(f"smoothed peak: {smoothed.max():.3f} Jy/beam")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for ax, (data, title) in zip(
    axs,
    [(image, f"input ({old_beam})"), (smoothed, f"smoothed ({new_beam})")],
):
    ax.imshow(data, origin="lower", vmin=-0.05, vmax=0.5, cmap="cubehelix")
    ax.set_title(title, fontsize=9)
plt.show()
```

## NaN handling

NaN pixels, like the sky beyond the image's edges, are missing data. Pixels
near them are divided by the share of their kernel that fell on data, so
their flux scale is right, and a pixel with more than half its kernel on
missing data is blanked:

```{code-cell} ipython3
image_nan = image.copy()
image_nan[100:120, 100:120] = np.nan

smoothed_nan = smooth(image_nan, old_beam, new_beam, dx_deg, dx_deg, bunit="Jy/beam")
print(f"NaN pixels in: {np.isnan(image_nan).sum()}, out: {np.isnan(smoothed_nan).sum()}")
```

`nan_fraction` sets that tolerance. A lower one keeps only pixels whose kernel
lies almost all on data:

```{code-cell} ipython3
strict = smooth(
    image_nan, old_beam, new_beam, dx_deg, dx_deg, bunit="Jy/beam", nan_fraction=0.01
)
print(f"NaN pixels out: {np.isnan(strict).sum()}")
```

For a rotated or skewed grid, pass the pixel-to-sky matrix in degrees, as
astropy's `WCS.pixel_scale_matrix` gives it, instead of `dx_deg` and `dy_deg`:

```{code-cell} ipython3
theta = np.radians(20.0)
matrix = dx_deg * np.array(
    [[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
)
rotated = smooth(image, old_beam, new_beam, pixel_scale_matrix=matrix)
```

## Working with FITS files

With [astropy](https://www.astropy.org/), reading the beam from a header and
writing the smoothed image back looks like:

```python
from astropy.io import fits
from convolve_rs import Beam, smooth

with fits.open("image.fits") as hdul:
    header = hdul[0].header
    data = hdul[0].data.squeeze().astype("float32")

    current = Beam.from_fits_header(header)
    target = Beam.from_arcsec(30.0, 30.0, 0.0)
    smoothed = smooth(
        data, current, target,
        header["CDELT1"], header["CDELT2"],
        bunit=header.get("BUNIT"),
    )

    header["BMAJ"], header["BMIN"], header["BPA"] = (
        target.major_deg, target.minor_deg, target.pa_deg,
    )
    fits.writeto("smoothed.fits", smoothed, header, overwrite=True)
```

## Smoothing files and cubes

`smooth_images` and `smooth_cubes` do what `convolvers 2d` and `convolvers 3d`
do, from Python: find the common beam, smooth every image or channel in
parallel, and write the results beside the inputs. They read beamlogs and CASA
multi-beam tables, and release the GIL while they run.

```python
from convolve_rs import Beam, cube_beams, smooth_cubes, smooth_images

result = smooth_images(["a.fits", "b.fits"], cutoff=25.0)
print(result.target)
for image in result.images:
    print(image.output, "blanked" if image.kernel is None else image.kernel)

(cube,) = smooth_cubes(["cube.fits"], mode="natural", ncores=8)
print(cube.output, cube.targets[0])
print(cube_beams(cube.output))
```

An image or channel with no usable beam, or one over the cutoff, is written as
NaN with a zero beam (masked in a cube's BEAMS table), so it is not mistaken
for one at the target resolution. See the [CLI reference](cli.md) for what
each option does.
