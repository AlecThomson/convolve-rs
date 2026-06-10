# convolve-rs

A Rust port of [`beamcon`](https://github.com/alecthomson/RACS-tools) from [RACS-tools](https://github.com/alecthomson/RACS-tools). Smooths FITS images and spectral cubes to a common beam using UV-plane (FFT) convolution to avoid numerical issues with undersampled kernels.

> **Note:** This is an experiment in LLM-assisted coding with Claude. Do not trust this software as far as you can throw it.

## Installation

### Python library

```sh
pip install convolve-rs
```

### CLI binary

Requires [Rust](https://rustup.rs/) 1.85+.

```sh
cargo install --path .
```

## Python usage

```python
from astropy.io import fits
import numpy as np
from convolve_rs import Beam, common_beam, smooth

hdu = fits.open("image.fits")
data = hdu[0].data.squeeze().astype(np.float32)
dx_deg = hdu[0].header["CDELT1"]   # may be negative
dy_deg = hdu[0].header["CDELT2"]

current = Beam.from_fits_header(hdu[0].header)
target = Beam(0.002, 0.002, 0.0)   # or common_beam([...]) across channels

smoothed = smooth(data, current, target, dx_deg, dy_deg)

hdu[0].data[0, 0] = smoothed
fits.writeto("smoothed.fits", hdu[0].data, hdu[0].header, overwrite=True)
```

## CLI usage

```sh
convolvers --help
convolvers 2d --help
convolvers 3d --help
```

## Development

Install in editable mode:

```sh
uv pip install -e .
```

After changing the Python-facing Rust API in `src/python.rs`, regenerate the type stubs:

```sh
uv run python -c "from convolve_rs._convolve_rs import _generate_stubs; _generate_stubs()"
```

This overwrites `convolve_rs/_convolve_rs.pyi` from the Rust annotations and docstrings. Commit the result alongside any API changes.