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
cargo install convolve-rs
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

# `bunit` selects the flux scaling: Jy/beam images are rescaled to stay in
# Jy/beam; Kelvin (brightness temperature) images are left unscaled. An
# unrecognised unit emits a UserWarning and is treated as Jy/beam.
smoothed = smooth(data, current, target, dx_deg, dy_deg, bunit=hdu[0].header.get("BUNIT"))

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

After changing the Python-facing Rust API in `src/python.rs`, rebuild with the
`stubgen` feature (the default build omits `_generate_stubs`) and regenerate
the type stubs:

```sh
uv run maturin develop --features stubgen
uv run --no-sync python -c "from convolve_rs._convolve_rs import _generate_stubs; _generate_stubs()"
```

This overwrites `convolve_rs/_convolve_rs.pyi` from the Rust annotations and docstrings. Commit the result alongside any API changes.

### Running tests

**Python** tests need the compiled extension and the `test` extra (`pytest`,
`radio-beam`, `astropy`). `uv sync` builds the maturin extension into `.venv`,
so a plain `uvx pytest` won't work — it runs in an isolated env with neither the
module nor the deps:

```sh
uv sync --extra test       # builds the extension + installs test deps
uv run --no-sync pytest
```

**Rust** tests run with `cargo test`. One integration test compares output
against [MIRIAD](https://github.com/csiro/miriad); point `MIRIAD_BIN` at a
MIRIAD `bin` directory to enable it (it is skipped when unset):

```sh
cargo test
MIRIAD_BIN=/path/to/miriad/bin cargo test   # include the MIRIAD comparison
```

### Pre-commit hooks

Formatters and linters run via [prek](https://github.com/j178/prek), a fast
drop-in [pre-commit](https://pre-commit.com) reimplementation. The hooks
(`.pre-commit-config.yaml`) are the same checks CI enforces: `ruff` lint +
format, [`ty`](https://github.com/astral-sh/ty) type checking, `cargo fmt`, and
`cargo clippy`.

```sh
uv sync --extra dev   # installs prek + ty into the venv
uvx prek install      # install the git hook (runs on every commit)
uvx prek run --all-files   # run all hooks manually
```

## License

convolve-rs is released under the [BSD 3-Clause License](LICENSE).

It builds on prior work in the radio-astronomy community: the UV-plane convolution
and cube handling are ported from [RACS-tools](https://github.com/alecthomson/RACS-tools),
and the common-beam computation follows [radio_beam](https://github.com/radio-astro-tools/radio-beam)
(both BSD). The Gaussian beam algebra implements the standard formulae of Wild
(1970), and [MIRIAD](https://github.com/csiro/miriad) (GPL) serves as a validation
reference in the test suite. See [NOTICE.md](NOTICE.md) for full attributions.
