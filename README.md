# convolve-rs

[![Docs](https://readthedocs.org/projects/convolve-rs/badge/?version=latest)](https://convolve-rs.readthedocs.io/en/latest/)
[![docs.rs](https://img.shields.io/docsrs/convolve-rs)](https://docs.rs/convolve-rs)
[![PyPI](https://img.shields.io/pypi/v/convolve-rs)](https://pypi.org/project/convolve-rs/)
[![crates.io](https://img.shields.io/crates/v/convolve-rs)](https://crates.io/crates/convolve-rs)

**Documentation:** Python API, CLI, and algorithm background at
[convolve-rs.readthedocs.io](https://convolve-rs.readthedocs.io/); Rust API at
[docs.rs/convolve-rs](https://docs.rs/convolve-rs).

<!-- SPHINX-START -->

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

## Usage

- Python API and quickstart:
  [convolve-rs.readthedocs.io/en/latest/quickstart.html](https://convolve-rs.readthedocs.io/en/latest/quickstart.html)
- CLI reference:
  [convolve-rs.readthedocs.io/en/latest/cli.html](https://convolve-rs.readthedocs.io/en/latest/cli.html)
- Rust API:
  [docs.rs/convolve-rs](https://docs.rs/convolve-rs)

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

convolve-rs is released under the
[BSD 3-Clause License](https://github.com/alecthomson/convolve-rs/blob/main/LICENSE).

It builds on prior work in the radio-astronomy community: the UV-plane convolution
and cube handling are ported from [RACS-tools](https://github.com/alecthomson/RACS-tools),
and the common-beam computation follows [radio_beam](https://github.com/radio-astro-tools/radio-beam)
(both BSD). The Gaussian beam algebra implements the standard formulae of Wild
(1970), and [MIRIAD](https://github.com/csiro/miriad) (GPL) serves as a validation
reference in the test suite. See
[NOTICE.md](https://github.com/alecthomson/convolve-rs/blob/main/NOTICE.md)
for full attributions.
