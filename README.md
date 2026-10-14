# convolve-rs

[![Docs](https://readthedocs.org/projects/convolve-rs/badge/?version=latest)](https://convolve-rs.readthedocs.io/en/latest/)
[![docs.rs](https://img.shields.io/docsrs/convolve-rs)](https://docs.rs/convolve-rs)
[![PyPI](https://img.shields.io/pypi/v/convolve-rs)](https://pypi.org/project/convolve-rs/)
[![crates.io](https://img.shields.io/crates/v/convolve-rs)](https://crates.io/crates/convolve-rs)

Documentation for the Python API, CLI and algorithms is at
[convolve-rs.readthedocs.io](https://convolve-rs.readthedocs.io/). The Rust API
is at [docs.rs/convolve-rs](https://docs.rs/convolve-rs).

<!-- SPHINX-START -->

A Rust port of `beamcon` from
[RACS-tools](https://github.com/alecthomson/RACS-tools). It smooths FITS images
and spectral cubes to a common beam. The convolution is done in the UV plane
(FFT), which avoids numerical problems with undersampled kernels.

> Note: this is an experiment in LLM-assisted coding with Claude. Do not trust this software as far as you can throw it.

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

This overwrites `convolve_rs/_convolve_rs.pyi` from the Rust annotations and
docstrings. Commit it with the API change.

### Running tests

Python tests need the compiled extension and the `test` extra (`pytest`,
`radio-beam`, `astropy`). `uv sync` builds the extension into `.venv`.
`uvx pytest` does not work, because it runs in an isolated environment without
the module or its dependencies.

```sh
uv sync --extra test       # builds the extension + installs test deps
uv run --no-sync pytest
```

Rust tests run with `cargo test`. One integration test compares output with
[MIRIAD](https://github.com/csiro/miriad). Set `MIRIAD_BIN` to a MIRIAD `bin`
directory to run it; it is skipped otherwise.

```sh
cargo test
MIRIAD_BIN=/path/to/miriad/bin cargo test   # include the MIRIAD comparison
```

### Pre-commit hooks

Formatters and linters run with [prek](https://github.com/j178/prek), a
reimplementation of [pre-commit](https://pre-commit.com). The hooks in
`.pre-commit-config.yaml` are the checks CI runs: `ruff` lint and format,
[`ty`](https://github.com/astral-sh/ty) type checking, `cargo fmt` and
`cargo clippy`.

```sh
uv sync --extra dev   # installs prek + ty into the venv
uvx prek install      # install the git hook (runs on every commit)
uvx prek run --all-files   # run all hooks manually
```

## License

convolve-rs is released under the
[BSD 3-Clause License](https://github.com/alecthomson/convolve-rs/blob/main/LICENSE).

The UV-plane convolution and cube handling are ported from
[RACS-tools](https://github.com/alecthomson/RACS-tools). The common-beam
computation follows [radio_beam](https://github.com/radio-astro-tools/radio-beam).
Both are BSD licensed. The beam algebra uses the formulae of Wild (1970).
[MIRIAD](https://github.com/csiro/miriad) (GPL) is used as a reference in the
tests. See
[NOTICE.md](https://github.com/alecthomson/convolve-rs/blob/main/NOTICE.md)
for full attributions.
