# convolve-rs

A Rust port of [`beamcon`](https://github.com/alecthomson/RACS-tools) from [RACS-tools](https://github.com/alecthomson/RACS-tools). Smooths FITS images and spectral cubes to a common beam using UV-plane (FFT) convolution to avoid numerical issues with undersampled kernels.

> **Note:** This is an experiment in LLM-assisted coding with Claude. Do not trust this software as far as you can throw it.

## Installation

Requires [Rust](https://rustup.rs/) 1.85+.

```sh
cargo install --path .
```

This installs the `convolvers` binary.

## Usage

```sh
convolvers --help
convolvers 2d --help
convolvers 3d --help
```