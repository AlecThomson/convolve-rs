#!/usr/bin/env python
"""Generate dummy FITS images for benchmarking convolve-rs against RACS-tools.

Each image is a celestial 2D map (RA/Dec) with a valid WCS, beam keywords
(BMAJ/BMIN/BPA), and BUNIT=Jy/beam. Pixel data is random point sources plus
Gaussian noise so the convolution does real work.

Usage:
    python make_dummy_fits.py --n 20 --size 2048 --outdir bench_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def make_header(size: int, pix_deg: float, bmaj_deg: float,
                bmin_deg: float, bpa_deg: float) -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.crpix = [size / 2, size / 2]
    w.wcs.cdelt = [-pix_deg, pix_deg]          # RA decreases with x
    w.wcs.crval = [180.0, -45.0]               # arbitrary field centre
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]
    hdr = w.to_header()
    hdr["BUNIT"] = "Jy/beam"
    hdr["BMAJ"] = bmaj_deg                      # degrees, FITS convention
    hdr["BMIN"] = bmin_deg
    hdr["BPA"] = bpa_deg                        # degrees
    return hdr


def make_data(size: int, n_sources: int, rng: np.random.Generator) -> np.ndarray:
    data = rng.normal(0.0, 1e-3, size=(size, size)).astype(np.float32)
    ys = rng.integers(0, size, n_sources)
    xs = rng.integers(0, size, n_sources)
    amps = rng.uniform(0.1, 10.0, n_sources).astype(np.float32)
    data[ys, xs] += amps                        # delta-function point sources
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20, help="number of images")
    p.add_argument("--size", type=int, default=2048, help="image side in pixels")
    p.add_argument("--pix-arcsec", type=float, default=2.5, help="pixel scale (arcsec)")
    p.add_argument("--sources", type=int, default=500, help="point sources per image")
    p.add_argument("--outdir", type=Path, default=Path("bench_data"))
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pix_deg = args.pix_arcsec / 3600.0

    for i in range(args.n):
        # vary beam per image so common_beam has something to chew on:
        # 10-15 arcsec major axis, axis ratio 0.6-1.0, random PA.
        bmaj = rng.uniform(10.0, 15.0) / 3600.0
        bmin = bmaj * rng.uniform(0.1, 0.9)
        bpa = rng.uniform(0.0, 180.0)
        hdr = make_header(args.size, pix_deg, bmaj, bmin, bpa)
        data = make_data(args.size, args.sources, rng)
        path = args.outdir / f"dummy_{i:04d}.fits"
        fits.writeto(path, data, hdr, overwrite=True)
        print(f"wrote {path}  bmaj={bmaj*3600:.2f}\" bmin={bmin*3600:.2f}\" bpa={bpa:.1f}")

    total_mb = args.n * args.size * args.size * 4 / 1e6
    print(f"\n{args.n} images, {args.size}x{args.size}, ~{total_mb:.0f} MB total in {args.outdir}/")


if __name__ == "__main__":
    main()
