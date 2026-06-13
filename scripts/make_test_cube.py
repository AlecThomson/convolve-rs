#!/usr/bin/env python3
"""Generate a synthetic FITS spectral cube for profiling `convolvers 3d`.

Writes a NAXIS=3 cube (nchan, ny, nx) with a CASA ``BEAMS`` binary-table
extension carrying a slightly different beam per channel, so a common-beam
smooth has a non-trivial convolving kernel. Pixel precision follows ``--dtype``
(float32 → BITPIX -32, float64 → BITPIX -64), which is what the convolver now
dispatches on.

Requires astropy (a dev/profiling dependency only):

    python scripts/make_test_cube.py --nx 2048 --ny 2048 --nchan 64 \
        --dtype float32 -o /tmp/profile_cube.fits
"""

from __future__ import annotations

import argparse

import numpy as np
from astropy.io import fits


def build_cube(nx: int, ny: int, nchan: int, dtype: str) -> fits.HDUList:
    rng = np.random.default_rng(0)
    # A handful of point sources plus mild noise: cheap to make, non-trivial to
    # convolve, and finite everywhere.
    data = rng.normal(0.0, 1e-3, size=(nchan, ny, nx)).astype(dtype)
    for k in range(nchan):
        for _ in range(20):
            i = rng.integers(0, ny)
            j = rng.integers(0, nx)
            data[k, i, j] += 1.0

    primary = fits.PrimaryHDU(data=data)
    h = primary.header
    h["CDELT1"] = -2.5 / 3600.0  # 2.5" pixels
    h["CDELT2"] = 2.5 / 3600.0
    h["CRPIX3"] = 1
    h["BUNIT"] = "Jy/beam"
    h["CASAMBM"] = True

    chans = np.arange(nchan, dtype=np.int32)
    bmaj = (15.0 + 0.05 * chans).astype(np.float32)  # arcsec
    bmin = (12.0 + 0.05 * chans).astype(np.float32)
    bpa = (chans.astype(np.float32) % 30.0)
    cols = fits.ColDefs(
        [
            fits.Column(name="BMAJ", format="E", unit="arcsec", array=bmaj),
            fits.Column(name="BMIN", format="E", unit="arcsec", array=bmin),
            fits.Column(name="BPA", format="E", unit="deg", array=bpa),
            fits.Column(name="CHAN", format="J", array=chans),
            fits.Column(name="POL", format="J", array=np.zeros(nchan, np.int32)),
        ]
    )
    beams = fits.BinTableHDU.from_columns(cols, name="BEAMS")
    beams.header["NCHAN"] = nchan
    beams.header["NPOL"] = 1
    return fits.HDUList([primary, beams])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=2048)
    p.add_argument("--ny", type=int, default=2048)
    p.add_argument("--nchan", type=int, default=64)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("-o", "--out", default="profile_cube.fits")
    args = p.parse_args()

    hdul = build_cube(args.nx, args.ny, args.nchan, args.dtype)
    hdul.writeto(args.out, overwrite=True)
    nbytes = args.nx * args.ny * args.nchan * (4 if args.dtype == "float32" else 8)
    print(
        f"Wrote {args.out}: {args.nchan}×{args.ny}×{args.nx} {args.dtype} "
        f"(~{nbytes / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
