"""convolve-rs against patchwork's convolution, on any pixel grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits
from parity import assert_matches_away_from_missing, reach_px

import convolve_rs as cr

if TYPE_CHECKING:
    from pathlib import Path

pw_convolve = pytest.importorskip("patchwork.convolve")
pw_io = pytest.importorskip("patchwork.io")

N = 96
OLD = (8.0, 6.0, 25.0)
NEW = (15.0, 11.0, 70.0)
ARCSEC = 1 / 3600
ROT = 0.4
# Pixels are at least 2 arcsec, and the kernel's FWHM under NEW's major axis
REACH = reach_px(NEW[0], 2.0)
MATRICES = {
    "usual": [[-2 * ARCSEC, 0.0], [0.0, 2 * ARCSEC]],
    "rotated": [
        [-2 * ARCSEC * np.cos(ROT), 2 * ARCSEC * np.sin(ROT)],
        [2 * ARCSEC * np.sin(ROT), 2 * ARCSEC * np.cos(ROT)],
    ],
    "skewed": [[-2 * ARCSEC, 0.6 * ARCSEC], [0.3 * ARCSEC, 3 * ARCSEC]],
    "east right": [[2 * ARCSEC, 0.0], [0.0, 2 * ARCSEC]],
}


def _sky(matrix) -> np.ndarray:
    """Noise at the ``OLD`` resolution on ``matrix``'s grid, with a blanked
    block and a lone blank pixel."""
    white = np.random.default_rng(1).normal(size=(N, N))
    point = cr.Beam.from_arcsec(1e-3, 1e-3, 0.0)
    image = cr.smooth(
        white,
        point,
        cr.Beam.from_arcsec(*OLD),
        pixel_scale_matrix=matrix,
        bunit="K",
    )
    image[30:50, 20:45] = np.nan
    image[70, 70] = np.nan
    return image


@pytest.mark.parametrize("name", list(MATRICES))
def test_matches_convolve_to_beam(name):
    matrix = MATRICES[name]
    image = _sky(matrix)
    theirs = pw_convolve.convolve_to_beam(
        image,
        pw_io.BeamShape(*OLD),
        pw_io.BeamShape(*NEW),
        np.array(matrix),
    )
    ours = cr.smooth(
        image,
        cr.Beam.from_arcsec(*OLD),
        cr.Beam.from_arcsec(*NEW),
        pixel_scale_matrix=matrix,
        nan_fraction=0.5,
    )
    # patchwork pads with zeros and does not normalise near NaNs, so the two
    # agree beyond the kernel's reach of the edges and NaNs.
    assert_matches_away_from_missing(ours, theirs, np.isnan(image), REACH)


def test_smooth_images_reads_the_cd_matrix(tmp_path: Path):
    matrix = MATRICES["rotated"]
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---SIN", "DEC--SIN"
    header["CRPIX1"], header["CRPIX2"] = N / 2, N / 2
    header["CRVAL1"], header["CRVAL2"] = 0.0, -30.0
    for i in range(2):
        for j in range(2):
            header[f"CD{i + 1}_{j + 1}"] = matrix[i][j]
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"], header["BMIN"], header["BPA"] = OLD[0] / 3600, OLD[1] / 3600, OLD[2]
    path = tmp_path / "rotated.fits"
    fits.writeto(path, _sky(matrix).astype(np.float32), header)

    target = pw_io.BeamShape(*NEW)
    theirs = fits.getdata(pw_convolve.convolve_fits(path, target, suffix="pw"))
    result = cr.smooth_images(
        [path], target_beam=cr.Beam.from_arcsec(*NEW), nan_fraction=0.5
    )
    ours = fits.getdata(result.images[0].output)
    # patchwork pads with zeros and does not normalise near NaNs, so the two
    # agree beyond the kernel's reach of the edges and NaNs.
    assert_matches_away_from_missing(ours, theirs, np.isnan(fits.getdata(path)), REACH)
