"""Comparing convolve-rs with tools that handle edges and NaNs differently."""

from __future__ import annotations

import numpy as np


def clear_of_missing(blank: np.ndarray, reach: int) -> np.ndarray:
    """Pixels more than ``reach`` from any ``blank`` pixel or the image edge,
    over the last two axes. There a convolution does not depend on how it
    treats missing data, so every correct tool gives the same result."""
    ny, nx = blank.shape[-2:]
    padded = np.pad(blank, [(0, 0)] * (blank.ndim - 2) + [(reach + 1, reach)] * 2, constant_values=True)
    counts = padded.astype(np.int64).cumsum(-2).cumsum(-1)
    w = 2 * reach + 1
    window = (
        counts[..., w:, w:]
        - counts[..., :-w, w:]
        - counts[..., w:, :-w]
        + counts[..., :-w, :-w]
    )
    return window[..., :ny, :nx] == 0


def reach_px(fwhm_arcsec: float, pix_arcsec: float) -> int:
    """6.1 standard deviations of a Gaussian of this FWHM, in pixels: no larger
    than the reach of a kernel up to this size."""
    return int(np.ceil(6.1 * fwhm_arcsec / (2 * np.sqrt(2 * np.log(2))) / pix_arcsec))


def assert_matches_away_from_missing(
    ours: np.ndarray,
    theirs: np.ndarray,
    missing: np.ndarray,
    reach: int,
    tolerance: float = 1e-4,
) -> None:
    """Equal, to ``tolerance`` of the noise, wherever the kernel cannot reach
    missing data; and ours blank inside each blanked region of theirs.

    A kernel narrower than about a pixel is cut off at the Nyquist frequency,
    so it rings, and edges then reach further than a Gaussian would: as 1/r.
    Tests of such kernels on small images need a larger ``tolerance``."""
    clear = clear_of_missing(missing, reach)
    assert clear.sum() > 0.05 * clear.size, "too little of the image to compare"
    assert np.isfinite(ours[clear]).all()
    np.testing.assert_allclose(
        ours[clear], theirs[clear], rtol=0, atol=tolerance * np.nanstd(theirs[clear])
    )
    deep = clear_of_missing(~missing, reach)
    assert np.isnan(ours[deep]).all()
