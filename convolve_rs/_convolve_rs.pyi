"""Type stubs for the convolve_rs Rust extension module."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

class Beam:
    """A 2-D Gaussian representation of a radio telescope's PSF (beam).

    All axes use FITS conventions: FWHM major and minor axes in degrees,
    position angle in degrees East of North.

    Args:
        major_deg: FWHM major axis in degrees (FITS BMAJ).
        minor_deg: FWHM minor axis in degrees (FITS BMIN). Must be <= major_deg.
        pa_deg: Position angle in degrees East of North (FITS BPA).

    Raises:
        ValueError: If minor_deg > major_deg or any value is non-finite.
    """

    major_deg: float
    """FWHM major axis in degrees (FITS BMAJ)."""
    minor_deg: float
    """FWHM minor axis in degrees (FITS BMIN)."""
    pa_deg: float
    """Position angle in degrees East of North (FITS BPA)."""
    major_arcsec: float
    """FWHM major axis in arcseconds."""
    minor_arcsec: float
    """FWHM minor axis in arcseconds."""

    def __new__(cls, major_deg: float, minor_deg: float, pa_deg: float) -> Beam: ...

    @classmethod
    def from_arcsec(
        cls,
        major_arcsec: float,
        minor_arcsec: float,
        pa_deg: float,
    ) -> Beam:
        """Construct a Beam from arcsecond axes.

        Args:
            major_arcsec: FWHM major axis in arcseconds.
            minor_arcsec: FWHM minor axis in arcseconds. Must be <= major_arcsec.
            pa_deg: Position angle in degrees East of North.

        Returns:
            The constructed beam.

        Raises:
            ValueError: If minor_arcsec > major_arcsec or any value is non-finite.
        """
        ...

    def area_sr(self) -> float:
        """Solid angle of the beam in steradians.

        Computed as ``(pi / (4 ln 2)) * major_rad * minor_rad``.

        Returns:
            Beam solid angle in steradians.
        """
        ...

    def deconvolve(self, other: Beam) -> Beam:
        """Deconvolve ``other`` from ``self`` (i.e. ``self`` = result ⊛ ``other``).

        Uses the MIRIAD GauDfac algorithm (R. Sault).

        Args:
            other: The PSF to deconvolve from this beam.

        Returns:
            The deconvolved beam.

        Raises:
            ValueError: If ``other`` is larger than ``self`` and deconvolution
                is impossible.
        """
        ...

    def convolve(self, other: Beam) -> Beam:
        """Convolve ``self`` with ``other``.

        Uses the MIRIAD GauCvl algorithm (R. Sault).

        Args:
            other: The beam to convolve with.

        Returns:
            The convolved beam.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


def common_beam(
    beams: list[Beam],
    tolerance: float = 1e-4,
    nsamps: int = 200,
    epsilon: float = 5e-4,
) -> Beam:
    """Find the smallest beam that every beam in ``beams`` can be convolved to.

    Uses the 2-beam analytic CASA algorithm when ``len(beams) == 2``, otherwise
    the Khachiyan minimum-volume-enclosing-ellipse algorithm — the same as
    ``radio_beam.Beams.common_beam(method='pts')``.

    Args:
        beams: Input beams. Must contain at least one element.
        tolerance: Convergence tolerance for the Khachiyan algorithm.
        nsamps: Number of points sampled from each beam ellipse boundary.
        epsilon: Fractional padding added to each beam before the MVE fit,
            to ensure the common beam can be marginally deconvolved from all inputs.

    Returns:
        The smallest common beam.

    Raises:
        ValueError: If ``beams`` is empty or no valid common beam is found.
    """
    ...


def smooth(
    image: npt.NDArray[np.float32],
    old_beam: Beam,
    new_beam: Beam,
    dx_deg: float,
    dy_deg: float,
    cutoff_arcsec: float | None = None,
) -> npt.NDArray[np.float32]:
    """Smooth a Jy/beam image from ``old_beam`` to ``new_beam``.

    Convolves ``image`` in the UV plane and applies the Jy/beam flux scaling
    factor so that the output is in the same units as the input.

    Args:
        image: Input image in Jy/beam, shape ``(ny, nx)``, dtype ``float32``.
        old_beam: Current (input) restoring beam.
        new_beam: Target (output) restoring beam. Must be larger than ``old_beam``.
        dx_deg: Pixel size along the x (RA) axis in degrees
            (FITS CDELT1, may be negative).
        dy_deg: Pixel size along the y (Dec) axis in degrees (FITS CDELT2).
        cutoff_arcsec: If given, raise ``ValueError`` if the deconvolved kernel
            FWHM exceeds this value in arcseconds.

    Returns:
        Smoothed image in Jy/beam, shape ``(ny, nx)``, dtype ``float32``.

    Raises:
        ValueError: If ``new_beam`` is smaller than ``old_beam``, all pixels
            are NaN, or the kernel exceeds ``cutoff_arcsec``.
    """
    ...


def gauss_factor(
    conv_beam: Beam,
    orig_beam: Beam,
    dx_arcsec: float,
    dy_arcsec: float,
) -> tuple[float, float, float, float, float]:
    """Compute the MIRIAD ``gaufac`` flux-scaling factor for a Jy/beam convolution.

    Returns the factor by which pixel values must be multiplied after
    convolving a Jy/beam image from ``orig_beam`` to ``conv_beam``.

    Args:
        conv_beam: The convolving beam (the kernel applied on top of ``orig_beam``).
        orig_beam: The original restoring beam of the image.
        dx_arcsec: Pixel size along the x axis in arcseconds.
        dy_arcsec: Pixel size along the y axis in arcseconds.

    Returns:
        Tuple of ``(fac, amp, bmaj_out, bmin_out, bpa_out_deg)`` where ``fac``
        is the pixel scaling factor, ``amp`` is the Gaussian kernel integral,
        and the remaining three are the output beam parameters (major/minor
        FWHM in arcseconds, PA in degrees).
    """
    ...
