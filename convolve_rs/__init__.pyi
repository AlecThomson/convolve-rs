from __future__ import annotations

from typing import TYPE_CHECKING, Any

from convolve_rs._convolve_rs import (
    Beam as _Beam,
    common_beam as common_beam,
    smooth as smooth,
    gauss_factor as gauss_factor,
)

if TYPE_CHECKING:
    from astropy.io.fits import Header

__all__ = ["Beam", "common_beam", "smooth", "gauss_factor"]


class Beam(_Beam):
    """A 2-D Gaussian representation of a radio telescope's PSF (beam).

    Extends the Rust Beam with convenience constructors for astropy and radio_beam.
    All core methods (convolve, deconvolve, area_sr, etc.) are inherited.

    Args:
        major_deg: FWHM major axis in degrees (FITS BMAJ).
        minor_deg: FWHM minor axis in degrees (FITS BMIN). Must be <= major_deg.
        pa_deg: Position angle in degrees East of North (FITS BPA).
    """

    @classmethod
    def from_fits_header(cls, header: Header) -> Beam:
        """Construct from an astropy FITS header.

        Reads BMAJ and BMIN (degrees) and BPA (degrees, defaults to 0).

        Args:
            header: An astropy.io.fits.Header with BMAJ and BMIN keys.

        Returns:
            The constructed beam.
        """
        ...

    @classmethod
    def from_radio_beam(cls, rb: Any) -> Beam:
        """Construct from a radio_beam.Beam or compatible duck-typed object.

        Args:
            rb: An object with major, minor, pa astropy Quantity attributes.

        Returns:
            The constructed beam.
        """
        ...
