"""Smooth radio images and cubes to a common beam by UV-plane convolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from convolve_rs._convolve_rs import Beam as _Beam
from convolve_rs._convolve_rs import common_beam, gauss_factor, smooth

if TYPE_CHECKING:
    from astropy.io.fits import Header


class Beam(_Beam):
    """A 2D elliptical Gaussian beam (PSF), with astropy and radio_beam constructors.

    Parameters
    ----------
    major_deg:
        FWHM major axis in degrees (FITS BMAJ).
    minor_deg:
        FWHM minor axis in degrees (FITS BMIN), at most ``major_deg``.
    pa_deg:
        Position angle in degrees East of North (FITS BPA).
    """

    @classmethod
    def from_fits_header(cls, header: Header) -> Beam:
        """Read ``BMAJ``, ``BMIN`` and ``BPA`` (default 0) from a FITS header.

        Examples
        --------
        >>> from astropy.io import fits
        >>> header = fits.Header({"BMAJ": 0.005, "BMIN": 0.004, "BPA": 30.0})
        >>> Beam.from_fits_header(header).major_deg
        0.005
        """
        return cls(
            cast("float", header["BMAJ"]),
            cast("float", header["BMIN"]),
            cast("float", header.get("BPA", 0.0)),
        )

    @classmethod
    def from_radio_beam(cls, rb: Any) -> Beam:
        """Convert a ``radio_beam.Beam``, or anything with angle ``major``,
        ``minor`` and ``pa`` quantities.

        Examples
        --------
        >>> import astropy.units as u
        >>> import radio_beam
        >>> rb = radio_beam.Beam(10 * u.arcsec, 8 * u.arcsec, 30 * u.deg)
        >>> round(Beam.from_radio_beam(rb).major_arcsec, 6)
        10.0
        """
        import astropy.units as u  # noqa: PLC0415  (optional dependency)

        return cls(
            rb.major.to(u.deg).value,
            rb.minor.to(u.deg).value,
            rb.pa.to(u.deg).value,
        )


__all__ = ["Beam", "common_beam", "gauss_factor", "smooth"]
