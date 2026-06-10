from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from convolve_rs._convolve_rs import Beam as _Beam
from convolve_rs._convolve_rs import common_beam, gauss_factor, smooth

if TYPE_CHECKING:
    from astropy.io.fits import Header


class Beam(_Beam):
    """A 2-D Gaussian representation of a radio telescope's PSF (beam).

    Extends the base :class:`Beam` with convenience constructors for the
    astropy and radio_beam ecosystems.  All core functionality (``convolve``,
    ``deconvolve``, ``area_sr``, etc.) is inherited from the Rust implementation.

    All axes use FITS conventions: FWHM major and minor axes in degrees,
    position angle in degrees East of North.

    Parameters
    ----------
    major_deg:
        FWHM major axis in degrees (FITS BMAJ).
    minor_deg:
        FWHM minor axis in degrees (FITS BMIN).  Must be <= major_deg.
    pa_deg:
        Position angle in degrees East of North (FITS BPA).

    See Also
    --------
    Beam.from_arcsec : Construct from arcsecond axes.
    Beam.from_fits_header : Construct from an astropy FITS header.
    Beam.from_radio_beam : Construct from a ``radio_beam.Beam`` object.
    common_beam : Find the smallest common beam for a set of beams.
    """

    @classmethod
    def from_fits_header(cls, header: Header) -> Beam:
        """Construct from an astropy FITS header.

        Reads ``BMAJ`` and ``BMIN`` (in degrees) and ``BPA`` (in degrees,
        defaults to 0 if absent) from the header.

        Parameters
        ----------
        header:
            An :class:`astropy.io.fits.Header` (or any mapping with ``BMAJ``
            and ``BMIN`` keys and an optional ``BPA`` key).

        Returns
        -------
        Beam

        Examples
        --------
        >>> from astropy.io import fits
        >>> hdu = fits.open("image.fits")[0]
        >>> beam = Beam.from_fits_header(hdu.header)
        """
        return cls(
            cast("float", header["BMAJ"]),
            cast("float", header["BMIN"]),
            cast("float", header.get("BPA", 0.0)),
        )

    @classmethod
    def from_radio_beam(cls, rb: Any) -> Beam:
        """Construct from a ``radio_beam.Beam`` object.

        Duck-typed: any object with ``major``, ``minor``, and ``pa`` attributes
        that are astropy :class:`~astropy.units.Quantity` angle values works.
        ``radio_beam`` is not a hard dependency.

        Parameters
        ----------
        rb:
            A ``radio_beam.Beam`` (or compatible object).

        Returns
        -------
        Beam

        Examples
        --------
        >>> import radio_beam, astropy.units as u
        >>> rb = radio_beam.Beam(10 * u.arcsec, 8 * u.arcsec, 30 * u.deg)
        >>> beam = Beam.from_radio_beam(rb)
        """
        import astropy.units as u  # noqa: PLC0415  (optional dependency, imported lazily)

        return cls(
            rb.major.to(u.deg).value,
            rb.minor.to(u.deg).value,
            rb.pa.to(u.deg).value,
        )


__all__ = ["Beam", "common_beam", "gauss_factor", "smooth"]
