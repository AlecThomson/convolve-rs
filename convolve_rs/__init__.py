from convolve_rs._convolve_rs import Beam as _Beam, common_beam, smooth, gauss_factor


class Beam(_Beam):
    @classmethod
    def from_fits_header(cls, header):
        """Construct from an astropy FITS header (BMAJ/BMIN/BPA in degrees)."""
        return cls(header["BMAJ"], header["BMIN"], header.get("BPA", 0.0))

    @classmethod
    def from_radio_beam(cls, rb):
        """Construct from a radio_beam.Beam object (duck-typed; no hard dependency)."""
        import astropy.units as u
        return cls(
            rb.major.to(u.deg).value,
            rb.minor.to(u.deg).value,
            rb.pa.to(u.deg).value,
        )


__all__ = ["Beam", "common_beam", "smooth", "gauss_factor"]
