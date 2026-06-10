import pytest
import numpy as np
import astropy.units as u
from astropy.io import fits
import radio_beam as rb

from convolve_rs import Beam, common_beam

ARCSEC = 1.0 / 3600.0  # degrees


def _rb(major_as, minor_as, pa_deg):
    return rb.Beam(major_as * u.arcsec, minor_as * u.arcsec, pa_deg * u.deg)


# ── Beam construction ─────────────────────────────────────────────────────────

class TestBeamConstructor:
    def test_basic(self):
        b = Beam(10 * ARCSEC, 8 * ARCSEC, 30.0)
        assert b.major_deg == pytest.approx(10 * ARCSEC)
        assert b.minor_deg == pytest.approx(8 * ARCSEC)
        assert b.pa_deg == pytest.approx(30.0)

    def test_arcsec_properties(self):
        b = Beam.from_arcsec(10.0, 8.0, 30.0)
        assert b.major_arcsec == pytest.approx(10.0)
        assert b.minor_arcsec == pytest.approx(8.0)

    def test_invalid_axes_raises(self):
        with pytest.raises(ValueError):
            Beam(5 * ARCSEC, 10 * ARCSEC, 0.0)  # minor > major

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError):
            Beam(float("nan"), 8 * ARCSEC, 0.0)


class TestBeamClassmethods:
    def test_from_fits_header(self):
        hdr = fits.Header()
        hdr["BMAJ"] = 10 * ARCSEC
        hdr["BMIN"] = 8 * ARCSEC
        hdr["BPA"] = 30.0
        b = Beam.from_fits_header(hdr)
        assert b.major_deg == pytest.approx(10 * ARCSEC)
        assert b.minor_deg == pytest.approx(8 * ARCSEC)
        assert b.pa_deg == pytest.approx(30.0)

    def test_from_fits_header_matches_radio_beam(self):
        hdr = fits.Header()
        hdr["BMAJ"] = 10 * ARCSEC
        hdr["BMIN"] = 8 * ARCSEC
        hdr["BPA"] = 30.0
        ours = Beam.from_fits_header(hdr)
        theirs = rb.Beam.from_fits_header(hdr)
        assert ours.major_arcsec == pytest.approx(theirs.major.to(u.arcsec).value)
        assert ours.minor_arcsec == pytest.approx(theirs.minor.to(u.arcsec).value)
        assert ours.pa_deg == pytest.approx(theirs.pa.to(u.deg).value)

    def test_from_fits_header_missing_bpa_defaults_zero(self):
        hdr = fits.Header()
        hdr["BMAJ"] = 10 * ARCSEC
        hdr["BMIN"] = 8 * ARCSEC
        assert Beam.from_fits_header(hdr).pa_deg == 0.0

    def test_from_radio_beam(self):
        rb_beam = _rb(10.0, 8.0, 30.0)
        b = Beam.from_radio_beam(rb_beam)
        assert b.major_arcsec == pytest.approx(rb_beam.major.to(u.arcsec).value)
        assert b.minor_arcsec == pytest.approx(rb_beam.minor.to(u.arcsec).value)
        assert b.pa_deg == pytest.approx(rb_beam.pa.to(u.deg).value)

    def test_from_arcsec_matches_from_radio_beam(self):
        ours = Beam.from_arcsec(10.0, 8.0, 30.0)
        rb_beam = _rb(10.0, 8.0, 30.0)
        via_rb = Beam.from_radio_beam(rb_beam)
        assert ours.major_deg == pytest.approx(via_rb.major_deg)
        assert ours.minor_deg == pytest.approx(via_rb.minor_deg)
        assert ours.pa_deg == pytest.approx(via_rb.pa_deg)


# ── Beam math vs radio_beam ───────────────────────────────────────────────────

class TestBeamMath:
    def test_convolve_matches_radio_beam(self):
        b1 = Beam.from_arcsec(10.0, 8.0, 30.0)
        b2 = Beam.from_arcsec(12.0, 6.0, 60.0)
        ours = b1.convolve(b2)
        theirs = _rb(10.0, 8.0, 30.0).convolve(_rb(12.0, 6.0, 60.0))
        assert ours.major_arcsec == pytest.approx(theirs.major.to(u.arcsec).value, rel=1e-6)
        assert ours.minor_arcsec == pytest.approx(theirs.minor.to(u.arcsec).value, rel=1e-6)
        assert ours.pa_deg == pytest.approx(theirs.pa.to(u.deg).value, abs=1e-6)

    def test_deconvolve_matches_radio_beam(self):
        b1 = Beam.from_arcsec(15.0, 10.0, 45.0)
        b2 = Beam.from_arcsec(10.0, 8.0, 30.0)
        ours = b1.deconvolve(b2)
        theirs = _rb(15.0, 10.0, 45.0).deconvolve(_rb(10.0, 8.0, 30.0))
        assert ours.major_arcsec == pytest.approx(theirs.major.to(u.arcsec).value, rel=1e-5)
        assert ours.minor_arcsec == pytest.approx(theirs.minor.to(u.arcsec).value, rel=1e-5)

    def test_deconvolve_fails_when_psf_larger(self):
        small = Beam.from_arcsec(5.0, 5.0, 0.0)
        large = Beam.from_arcsec(10.0, 10.0, 0.0)
        with pytest.raises(ValueError):
            small.deconvolve(large)

    def test_convolve_deconvolve_roundtrip(self):
        a = Beam.from_arcsec(10.0, 8.0, 30.0)
        b = Beam.from_arcsec(6.0, 5.0, 15.0)
        recovered = a.convolve(b).deconvolve(a)
        assert recovered.major_arcsec == pytest.approx(b.major_arcsec, rel=1e-6)
        assert recovered.minor_arcsec == pytest.approx(b.minor_arcsec, rel=1e-6)

    def test_area_sr(self):
        b = Beam.from_arcsec(10.0, 10.0, 0.0)
        fwhm_rad = (10.0 * ARCSEC) * np.pi / 180.0
        expected = np.pi / (4 * np.log(2)) * fwhm_rad ** 2
        assert b.area_sr() == pytest.approx(expected, rel=1e-10)


# ── common_beam vs radio_beam ─────────────────────────────────────────────────

class TestCommonBeam:
    def test_two_beams_matches_radio_beam(self):
        params = [(10.0, 8.0, 30.0), (12.0, 6.0, 60.0)]
        beams = [Beam.from_arcsec(*p) for p in params]
        theirs = _rb(*params[0]).commonbeam_with(_rb(*params[1]))
        ours = common_beam(beams)
        assert ours.major_arcsec == pytest.approx(theirs.major.to(u.arcsec).value, rel=1e-4)
        assert ours.minor_arcsec == pytest.approx(theirs.minor.to(u.arcsec).value, rel=1e-4)

    def test_many_beams_matches_radio_beam(self):
        params = [(10.0, 8.0, 30.0), (12.0, 6.0, 60.0), (11.0, 9.0, 45.0), (9.0, 7.0, 15.0)]
        beams = [Beam.from_arcsec(*p) for p in params]
        rb_beams = rb.Beams(
            major=[p[0] for p in params] * u.arcsec,
            minor=[p[1] for p in params] * u.arcsec,
            pa=[p[2] for p in params] * u.deg,
        )
        ours = common_beam(beams)
        theirs = rb_beams.common_beam()
        assert ours.major_arcsec == pytest.approx(theirs.major.to(u.arcsec).value, rel=1e-3)
        assert ours.minor_arcsec == pytest.approx(theirs.minor.to(u.arcsec).value, rel=1e-3)

    def test_identical_beams(self):
        b = Beam.from_arcsec(10.0, 8.0, 30.0)
        result = common_beam([b, b, b])
        assert result.major_arcsec == pytest.approx(b.major_arcsec, rel=1e-6)
        assert result.minor_arcsec == pytest.approx(b.minor_arcsec, rel=1e-6)

    def test_single_beam_returns_itself(self):
        b = Beam.from_arcsec(10.0, 8.0, 30.0)
        result = common_beam([b])
        assert result.major_arcsec == pytest.approx(b.major_arcsec)
        assert result.minor_arcsec == pytest.approx(b.minor_arcsec)

    def test_common_beam_contains_all_inputs(self):
        params = [(10.0, 8.0, 30.0), (12.0, 6.0, 60.0), (11.0, 9.0, 45.0)]
        beams = [Beam.from_arcsec(*p) for p in params]
        result = common_beam(beams)
        for b in beams:
            result.deconvolve(b)  # must not raise

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            common_beam([])
