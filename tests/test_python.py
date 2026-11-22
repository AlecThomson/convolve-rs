import astropy.units as u
import numpy as np
import pytest
import radio_beam as rb
from astropy.io import fits
from parity import assert_matches_away_from_missing, reach_px

from convolve_rs import Beam, common_beam, smooth

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


class TestBeamSubclass:
    def test_methods_return_subclass(self):
        a = Beam.from_arcsec(10.0, 8.0, 30.0)
        b = Beam.from_arcsec(5.0, 4.0, 0.0)
        results = [
            a,
            a.convolve(b),
            a.convolve(b).deconvolve(b),
            common_beam([a, b]),
            common_beam([a]),
            common_beam([a, b, Beam.from_arcsec(9.0, 9.0, 0.0)]),
        ]
        for r in results:
            assert type(r) is Beam


# ── Beam math vs radio_beam ───────────────────────────────────────────────────


class TestBeamMath:
    def test_convolve_matches_radio_beam(self):
        b1 = Beam.from_arcsec(10.0, 8.0, 30.0)
        b2 = Beam.from_arcsec(12.0, 6.0, 60.0)
        ours = b1.convolve(b2)
        theirs = _rb(10.0, 8.0, 30.0).convolve(_rb(12.0, 6.0, 60.0))
        assert ours.major_arcsec == pytest.approx(
            theirs.major.to(u.arcsec).value, rel=1e-6
        )
        assert ours.minor_arcsec == pytest.approx(
            theirs.minor.to(u.arcsec).value, rel=1e-6
        )
        assert ours.pa_deg == pytest.approx(theirs.pa.to(u.deg).value, abs=1e-6)

    def test_deconvolve_matches_radio_beam(self):
        b1 = Beam.from_arcsec(15.0, 10.0, 45.0)
        b2 = Beam.from_arcsec(10.0, 8.0, 30.0)
        ours = b1.deconvolve(b2)
        theirs = _rb(15.0, 10.0, 45.0).deconvolve(_rb(10.0, 8.0, 30.0))
        assert ours.major_arcsec == pytest.approx(
            theirs.major.to(u.arcsec).value, rel=1e-5
        )
        assert ours.minor_arcsec == pytest.approx(
            theirs.minor.to(u.arcsec).value, rel=1e-5
        )

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
        expected = np.pi / (4 * np.log(2)) * fwhm_rad**2
        assert b.area_sr() == pytest.approx(expected, rel=1e-10)


# ── common_beam vs radio_beam ─────────────────────────────────────────────────


class TestCommonBeam:
    def test_two_beams_matches_radio_beam(self):
        params = [(10.0, 8.0, 30.0), (12.0, 6.0, 60.0)]
        beams = [Beam.from_arcsec(*p) for p in params]
        theirs = _rb(*params[0]).commonbeam_with(_rb(*params[1]))
        ours = common_beam(beams)
        assert ours.major_arcsec == pytest.approx(
            theirs.major.to(u.arcsec).value, rel=1e-4
        )
        assert ours.minor_arcsec == pytest.approx(
            theirs.minor.to(u.arcsec).value, rel=1e-4
        )

    def test_many_beams_matches_radio_beam(self):
        params = [
            (10.0, 8.0, 30.0),
            (12.0, 6.0, 60.0),
            (11.0, 9.0, 45.0),
            (9.0, 7.0, 15.0),
        ]
        beams = [Beam.from_arcsec(*p) for p in params]
        rb_beams = rb.Beams(
            major=[p[0] for p in params] * u.arcsec,
            minor=[p[1] for p in params] * u.arcsec,
            pa=[p[2] for p in params] * u.deg,
        )
        ours = common_beam(beams)
        theirs = rb_beams.common_beam()
        assert ours.major_arcsec == pytest.approx(
            theirs.major.to(u.arcsec).value, rel=1e-3
        )
        assert ours.minor_arcsec == pytest.approx(
            theirs.minor.to(u.arcsec).value, rel=1e-3
        )

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

    def test_circular_plus_symmetric_elongated(self):
        # Circular beam + two elongated beams at equal and opposite PAs.
        # Both implementations find a valid common beam but the iterative MVE
        # algorithm can converge to slightly different solutions, so we check
        # correctness properties rather than exact numerical agreement.
        beams = [Beam(10, 10, 0), Beam(20, 10, 30), Beam(20, 10, -30)]
        result = common_beam(beams)
        # Must be larger than the largest input beam
        assert result.major_deg >= max(b.major_deg for b in beams)
        # Must contain every input beam
        for b in beams:
            result.deconvolve(b)  # must not raise

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            common_beam([])


# ── smooth ────────────────────────────────────────────────────────────────────


class TestSmooth:
    """Flux-scaling behaviour of smooth() for different brightness units."""

    OLD = Beam(10 * ARCSEC, 10 * ARCSEC, 0.0)
    NEW = Beam(20 * ARCSEC, 20 * ARCSEC, 0.0)
    PIX = 2.5 * ARCSEC

    def _flat(self):
        return np.ones((32, 32), dtype=np.float32)

    def test_default_is_jy_per_beam(self):
        # Convolving a flat Jy/beam image scales it by Ω_new/Ω_old = 4.
        out = smooth(self._flat(), self.OLD, self.NEW, self.PIX, self.PIX)
        assert out[16, 16] == pytest.approx(4.0, abs=1e-3)

    def test_explicit_jy_per_beam(self):
        out = smooth(
            self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, bunit="Jy/beam"
        )
        assert out[16, 16] == pytest.approx(4.0, abs=1e-3)

    def test_kelvin_conserves_surface_brightness(self):
        # Brightness temperature: a flat image stays flat (no flux scaling).
        out = smooth(self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, bunit="K")
        assert out[16, 16] == pytest.approx(1.0, abs=1e-3)

    def test_unrecognised_bunit_warns_and_assumes_jy_per_beam(self):
        with pytest.warns(UserWarning, match="Could not determine brightness unit"):
            out = smooth(
                self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, bunit="furlongs"
            )
        assert out[16, 16] == pytest.approx(4.0, abs=1e-3)

    def test_recognised_bunit_does_not_warn(self):
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            smooth(self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, bunit="K")
            smooth(
                self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, bunit="Jy/beam"
            )


class TestSmoothErrors:
    OLD = Beam(10 * ARCSEC, 10 * ARCSEC, 0.0)
    NEW = Beam(20 * ARCSEC, 20 * ARCSEC, 0.0)
    PIX = 2.5 * ARCSEC

    def test_unreachable_target_raises(self):
        img = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="cannot be reached"):
            smooth(img, self.NEW, self.OLD, self.PIX, self.PIX)

    def test_zero_size_beam_raises(self):
        img = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="zero size"):
            smooth(img, Beam(0.0, 0.0, 0.0), self.NEW, self.PIX, self.PIX)

    def test_cutoff_raises(self):
        img = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="cutoff"):
            smooth(img, self.OLD, self.NEW, self.PIX, self.PIX, cutoff_arcsec=5.0)

    def test_all_nan_returned_unchanged(self):
        img = np.full((16, 16), np.nan, dtype=np.float32)
        out = smooth(img, self.OLD, self.NEW, self.PIX, self.PIX)
        assert np.isnan(out).all()

    def test_negative_cdelt_matches_positive(self):
        rng = np.random.default_rng(3)
        img = rng.normal(size=(40, 36)).astype(np.float32)
        old = Beam.from_arcsec(8.0, 5.0, 30.0)
        new = Beam.from_arcsec(16.0, 9.0, 70.0)
        pos = smooth(img, old, new, self.PIX, self.PIX)
        neg = smooth(img, old, new, -self.PIX, self.PIX)
        np.testing.assert_array_equal(pos, neg)


class TestSmoothDtype:
    """smooth() runs at the input precision and returns the matching dtype."""

    OLD = Beam(10 * ARCSEC, 10 * ARCSEC, 0.0)
    NEW = Beam(20 * ARCSEC, 20 * ARCSEC, 0.0)
    PIX = 2.5 * ARCSEC

    def test_float32_returns_float32(self):
        img = np.ones((32, 32), dtype=np.float32)
        out = smooth(img, self.OLD, self.NEW, self.PIX, self.PIX)
        assert out.dtype == np.float32
        assert out[16, 16] == pytest.approx(4.0, abs=1e-3)

    def test_float64_returns_float64(self):
        img = np.ones((32, 32), dtype=np.float64)
        out = smooth(img, self.OLD, self.NEW, self.PIX, self.PIX)
        assert out.dtype == np.float64
        assert out[16, 16] == pytest.approx(4.0, abs=1e-3)

    def test_float32_and_float64_agree(self):
        rng = np.random.default_rng(0)
        img64 = rng.normal(size=(48, 48))
        img32 = img64.astype(np.float32)
        out64 = smooth(img64, self.OLD, self.NEW, self.PIX, self.PIX)
        out32 = smooth(img32, self.OLD, self.NEW, self.PIX, self.PIX)
        np.testing.assert_allclose(out32, out64, atol=1e-3)

    def test_non_float_array_raises(self):
        img = np.ones((32, 32), dtype=np.int32)
        with pytest.raises(ValueError):
            smooth(img, self.OLD, self.NEW, self.PIX, self.PIX)


class TestSmoothInputs:
    """smooth() takes arrays as they come from FITS and from other threads."""

    OLD = Beam.from_arcsec(12.0, 10.0, 20.0)
    NEW = Beam.from_arcsec(18.0, 15.0, 35.0)
    PIX = 2.5 * ARCSEC

    def _image(self, shape=(65, 48), dtype=np.float32, nans=True):
        rng = np.random.default_rng(1234)
        img = rng.normal(size=shape).astype(dtype)
        img[shape[0] // 2 :, : shape[1] // 3] += 5.0
        if nans:
            img[:20, :25] = np.nan
        return img

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_big_endian_fits_data(self, tmp_path, dtype):
        # astropy hands back FITS data big-endian (">f4"/">f8"); it must be
        # accepted and give the same answer as the native-order array.
        path = tmp_path / "plane.fits"
        fits.PrimaryHDU(data=self._image(dtype=dtype)).writeto(path)
        data = fits.getdata(path)
        assert not data.dtype.isnative

        out = smooth(data, self.OLD, self.NEW, self.PIX, self.PIX)
        native = smooth(data.astype(dtype), self.OLD, self.NEW, self.PIX, self.PIX)
        assert out.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(out, native)

    def test_non_contiguous_input(self):
        base = self._image(shape=(96, 130))
        for view in (base.T, base[::2, 1::3], np.asfortranarray(base)):
            out = smooth(view, self.OLD, self.NEW, self.PIX, self.PIX)
            expected = smooth(
                np.ascontiguousarray(view), self.OLD, self.NEW, self.PIX, self.PIX
            )
            np.testing.assert_array_equal(out, expected)

    def test_input_is_left_alone(self):
        for nans in (False, True):
            img = self._image(nans=nans)
            before = img.copy()
            smooth(img, self.OLD, self.NEW, self.PIX, self.PIX)
            np.testing.assert_array_equal(img, before)

    def test_concurrent_threads_match_sequential(self):
        # The GIL is released during the convolution, so a thread-pool executor
        # (racs_tools' default) runs planes in parallel; results must not mix.
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        images = [self._image(shape=(200, 180 + k)) * (k + 1) for k in range(6)]
        expected = [smooth(im, self.OLD, self.NEW, self.PIX, self.PIX) for im in images]
        with ThreadPoolExecutor(max_workers=4) as pool:
            got = list(
                pool.map(
                    lambda im: smooth(im, self.OLD, self.NEW, self.PIX, self.PIX),
                    images,
                )
            )
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g, e)

    @pytest.mark.parametrize("shape", [(128, 96), (129, 97)])
    def test_matches_racs_tools_robust(self, shape):
        # Parity with the Python implementation this ports, when it is installed.
        convolve_uv = pytest.importorskip("racs_tools.convolve_uv")
        img = self._image(shape=shape)
        old = rb.Beam(12 * u.arcsec, 10 * u.arcsec, 20 * u.deg)
        new = rb.Beam(18 * u.arcsec, 15 * u.arcsec, 35 * u.deg)
        theirs = convolve_uv.smooth(
            img, old, new, self.PIX * u.deg, self.PIX * u.deg, conv_mode="robust"
        )
        ours = smooth(
            img,
            Beam.from_radio_beam(old),
            Beam.from_radio_beam(new),
            self.PIX,
            self.PIX,
        )
        # racs_tools wraps round the edges and does not normalise near NaNs,
        # so the two agree only beyond the kernel's reach of either.
        assert_matches_away_from_missing(
            ours, theirs, np.isnan(img), reach_px(18, self.PIX * 3600)
        )


class TestMissingData:
    """The sky beyond the edges and under NaN pixels is missing data."""

    PIX = 2.0 * ARCSEC
    OLD, NEW = Beam.from_arcsec(10, 8, 30), Beam.from_arcsec(24, 20, 60)

    def _flat(self):
        image = np.ones((96, 80))
        image[40:60, 30:50] = np.nan
        return image

    def test_flat_image_stays_flat(self):
        out = smooth(self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, bunit="K")
        kept = out[~np.isnan(out)]
        assert kept.size > 0.8 * out.size
        np.testing.assert_allclose(kept, 1.0, rtol=0, atol=1e-6)

    def test_corners_and_nan_interiors_are_blank(self):
        out = smooth(self._flat(), self.OLD, self.NEW, self.PIX, self.PIX)
        assert np.isnan(out[[0, 0, -1, -1], [0, -1, 0, -1]]).all()
        assert np.isnan(out[45:55, 35:45]).all()
        assert np.isfinite(out[0, 40])
        assert np.isfinite(out[48, 0])

    def test_nan_fraction_is_a_tolerance(self):
        blanked = [
            np.isnan(
                smooth(
                    self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, nan_fraction=f
                )
            ).sum()
            for f in (0.01, 0.5, 0.9)
        ]
        assert blanked[0] > blanked[1] > blanked[2] > 0
        with pytest.raises(ValueError, match="between 0 and 1"):
            smooth(self._flat(), self.OLD, self.NEW, self.PIX, self.PIX, nan_fraction=2)

    def test_no_flux_wraps_round(self):
        image = np.zeros((96, 80))
        image[48, 0] = 1.0
        out = smooth(image, self.OLD, self.NEW, self.PIX, self.PIX)
        assert abs(out[48, -1]) < 1e-9


class TestExactTaper:
    """smooth() against the closed-form answer: convolve the image's
    band-limited interpolant with the kernel exactly and resample. On an even
    axis that averages the taper over the Nyquist bin's +-1/2 cycles/pixel,
    which turns the cross term exp(b u v) into cosh(b u v) there."""

    PIX = 2.0 / 3600.0
    CASES = (
        ((9, 9, 0), (12.3, 9.1, 30)),
        ((16, 11, 40), (16.1, 15.1, 40.03)),
        ((10, 8, 30), (12.3, 9.1, 65.8)),
    )

    @staticmethod
    def _quadratic_form(beam, pix):
        """sigma^2 terms (a, b, c) of a beam in pixels, u along rows, v along
        columns, PA from the row axis, as racs_tools.gaussft writes them."""
        f = 2 * np.sqrt(2 * np.log(2))
        sx, sy = beam[0] / 3600 / pix / f, beam[1] / 3600 / pix / f
        t = np.deg2rad(beam[2])
        c, s = np.cos(t), np.sin(t)
        return np.array(
            [
                sx**2 * c**2 + sy**2 * s**2,
                2 * c * s * (sy**2 - sx**2),
                sx**2 * s**2 + sy**2 * c**2,
            ]
        )

    @staticmethod
    def _fast_len(n):
        """The smallest length >= n with no prime factor above 5."""
        while True:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 1

    def _exact(self, image, old, new):
        """The image padded beyond the kernel's 6.1 sigma reach, with the sky
        beyond it missing, through the exact taper on the padded grid. Kept
        pixels are divided by the share of the kernel on data; those with
        more than half on missing data are blanked."""
        difference = self._quadratic_form(new, self.PIX) - self._quadratic_form(
            old, self.PIX
        )
        a, b, c = -2 * np.pi**2 * difference
        cov = [[difference[0], difference[1] / 2], [difference[1] / 2, difference[2]]]
        reach = int(np.ceil(6.1 * np.sqrt(np.linalg.eigvalsh(cov).max()))) + 1
        ny, nx = image.shape
        py, px = self._fast_len(ny + reach), self._fast_len(nx + reach)
        ratio = new[0] * new[1] / (old[0] * old[1])
        u = np.fft.fftfreq(py)[:, None]
        v = np.fft.fftfreq(px)[None, :]
        cross = np.exp(b * u * v)
        nyquist = np.zeros((py, px), dtype=bool)
        if py % 2 == 0:
            nyquist[py // 2, :] = True
        if px % 2 == 0:
            nyquist[:, px // 2] = True
        cross = np.where(nyquist, np.cosh(b * u * v), cross)
        taper = np.exp(a * u**2 + c * v**2) * cross

        def convolve(plane):
            out = np.fft.ifft2(np.fft.fft2(plane) * taper)
            assert np.abs(out.imag).max() < 1e-12 * max(np.abs(out.real).max(), 1)
            return out.real[:ny, :nx]

        padded = np.zeros((py, px))
        padded[:ny, :nx] = image
        missing = np.ones((py, px))
        missing[:ny, :nx] = 0
        data, m = ratio * convolve(padded), convolve(missing)
        exact = data / (1 - m)
        exact[m > 0.5] = np.nan
        return exact

    @pytest.mark.parametrize("shape", [(64, 64), (64, 65), (65, 64), (65, 65)])
    @pytest.mark.parametrize(("old", "new"), CASES)
    def test_matches_closed_form(self, shape, old, new):
        image = np.random.default_rng(2).normal(size=shape)
        ours = smooth(
            image,
            Beam.from_arcsec(*old),
            Beam.from_arcsec(*new),
            self.PIX,
            self.PIX,
        )
        exact = self._exact(image, old, new)
        # convolve-rs leaves alone pixels with under 1e-8 of their kernel on
        # missing data, where the reference still divides.
        np.testing.assert_allclose(
            ours, exact, rtol=0, atol=2e-8 * np.nanmax(np.abs(exact))
        )

    @pytest.mark.parametrize(("old", "new"), CASES)
    def test_mirror_symmetric(self, old, new):
        # Mirroring the image and its beams' PAs mirrors the result. Taking one
        # sign of the Nyquist bin, as before, broke this by up to 15% of the noise.
        image = np.random.default_rng(3).normal(size=(64, 64))
        mirror = lambda b: Beam.from_arcsec(b[0], b[1], -b[2])  # noqa: E731
        direct = smooth(
            image, Beam.from_arcsec(*old), Beam.from_arcsec(*new), self.PIX, self.PIX
        )
        mirrored = smooth(
            image[:, ::-1].copy(), mirror(old), mirror(new), self.PIX, self.PIX
        )[:, ::-1]
        np.testing.assert_allclose(
            mirrored, direct, rtol=0, atol=1e-12 * np.nanstd(direct)
        )
