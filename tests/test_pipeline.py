"""The file-level API against racs_tools' beamcon_2D and beamcon_3D."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

import convolve_rs as cr

if TYPE_CHECKING:
    from pathlib import Path

    from convolve_rs._convolve_rs import Beam as _Beam

PIX = 2.0 / 3600.0


def _header(bmaj: float, bmin: float, bpa: float, *, cube: bool = False) -> fits.Header:
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---SIN", "DEC--SIN"
    h["CDELT1"], h["CDELT2"] = -PIX, PIX
    h["CRPIX1"], h["CRPIX2"] = 33.0, 33.0
    h["CRVAL1"], h["CRVAL2"] = 0.0, -30.0
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    h["BUNIT"] = "Jy/beam"
    h["BMAJ"], h["BMIN"], h["BPA"] = bmaj / 3600, bmin / 3600, bpa
    if cube:
        h["CTYPE3"], h["CRPIX3"], h["CRVAL3"], h["CDELT3"] = "FREQ", 1.0, 1e9, 1e6
        h["CUNIT3"] = "Hz"
    return h


def _sky(beam, seed=0) -> np.ndarray:
    """A 64x64 noise image at resolution ``beam`` (arcsec, arcsec, deg), as a
    real image is: white noise holds power at the Nyquist frequency that no
    image made with a beam has."""
    white = np.random.default_rng(seed).normal(size=(64, 64))
    point = cr.Beam.from_arcsec(1e-3, 1e-3, 0.0)
    target = cr.Beam.from_arcsec(*beam)
    return cr.smooth(white, point, target, PIX, PIX, bunit="K").astype(np.float32)


def _image(path: Path, beam, seed=0) -> Path:
    fits.writeto(path, _sky(beam, seed), _header(*beam), overwrite=True)
    return path


def _cube(path: Path, beams) -> Path:
    """A cube with a CASA BEAMS table of ``beams`` (arcsec, arcsec, deg)."""
    data = np.stack([_sky(b, seed=i) for i, b in enumerate(beams)])[None]
    data[0, 1, 10:20, 10:20] = np.nan
    header = _header(*beams[0], cube=True)
    header["CTYPE4"], header["CRPIX4"], header["CRVAL4"], header["CDELT4"] = (
        "STOKES",
        1.0,
        1.0,
        1.0,
    )
    header["CASAMBM"] = True
    bmaj, bmin, bpa = (
        np.array(col, dtype=np.float32) for col in zip(*beams, strict=True)
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="BMAJ", format="1E", unit="arcsec", array=bmaj),
            fits.Column(name="BMIN", format="1E", unit="arcsec", array=bmin),
            fits.Column(name="BPA", format="1E", unit="deg", array=bpa),
            fits.Column(name="CHAN", format="1J", array=np.arange(len(beams))),
            fits.Column(name="POL", format="1J", array=np.zeros(len(beams))),
        ],
        name="BEAMS",
    )
    table.header["NCHAN"], table.header["NPOL"] = len(beams), 1
    fits.HDUList([fits.PrimaryHDU(data, header), table]).writeto(path, overwrite=True)
    return path


def _assert_close(ours: np.ndarray, theirs: np.ndarray) -> None:
    """The tolerances of test_python's racs_tools parity test: convolve-rs
    blanks a thin extra rim at NaN edges, and otherwise matches."""
    assert np.isnan(ours)[np.isnan(theirs)].all()
    assert (
        np.isnan(ours).sum() - np.isnan(theirs).sum()
        <= max(0.02 * np.isnan(theirs).sum(), 0) + 4 * 64
    )
    finite = ~np.isnan(ours)
    np.testing.assert_allclose(
        ours[finite], theirs[finite], rtol=0, atol=1e-4 * np.nanstd(theirs)
    )


def _same_beam(ours: _Beam | None, theirs, *, fitted: bool = False) -> None:
    """Equal beams; a ``fitted`` common beam to within the minimum-volume
    ellipse fit, which lands up to one 0.1 arcsec rounding step from
    radio_beam's in the axes, and a little off in PA. A PA error moves the
    beam's outline in proportion to its ellipticity, so that is what's bounded."""
    assert ours is not None
    axes = 0.1 + 1e-9 if fitted else 1e-6
    major, minor = theirs.major.to_value(u.arcsec), theirs.minor.to_value(u.arcsec)
    assert ours.major_arcsec == pytest.approx(major, abs=axes)
    assert ours.minor_arcsec == pytest.approx(minor, abs=axes)
    d_pa = np.deg2rad((ours.pa_deg - theirs.pa.to_value(u.deg) + 90) % 180 - 90)
    assert abs(d_pa) * (1 - (minor / major) ** 2) <= (2e-3 if fitted else 1e-8)


BEAMS_2D = [(10.0, 8.0, 30.0), (12.0, 7.0, 75.0), (9.0, 9.0, 0.0)]


class TestSmoothImages:
    def _inputs(self, d: Path) -> list[Path]:
        d.mkdir()
        return [_image(d / f"im{i}.fits", b, seed=i) for i, b in enumerate(BEAMS_2D)]

    def test_matches_racs_tools(self, tmp_path):
        pytest.importorskip("racs_tools")
        beamcon_2D = pytest.importorskip("racs_tools.beamcon_2D")
        ref_in, rs_in = self._inputs(tmp_path / "ref"), self._inputs(tmp_path / "rs")
        ref_beam, _ = beamcon_2D.get_common_beam(ref_in)
        for f in ref_in:
            beamcon_2D.beamcon_2d_on_fits(f, ref_beam, "robust", suffix="sm")
        _same_beam(cr.smooth_images(rs_in, dryrun=True).target, ref_beam, fitted=True)
        # The same target for both, so the smoothing itself is compared
        result = cr.smooth_images(rs_in, target_beam=cr.Beam.from_radio_beam(ref_beam))
        _same_beam(result.target, ref_beam)
        for ref, img in zip(ref_in, result.images, strict=True):
            assert img.output == img.input.with_suffix(".sm.fits")
            theirs = fits.getdata(ref.with_suffix(".sm.fits"))
            _assert_close(fits.getdata(img.output), theirs)
            assert fits.getheader(img.output)["BMAJ"] == pytest.approx(
                result.target.major_deg
            )

    def test_blanked_image_gets_a_zero_beam(self, tmp_path):
        files = self._inputs(tmp_path / "in")
        result = cr.smooth_images(files, cutoff=11.0)
        blanked = result.images[1]
        assert blanked.kernel is None
        assert np.isnan(fits.getdata(blanked.output)).all()
        h = fits.getheader(blanked.output)
        assert (h["BMAJ"], h["BMIN"], h["BPA"]) == (0.0, 0.0, 0.0)
        assert result.images[0].kernel is not None

    def test_target_beam_dryrun_and_names(self, tmp_path):
        files = self._inputs(tmp_path / "in")
        target = cr.Beam.from_arcsec(20.0, 20.0, 0.0)
        result = cr.smooth_images(
            files, target_beam=target, dryrun=True, prefix="p.", suffix="x"
        )
        assert result.target.major_arcsec == pytest.approx(20.0)
        assert [i.output.name for i in result.images] == [
            f"p.im{i}.x.fits" for i in range(3)
        ]
        assert not any(i.output.exists() for i in result.images)

    def test_errors(self, tmp_path):
        files = self._inputs(tmp_path / "in")
        with pytest.raises(FileNotFoundError):
            cr.smooth_images([tmp_path / "missing.fits"])
        with pytest.raises(ValueError, match="too small"):
            cr.smooth_images(files, target_beam=cr.Beam.from_arcsec(5, 5, 0))
        with pytest.raises(ValueError, match="ncores"):
            cr.smooth_images(files, ncores=0, dryrun=True)


BEAMS_3D = [
    (14.0, 12.0, 10.0),
    (16.0, 11.0, 40.0),
    (15.0, 15.0, 0.0),
    (13.0, 9.0, 80.0),
]


class TestSmoothCubes:
    @pytest.mark.parametrize("mode", ["natural", "total"])
    def test_matches_racs_tools(self, tmp_path, mode):
        pytest.importorskip("racs_tools")
        beamcon_3D = pytest.importorskip("racs_tools.beamcon_3D")
        (tmp_path / "ref").mkdir()
        (tmp_path / "rs").mkdir()
        ref_in = _cube(tmp_path / "ref" / "c.fits", BEAMS_3D)
        rs_in = _cube(tmp_path / "rs" / "c.fits", BEAMS_3D)
        _, common, outs = beamcon_3D.smooth_fits_cube(
            [ref_in], mode=mode, suffix="sm", ncores=1
        )
        ref_targets = list(common[0].commonbeams)
        (solved,) = cr.smooth_cubes([rs_in], mode=mode, dryrun=True)
        for ours, theirs in zip(solved.targets, ref_targets, strict=True):
            _same_beam(ours, theirs, fitted=True)
        # The same targets for both, so the smoothing itself is compared
        (cube,) = cr.smooth_cubes(
            [rs_in],
            mode=mode,
            target_beams=[cr.Beam.from_radio_beam(b) for b in ref_targets],
        )
        assert cube.output == rs_in.with_suffix(".sm.fits")
        for ours, theirs in zip(cube.targets, ref_targets, strict=True):
            _same_beam(ours, theirs)
        _assert_close(fits.getdata(cube.output), fits.getdata(outs[0]))
        if mode == "total":
            # racs_tools keeps the input's CASAMBM = T but writes no BEAMS table
            ours_h, theirs_h = fits.getheader(cube.output), fits.getheader(outs[0])
            for key in ("BMAJ", "BMIN"):
                assert ours_h[key] == pytest.approx(theirs_h[key], rel=1e-6)
            assert not ours_h["CASAMBM"]
            return
        for ours, theirs in zip(
            cr.cube_beams(cube.output), cr.cube_beams(outs[0]), strict=True
        ):
            assert ours is not None
            assert theirs is not None
            assert ours.major_arcsec == pytest.approx(theirs.major_arcsec, abs=1e-4)

    def test_cube_beams_matches_racs_tools(self, tmp_path):
        beamcon_3D = pytest.importorskip("racs_tools.beamcon_3D")
        path = _cube(tmp_path / "c.fits", BEAMS_3D)
        (data,) = beamcon_3D.make_data(files=[path], outdir=[tmp_path])
        for ours, theirs in zip(cr.cube_beams(path), data.beams, strict=True):
            _same_beam(ours, theirs)

    def test_cutoff_channel_is_masked(self, tmp_path):
        path = _cube(tmp_path / "c.fits", BEAMS_3D)
        (cube,) = cr.smooth_cubes(
            [path], target_beam=cr.Beam.from_arcsec(30, 30, 0), cutoff=15.5
        )
        data = fits.getdata(cube.output)[0]
        assert np.isnan(data[1]).all()
        assert np.isfinite(data[0]).all()
        beams = cr.cube_beams(cube.output)
        assert beams[1] is None
        assert all(b is not None for i, b in enumerate(beams) if i != 1)

    def test_per_channel_targets_and_uselogs(self, tmp_path):
        path = _cube(tmp_path / "c.fits", BEAMS_3D)
        targets = [cr.Beam.from_arcsec(20 + i, 20, 0) for i in range(3)] + [None]
        (cube,) = cr.smooth_cubes([path], target_beams=targets, ref_chan="first")
        assert [b and round(b.major_arcsec, 6) for b in cube.targets] == [
            20.0,
            21.0,
            22.0,
            None,
        ]
        assert np.isnan(fits.getdata(cube.output)[0, 3]).all()
        assert fits.getheader(cube.output)["BMAJ"] == pytest.approx(20 / 3600)
        (again,) = cr.smooth_cubes([path], uselogs=True, dryrun=True)
        assert [b and round(b.major_arcsec, 6) for b in again.targets] == [
            20.0,
            21.0,
            22.0,
            None,
        ]

    def test_bad_options(self, tmp_path):
        path = _cube(tmp_path / "c.fits", BEAMS_3D)
        beam = cr.Beam.from_arcsec(30, 30, 0)
        with pytest.raises(ValueError, match="mode"):
            cr.smooth_cubes([path], mode="median")
        with pytest.raises(ValueError, match="ref_chan"):
            cr.smooth_cubes([path], ref_chan="centre")
        with pytest.raises(ValueError, match="uselogs"):
            cr.smooth_cubes([path], uselogs=True, target_beam=beam)
        with pytest.raises(ValueError, match="channels"):
            cr.smooth_cubes([path], target_beams=[beam])
