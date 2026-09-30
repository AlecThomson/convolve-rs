"""Peak memory and wall time: racs_tools' robust smooth vs convolve_rs.smooth.

Needs both packages (and radio-beam) installed in the running Python. Each case
runs in a fresh subprocess, so ``ru_maxrss`` is that case's own peak. Memory is
reported as (peak RSS - RSS just before the call) / image bytes: the working
memory on top of the caller's input array. The image is built without
temporaries so they cannot inflate the baseline, and numba's JIT for
racs_tools' ``gaussft`` is compiled before the timed call.

Usage::

    python scripts/bench_vs_racs_tools.py [NYxNX ...]   # default 4500x3900 9000x7800
"""

from __future__ import annotations

import json
import subprocess
import sys

CASE = r"""
import gc, json, resource, sys, time
import astropy.units as u
import numpy as np
from radio_beam import Beam as RadioBeam

impl, ny, nx, nans, dtype = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
img = np.empty((ny, nx), dtype=dtype)
np.random.default_rng(1).standard_normal(out=img, dtype=img.dtype)
if nans:
    img[: int(ny * 0.65), : int(nx * 0.65)] = np.nan  # ~42% blanked
old = RadioBeam(12 * u.arcsec, 10 * u.arcsec, 20 * u.deg)
new = RadioBeam(18 * u.arcsec, 15 * u.arcsec, 35 * u.deg)
pix = 2.5 / 3600
if impl == "racs_tools":
    from racs_tools.convolve_uv import smooth
    def run(im):
        return smooth(im, old, new, pix * u.deg, pix * u.deg, conv_mode="robust")
    run(np.ones((64, 64), dtype=img.dtype))  # compile numba outside the timing
else:
    import convolve_rs
    ob, nb = convolve_rs.Beam.from_radio_beam(old), convolve_rs.Beam.from_radio_beam(new)
    def run(im):
        return convolve_rs.smooth(im, ob, nb, pix, pix)
gc.collect()
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
t0 = time.perf_counter()
run(img)
dt = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
print(json.dumps({"t": dt, "mem": (peak - base) / img.nbytes}))
"""


def run_case(impl: str, ny: int, nx: int, nans: bool, dtype: str) -> dict:
    """Run one benchmark case in a fresh interpreter."""
    args = [sys.executable, "-c", CASE, impl, str(ny), str(nx), str(int(nans)), dtype]
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    shapes = [tuple(map(int, s.split("x"))) for s in sys.argv[1:]] or [
        (4500, 3900),
        (9000, 7800),
    ]
    print(
        f"{'shape':>11} {'dtype':>8} {'NaN':>4} | {'racs_tools':>16} "
        f"| {'convolve_rs':>16} | speed-up"
    )
    for ny, nx in shapes:
        for dtype in ("float32", "float64"):
            for nans in (False, True):
                py = run_case("racs_tools", ny, nx, nans, dtype)
                rs = run_case("convolve_rs", ny, nx, nans, dtype)
                print(
                    f"{f'{ny}x{nx}':>11} {dtype:>8} {'yes' if nans else 'no':>4} "
                    f"| {py['t']:7.2f} s {py['mem']:5.2f}x "
                    f"| {rs['t']:7.2f} s {rs['mem']:5.2f}x "
                    f"| {py['t'] / rs['t']:5.1f}x"
                )


if __name__ == "__main__":
    main()
