# Third-party notices and acknowledgements

`convolve-rs` is licensed under the BSD 3-Clause License (see [LICENSE](LICENSE)).
It builds on prior work in the radio-astronomy community. This file records the
provenance of that work and reproduces the required upstream license notices.

---

## RACS-tools (BSD 3-Clause)

The UV-plane (FFT) convolution and Gaussian-Fourier-transform routines
(`src/convolve_uv.rs`, `src/smooth.rs`) and the spectral-cube / CASAMBM beam-table
handling (`src/cube_io.rs`) are ports of algorithms from **RACS-tools**
(`racs_tools.convolve_uv`, `racs_tools.gaussft`):

  https://github.com/alecthomson/RACS-tools

RACS-tools is distributed under the BSD 3-Clause License:

> Copyright (c) 2020, Alec Thomson
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> [BSD 3-Clause terms — see LICENSE for the full text.]

---

## radio_beam (BSD 3-Clause)

The common-beam computation (`src/common_beam.rs`) — minimum-enclosing-ellipse
(Khachiyan algorithm), ellipse-transform helpers, and the `method='pts'` common
beam — is an independent implementation of the approach used by **radio_beam**
(`radio_beam.commonbeam`, `radio_beam.utils`):

  https://github.com/radio-astro-tools/radio-beam

radio_beam is distributed under the BSD 3-Clause License:

> Copyright (c) 2016, radio-astro-tools developers
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> [BSD 3-Clause terms — see LICENSE for the full text.]

---

## MIRIAD (GPL) — reference only, no code included

MIRIAD is distributed under the **GNU General Public License**:

  https://github.com/csiro/miriad
  https://www.atnf.csiro.au/computing/software/miriad/

The Gaussian beam algebra in `src/beam.rs` (convolution, deconvolution, and the
Jy/beam flux-scaling factor) is an **independent implementation** of the standard
second-moment / covariance formulation of an elliptical Gaussian (see Wild 1970,
*Aust. J. Phys.* 23, 113, https://ui.adsabs.harvard.edu/abs/1970AuJPh..23..113W). It expresses each beam as a 2×2 covariance matrix and
combines beams by matrix addition/subtraction and eigendecomposition. These are
textbook results, used identically across the field (including by `radio_beam`
and RACS-tools, both BSD-licensed). **No MIRIAD source code is copied, ported, or
linked**, so this project carries no GPL obligation from MIRIAD.

MIRIAD is acknowledged because its `convol` task and `gaupar.for` documentation
were used as a numerical *reference and validation target*: the test suite
(`tests/miriad_compat.rs`) optionally shells out to the separately-installed
MIRIAD `convol`/`fits` binaries to confirm bit-comparable results. Running those
binaries as external processes during testing does not incorporate MIRIAD into
this software, and no MIRIAD code or binary is bundled or distributed here.
