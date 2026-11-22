# Algorithm background

The beam algebra is the standard one for Gaussian point-spread functions
(Wild 1970, *Aust. J. Phys.* 23, 113), as used by
[radio_beam](https://github.com/radio-astro-tools/radio-beam) and
[RACS-tools](https://github.com/alecthomson/RACS-tools).

## Beam algebra

A restoring beam is a 2D elliptical Gaussian, stored in FITS conventions:
FWHM major/minor axes (`BMAJ`/`BMIN`, degrees) and position angle (`BPA`,
degrees East of North).

An elliptical Gaussian is fully described by a symmetric $2 \times 2$
covariance matrix in (East, North) axes:

$$
C = \begin{pmatrix} C_{xx} & C_{xy} \\ C_{xy} & C_{yy} \end{pmatrix},
$$

whose eigenvalues are the squared axis lengths and whose eigenvectors give the
orientation. For FWHM axes $a, b$ and position angle $\theta$:

$$
\begin{aligned}
C_{xx} &= a^2 \sin^2\theta + b^2 \cos^2\theta, \\
C_{yy} &= a^2 \cos^2\theta + b^2 \sin^2\theta, \\
C_{xy} &= (a^2 - b^2) \sin\theta \cos\theta.
\end{aligned}
$$

In this representation the beam operations are linear:

- **Convolution** adds covariance matrices: $C = C_1 + C_2$.
- **Deconvolution** subtracts them: $C = C_1 - C_2$, valid only while the
  residual stays positive-definite. If it does not, the source beam is smaller
  than the PSF and deconvolution fails.

A target beam can be reached from an input beam when the residual is positive
semi-definite. The convolving kernel may then be a line (e.g. 20″×10″ to
20″×20″), which the UV-plane filter handles without trouble.

The resulting axes and position angle are read off the eigen-pairs of $C$.

## Flux scaling

The integral of a 2D Gaussian is proportional to $\sqrt{\det C}$. For an image
in Jy/beam, convolving from beam area $\Omega_\text{old}$ to
$\Omega_\text{new}$ rescales pixel values by the beam-area ratio
$\Omega_\text{new}/\Omega_\text{old}$ so the map stays in Jy/beam. For images
in Kelvin (brightness temperature), surface brightness is conserved under
convolution and no scaling is applied.

The peak amplitude of the convolving Gaussian needed to preserve Jy/beam units
is

$$
A = \frac{\pi}{4 \ln 2}
    \sqrt{\frac{\det C_\text{orig} \, \det C_\text{conv}}
               {\det (C_\text{orig} + C_\text{conv})}},
$$

implemented in {func}`convolve_rs.gauss_factor`.

## UV-plane convolution

An image-domain kernel is unreliable when it is comparable to or smaller than
a pixel, so convolve-rs works in the Fourier (UV) plane, following
the "robust" mode of RACS-tools:

1. FFT the input image.
2. Multiply by the analytic Fourier transform of the convolving Gaussian,
   evaluated exactly at each UV point (no kernel image is ever constructed).
3. Inverse FFT.

The image is padded first (see below), and the result is exactly the padded
image's band-limited interpolant convolved with the kernel and resampled. On
an even-sized axis that needs one more step: the
Nyquist bin holds +1/2 and -1/2 cycles per pixel at once, and a rotated kernel
tapers the two differently. Writing the taper as
`r exp(a u² + b u v + c v²)`, the exact value there is their average, which
replaces the cross term `exp(b u v)` with `cosh(b u v)` on the Nyquist row and
column. Taking either sign alone leaves a result that changes when the image
is mirrored. On images already at their beam the Nyquist bin holds almost no
power and the difference is far below the noise, but it shows at sharp edges
such as blanked regions, and on under-sampled beams.

The UV coordinates come from the pixel grid. With CDELTs alone, as RACS-tools
does, the grid is taken to be unrotated with east to the left. Given the full
pixel-to-sky matrix `M` (CD, or PC and CDELT; astropy's
`WCS.pixel_scale_matrix`), the kernel's sky covariance `Σ_new − Σ_old` becomes
`M⁻¹ (Σ_new − Σ_old) M⁻ᵀ` in pixels, and the taper is evaluated in cycles per
pixel. That covers rotated and skewed grids, and grids with east to the right.
The file functions and the CLI always read the full matrix; for the usual grid
it takes the CDELT path, so its output is unchanged.

## Edges and missing data

The sky beyond an image's edges is unknown, as is the sky under its NaN
pixels, and both are treated the same way: as missing data.

- **Edges.** The image is padded by the kernel's reach (6.1 standard
  deviations), so the convolution is linear. A transform of the image alone
  would be circular, carrying flux off one edge onto the opposite one, as
  RACS-tools 4.6 and convolve-rs before 3.0 did.
- **Normalisation.** Missing pixels are filled with zeros, and a mask of them
  is convolved through the same filter. Each output pixel is divided by the
  share of its kernel's weight that fell on data. Without that, a pixel near
  missing data is too faint by the share that fell on it: zero filling alone
  is biased.
- **Blanking.** A pixel with more than `nan_fraction` (default 0.5) of its
  kernel's weight on missing data is mostly extrapolated, and is blanked. The
  corners of an image, with about three quarters of their kernel beyond it,
  are blanked; so is anything deep enough in a NaN region.

So a flat image stays flat up to its edges and its NaN regions.
`nan_fraction` is a tolerance: lower it to keep only pixels whose kernel
lies almost all on data, at the cost of a wider blanked margin.

A kernel narrower than about a pixel is cut off at the Nyquist frequency, so
it rings, and the missing-data share it gives can dip a little below zero
near an edge. Those pixels are divided all the same, which keeps a flat image
flat.

Only the part of the mask near missing data is convolved, which gives the
same shares at a fraction of the cost.

Because the image is real-valued, a real-input FFT is used along the
contiguous axis, which halves the spectrum's memory.

## Common beam

Given a set of beams (e.g. per channel of a cube), the *common beam* is the
smallest beam that every input can be convolved to. Two algorithms are used,
matching `radio_beam.Beams.common_beam(method='pts')`:

- **Two beams**: the analytic CASA algorithm (`ia.commonbeam`): transform to a
  frame where the larger beam is circular, take the enclosing ellipse, and
  transform back. If the result does not contain both beams, the many-beam
  fit is used instead.
- **Many beams**: sample points on each beam-ellipse boundary, take the convex
  hull, and find the minimum-volume enclosing ellipse with the Khachiyan
  algorithm. The ellipses are inflated by a small `epsilon` first so the
  result can be marginally deconvolved from every input; if validation fails,
  `epsilon` is increased and the fit retried.

Exposed as {func}`convolve_rs.common_beam`.
