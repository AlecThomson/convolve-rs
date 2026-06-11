# Algorithm background

This page summarises the maths implemented in convolve-rs. Everything here is
a standard result for combining Gaussian point-spread functions — see Wild
(1970, *Aust. J. Phys.* 23, 113) for the radio-astronomy form. The same
formulae underpin
[radio_beam](https://github.com/radio-astro-tools/radio-beam) and
[RACS-tools](https://github.com/alecthomson/RACS-tools); convolve-rs is an
independent implementation in terms of the covariance matrix and its
eigendecomposition.

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

Rather than convolving with an image-domain kernel — which becomes numerically
unreliable when the convolving kernel is undersampled (comparable to or
smaller than a pixel) — convolve-rs works in the Fourier (UV) plane, following
the "robust" mode of RACS-tools:

1. FFT the input image.
2. Multiply by the analytic Fourier transform of the convolving Gaussian,
   evaluated exactly at each UV point (no kernel image is ever constructed).
3. Inverse FFT.

NaN pixels are handled by zero-filling the data, convolving a NaN mask through
the same filter, and re-blanking every output pixel where the smeared mask
reaches 1.

Because the image is real-valued, a real-input FFT is used along the
contiguous axis, halving spectrum memory — the dominant cost for large images.

## Common beam

Given a set of beams (e.g. per channel of a cube), the *common beam* is the
smallest beam that every input can be convolved to. Two algorithms are used,
matching `radio_beam.Beams.common_beam(method='pts')`:

- **Two beams**: the analytic CASA algorithm (`ia.commonbeam`): transform to a
  frame where the larger beam is circular, take the enclosing ellipse, and
  transform back.
- **Many beams**: sample points on each beam-ellipse boundary, take the convex
  hull, and find the minimum-volume enclosing ellipse with the Khachiyan
  algorithm. The ellipses are inflated by a small `epsilon` first so the
  result can be marginally deconvolved from every input; if validation fails,
  `epsilon` is increased and the fit retried.

Exposed as {func}`convolve_rs.common_beam`.
