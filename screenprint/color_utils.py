"""Color space utilities for screen print color detection.

Provides OKLab color space conversions and color distance calculations.
"""

import numpy as np
from numpy.typing import NDArray


def srgb_to_linear(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert sRGB values (0-1) to linear RGB.

    Uses the standard sRGB transfer function.
    """
    # Threshold for linear portion
    threshold = 0.04045

    linear = np.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear


def linear_to_srgb(linear: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert linear RGB to sRGB (0-1)."""
    threshold = 0.0031308

    srgb = np.where(
        linear <= threshold,
        linear * 12.92,
        1.055 * (linear ** (1/2.4)) - 0.055
    )
    return np.clip(srgb, 0, 1)


def linear_rgb_to_oklab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert linear RGB to OKLab color space.

    OKLab is a perceptually uniform color space that works well for
    measuring color differences as humans perceive them.

    Args:
        rgb: Array of shape (..., 3) with linear RGB values in [0, 1]

    Returns:
        Array of shape (..., 3) with OKLab values (L in [0,1], a/b approximately [-0.4, 0.4])
    """
    # Ensure input is float
    rgb = np.asarray(rgb, dtype=np.float64)

    # Matrix to convert linear sRGB to LMS (cone responses)
    # Using the OKLab M1 matrix
    m1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])

    # Matrix to convert LMS' to OKLab
    m2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])

    # Apply first matrix to get LMS
    lms = np.einsum('...j,ij->...i', rgb, m1)

    # Apply cube root (non-linearity)
    lms_prime = np.cbrt(lms)

    # Apply second matrix to get OKLab
    oklab = np.einsum('...j,ij->...i', lms_prime, m2)

    return oklab


def oklab_to_linear_rgb(oklab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert OKLab to linear RGB.

    Args:
        oklab: Array of shape (..., 3) with OKLab values

    Returns:
        Array of shape (..., 3) with linear RGB values (may be outside [0,1] for out-of-gamut colors)
    """
    oklab = np.asarray(oklab, dtype=np.float64)

    # Inverse of m2
    m2_inv = np.array([
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480]
    ])

    # Inverse of m1
    m1_inv = np.array([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010]
    ])

    # Apply inverse m2 to get LMS'
    lms_prime = np.einsum('...j,ij->...i', oklab, m2_inv)

    # Cube to get LMS
    lms = lms_prime ** 3

    # Apply inverse m1 to get linear RGB
    rgb = np.einsum('...j,ij->...i', lms, m1_inv)

    return rgb


def srgb_to_oklab(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert sRGB (0-255 or 0-1) to OKLab.

    Args:
        srgb: Array of shape (..., 3) with sRGB values

    Returns:
        Array of shape (..., 3) with OKLab values
    """
    srgb = np.asarray(srgb, dtype=np.float64)

    # Normalize to 0-1 if values appear to be in 0-255 range
    if srgb.max() > 1.0:
        srgb = srgb / 255.0

    linear = srgb_to_linear(srgb)
    return linear_rgb_to_oklab(linear)


def oklab_to_srgb(oklab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert OKLab to sRGB (0-1).

    Args:
        oklab: Array of shape (..., 3) with OKLab values

    Returns:
        Array of shape (..., 3) with sRGB values in [0, 1]
    """
    linear = oklab_to_linear_rgb(oklab)
    return linear_to_srgb(linear)


def oklab_distance(color1: NDArray[np.float64], color2: NDArray[np.float64]) -> float:
    """Calculate Euclidean distance in OKLab space.

    This gives a perceptually uniform measure of color difference.
    A distance of ~0.02-0.03 is roughly a just-noticeable difference.
    """
    return float(np.linalg.norm(color1 - color2))


def batch_oklab_distance(colors: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate distances from multiple colors to a reference color in OKLab space.

    Args:
        colors: Array of shape (N, 3) with OKLab colors
        reference: Array of shape (3,) with reference OKLab color

    Returns:
        Array of shape (N,) with distances
    """
    return np.linalg.norm(colors - reference, axis=1)
