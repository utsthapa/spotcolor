"""Tests for color space conversion utilities."""

import pytest
import numpy as np
from screenprint.color_utils import (
    srgb_to_linear,
    linear_to_srgb,
    srgb_to_oklab,
    oklab_to_srgb,
    linear_rgb_to_oklab,
    oklab_to_linear_rgb,
    oklab_distance,
    batch_oklab_distance
)


class TestSRGBLinearConversion:
    """Tests for sRGB to linear RGB conversion."""

    def test_srgb_to_linear_black(self):
        """Test conversion of black (0, 0, 0)."""
        srgb = np.array([0.0, 0.0, 0.0])
        linear = srgb_to_linear(srgb)
        np.testing.assert_allclose(linear, [0.0, 0.0, 0.0])

    def test_srgb_to_linear_white(self):
        """Test conversion of white (1, 1, 1)."""
        srgb = np.array([1.0, 1.0, 1.0])
        linear = srgb_to_linear(srgb)
        np.testing.assert_allclose(linear, [1.0, 1.0, 1.0])

    def test_linear_to_srgb_roundtrip(self):
        """Test that linear->srgb->linear conversion is reversible."""
        original = np.array([0.5, 0.3, 0.7])
        srgb = linear_to_srgb(original)
        recovered = srgb_to_linear(srgb)
        np.testing.assert_allclose(recovered, original, rtol=1e-5)

    def test_srgb_to_linear_roundtrip(self):
        """Test that srgb->linear->srgb conversion is reversible."""
        original = np.array([0.5, 0.3, 0.7])
        linear = srgb_to_linear(original)
        recovered = linear_to_srgb(linear)
        np.testing.assert_allclose(recovered, original, rtol=1e-5)


class TestOKLabConversion:
    """Tests for OKLab color space conversion."""

    def test_srgb_to_oklab_white(self):
        """Test conversion of white to OKLab."""
        srgb = np.array([1.0, 1.0, 1.0])
        oklab = srgb_to_oklab(srgb)
        # White should have L=1, a=0, b=0
        assert oklab[0] == pytest.approx(1.0, abs=0.01)
        assert oklab[1] == pytest.approx(0.0, abs=0.01)
        assert oklab[2] == pytest.approx(0.0, abs=0.01)

    def test_srgb_to_oklab_black(self):
        """Test conversion of black to OKLab."""
        srgb = np.array([0.0, 0.0, 0.0])
        oklab = srgb_to_oklab(srgb)
        # Black should have L=0, a=0, b=0
        np.testing.assert_allclose(oklab, [0.0, 0.0, 0.0], atol=0.01)

    def test_srgb_to_oklab_red(self):
        """Test conversion of red to OKLab."""
        srgb = np.array([1.0, 0.0, 0.0])
        oklab = srgb_to_oklab(srgb)
        # Red should have positive a value
        assert oklab[1] > 0.1

    def test_srgb_to_oklab_green(self):
        """Test conversion of green to OKLab."""
        srgb = np.array([0.0, 1.0, 0.0])
        oklab = srgb_to_oklab(srgb)
        # Green should have negative a value
        assert oklab[1] < -0.1

    def test_srgb_to_oklab_blue(self):
        """Test conversion of blue to OKLab."""
        srgb = np.array([0.0, 0.0, 1.0])
        oklab = srgb_to_oklab(srgb)
        # Blue should have negative b value
        assert oklab[2] < -0.1

    def test_srgb_to_oklab_255_range(self):
        """Test that function handles 0-255 range."""
        srgb_255 = np.array([255.0, 0.0, 0.0])
        srgb_1 = np.array([1.0, 0.0, 0.0])

        oklab_255 = srgb_to_oklab(srgb_255)
        oklab_1 = srgb_to_oklab(srgb_1)

        np.testing.assert_allclose(oklab_255, oklab_1, rtol=1e-5)

    def test_oklab_to_srgb_roundtrip(self):
        """Test that oklab->srgb->oklab conversion is reversible."""
        original = np.array([0.6, 0.1, -0.05])
        srgb = oklab_to_srgb(original)
        recovered = srgb_to_oklab(srgb)
        np.testing.assert_allclose(recovered, original, rtol=1e-3)

    def test_srgb_oklab_roundtrip_multiple_colors(self):
        """Test roundtrip conversion for multiple colors."""
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [0.5, 0.5, 0.5],  # Gray
        ])

        oklab = srgb_to_oklab(colors)
        recovered = oklab_to_srgb(oklab)

        # Allow more tolerance for edge cases and out-of-gamut colors
        np.testing.assert_allclose(recovered, colors, rtol=0.01, atol=0.01)


class TestOKLabDistance:
    """Tests for OKLab distance calculations."""

    def test_distance_identical_colors(self):
        """Test distance between identical colors is zero."""
        color = np.array([0.5, 0.1, -0.05])
        distance = oklab_distance(color, color)
        assert distance == pytest.approx(0.0)

    def test_distance_symmetry(self):
        """Test that distance is symmetric."""
        color1 = np.array([0.5, 0.1, -0.05])
        color2 = np.array([0.6, 0.15, -0.03])

        dist1 = oklab_distance(color1, color2)
        dist2 = oklab_distance(color2, color1)

        assert dist1 == pytest.approx(dist2)

    def test_distance_positive(self):
        """Test that distance is always positive."""
        color1 = np.array([0.5, 0.1, -0.05])
        color2 = np.array([0.6, 0.15, -0.03])

        distance = oklab_distance(color1, color2)
        assert distance > 0

    def test_batch_distance(self):
        """Test batch distance calculation."""
        colors = np.array([
            [0.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.7, 0.0, 0.0],
        ])
        reference = np.array([0.5, 0.0, 0.0])

        distances = batch_oklab_distance(colors, reference)

        assert len(distances) == 3
        assert distances[0] == pytest.approx(0.0)
        assert distances[1] > 0
        assert distances[2] > distances[1]

    def test_batch_distance_matches_individual(self):
        """Test that batch distance matches individual calculations."""
        colors = np.array([
            [0.5, 0.1, -0.05],
            [0.6, 0.15, -0.03],
            [0.7, -0.1, 0.02],
        ])
        reference = np.array([0.5, 0.0, 0.0])

        batch_dist = batch_oklab_distance(colors, reference)

        for i, color in enumerate(colors):
            individual_dist = oklab_distance(color, reference)
            assert batch_dist[i] == pytest.approx(individual_dist)


class TestColorConversionEdgeCases:
    """Tests for edge cases in color conversion."""

    def test_out_of_gamut_colors(self):
        """Test handling of out-of-gamut colors."""
        # Create an OKLab color that might be out of sRGB gamut
        oklab = np.array([0.8, 0.3, 0.3])
        srgb = oklab_to_srgb(oklab)

        # Should be clipped to valid range
        assert np.all(srgb >= 0.0)
        assert np.all(srgb <= 1.0)

    def test_batch_conversion(self):
        """Test batch conversion of multiple colors."""
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
        ])

        oklab = srgb_to_oklab(colors)

        assert oklab.shape == (4, 3)
        # Each color should have valid OKLab values
        assert np.all(oklab[:, 0] >= 0.0)  # L should be positive
        assert np.all(oklab[:, 0] <= 1.0)  # L should be <= 1

    def test_single_channel_array(self):
        """Test handling of 1D array."""
        color = np.array([128, 64, 192])
        oklab = srgb_to_oklab(color)

        assert oklab.shape == (3,)

    def test_very_dark_colors(self):
        """Test conversion of very dark colors."""
        # Use 0-1 range for dark color
        dark = np.array([0.01, 0.01, 0.01])  # Almost black
        oklab = srgb_to_oklab(dark)

        assert oklab[0] < 0.15  # Should have low lightness

    def test_very_bright_colors(self):
        """Test conversion of very bright colors."""
        bright = np.array([254, 254, 254])  # Almost white
        oklab = srgb_to_oklab(bright)

        assert oklab[0] > 0.9  # Should have high lightness


class TestPerceptualUniformity:
    """Tests for perceptual uniformity of OKLab."""

    def test_similar_srgb_distances_similar_oklab(self):
        """Test that similar sRGB differences produce similar OKLab distances."""
        # Two pairs of colors with similar RGB differences
        red1 = np.array([255, 0, 0])
        red2 = np.array([200, 0, 0])

        green1 = np.array([0, 255, 0])
        green2 = np.array([0, 200, 0])

        red1_oklab = srgb_to_oklab(red1)
        red2_oklab = srgb_to_oklab(red2)
        green1_oklab = srgb_to_oklab(green1)
        green2_oklab = srgb_to_oklab(green2)

        red_dist = oklab_distance(red1_oklab, red2_oklab)
        green_dist = oklab_distance(green1_oklab, green2_oklab)

        # Distances should be relatively similar (within 50%)
        # This is a loose test since RGB differences don't directly translate
        ratio = red_dist / green_dist
        assert 0.5 < ratio < 2.0
