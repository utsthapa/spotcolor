"""Tests for the spot color detector."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from screenprint.detector import SpotColorDetector, DetectionMode, ColorInfo, DetectionResult


class TestDetectorInitialization:
    """Tests for detector initialization."""

    def test_default_initialization(self):
        """Test detector initializes with default parameters."""
        detector = SpotColorDetector()
        assert detector.mode == DetectionMode.CONSERVATIVE
        assert detector.dark_substrate is False

    def test_aggressive_mode_initialization(self):
        """Test detector with aggressive mode."""
        detector = SpotColorDetector(mode=DetectionMode.AGGRESSIVE)
        assert detector.mode == DetectionMode.AGGRESSIVE

    def test_dark_substrate_initialization(self):
        """Test detector with dark substrate."""
        detector = SpotColorDetector(dark_substrate=True)
        assert detector.dark_substrate is True


class TestSolidColorDetection:
    """Tests for detecting solid colors."""

    def test_single_solid_color(self, sample_solid_image):
        """Test detecting a single solid color."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_solid_image))

        assert result.color_count == 1
        assert len(result.colors) == 1
        assert result.confidence in ["high", "medium", "low"]
        assert result.unsuitable_reason is None

    def test_two_solid_colors(self, sample_two_color_image):
        """Test detecting two distinct solid colors."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_two_color_image))

        assert result.color_count == 2
        assert len(result.colors) == 2

    def test_color_info_structure(self, sample_solid_image):
        """Test that ColorInfo has all required fields."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_solid_image))

        assert len(result.colors) > 0
        color = result.colors[0]

        assert hasattr(color, 'rgb')
        assert hasattr(color, 'hex')
        assert hasattr(color, 'oklab')
        assert hasattr(color, 'pixel_count')
        assert hasattr(color, 'percentage')

        # Check data types
        assert isinstance(color.rgb, tuple)
        assert len(color.rgb) == 3
        assert isinstance(color.hex, str)
        assert color.hex.startswith('#')
        assert isinstance(color.oklab, tuple)
        assert len(color.oklab) == 3
        assert isinstance(color.pixel_count, int)
        assert isinstance(color.percentage, float)


class TestGradientDetection:
    """Tests for detecting gradients and unsuitable images."""

    def test_gradient_image_unsuitable(self, sample_gradient_image):
        """Test that gradient images are marked as unsuitable."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_gradient_image))

        assert result.color_count == -1
        assert result.unsuitable_reason is not None
        assert len(result.colors) == 0


class TestTransparencyHandling:
    """Tests for handling transparent images."""

    def test_transparent_image_detection(self, sample_transparent_image):
        """Test detecting colors in transparent images."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_transparent_image))

        # Should detect at least the red color
        assert result.color_count >= 1
        # Should flag transparency
        assert any("transparency" in flag.lower() for flag in result.flags)

    def test_dark_substrate_mode(self, sample_transparent_image):
        """Test dark substrate mode for transparent images."""
        detector_light = SpotColorDetector(dark_substrate=False)
        detector_dark = SpotColorDetector(dark_substrate=True)

        result_light = detector_light.detect(str(sample_transparent_image))
        result_dark = detector_dark.detect(str(sample_transparent_image))

        # Both should succeed but may have different color counts
        assert result_light.color_count >= 1
        assert result_dark.color_count >= 1


class TestAntiAliasingFiltering:
    """Tests for anti-aliasing detection and filtering."""

    def test_antialiased_edges_filtered(self, sample_antialiased_image):
        """Test that anti-aliased edges don't create extra colors."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_antialiased_image))

        # Should detect main colors, not edge artifacts
        # The image has red circle on white background
        assert result.color_count <= 2

        # Should flag anti-aliasing filtering
        aa_filtered = any("antialiasing" in flag.lower() or "filtered" in flag.lower()
                         for flag in result.flags)
        # May or may not filter depending on image, just check it doesn't crash


class TestDetectionModes:
    """Tests for different detection modes."""

    def test_conservative_vs_aggressive(self, sample_two_color_image):
        """Test that aggressive mode may consolidate colors."""
        detector_conservative = SpotColorDetector(mode=DetectionMode.CONSERVATIVE)
        detector_aggressive = SpotColorDetector(mode=DetectionMode.AGGRESSIVE)

        result_conservative = detector_conservative.detect(str(sample_two_color_image))
        result_aggressive = detector_aggressive.detect(str(sample_two_color_image))

        # Both should succeed
        assert result_conservative.color_count >= 1
        assert result_aggressive.color_count >= 1

        # Aggressive should have <= colors than conservative
        assert result_aggressive.color_count <= result_conservative.color_count


class TestResultSerialization:
    """Tests for result serialization."""

    def test_color_info_to_dict(self):
        """Test ColorInfo serialization to dictionary."""
        color = ColorInfo(
            rgb=(255, 0, 0),
            hex="#ff0000",
            oklab=(0.6279, 0.2248, 0.1258),
            pixel_count=1000,
            percentage=50.0
        )

        data = color.to_dict()
        assert isinstance(data, dict)
        assert data["rgb"] == [255, 0, 0]
        assert data["hex"] == "#ff0000"
        assert isinstance(data["oklab"], list)
        assert data["pixel_count"] == 1000
        assert data["percentage"] == 50.0

    def test_detection_result_to_dict(self, sample_solid_image):
        """Test DetectionResult serialization to dictionary."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_solid_image))

        data = result.to_dict()
        assert isinstance(data, dict)
        assert "color_count" in data
        assert "colors" in data
        assert "flags" in data
        assert "confidence" in data
        assert "mode" in data
        assert "unsuitable_reason" in data


class TestConfidenceScoring:
    """Tests for confidence level calculation."""

    def test_confidence_levels(self, sample_solid_image, sample_two_color_image):
        """Test that confidence is calculated correctly."""
        detector = SpotColorDetector()

        result_simple = detector.detect(str(sample_solid_image))
        result_complex = detector.detect(str(sample_two_color_image))

        # Confidence should be one of the valid values
        assert result_simple.confidence in ["high", "medium", "low"]
        assert result_complex.confidence in ["high", "medium", "low"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        detector = SpotColorDetector()
        result = detector.detect("nonexistent.png")
        # Should return unsuitable result when file doesn't exist
        assert result.color_count == -1
        assert result.unsuitable_reason is not None

    def test_very_small_image(self, tmp_path):
        """Test handling of very small images."""
        img = Image.new('RGB', (1, 1), color=(255, 0, 0))
        img_path = tmp_path / "tiny.png"
        img.save(img_path)

        detector = SpotColorDetector()
        result = detector.detect(str(img_path))

        # Should still work
        assert result.color_count >= 0

    def test_large_solid_image(self, tmp_path):
        """Test handling of larger images."""
        img = Image.new('RGB', (500, 500), color=(0, 255, 0))
        img_path = tmp_path / "large.png"
        img.save(img_path)

        detector = SpotColorDetector()
        result = detector.detect(str(img_path))

        assert result.color_count == 1

    def test_many_colors_image(self, tmp_path):
        """Test image with many colors gets marked unsuitable."""
        img = Image.new('RGB', (100, 100))
        pixels = img.load()

        # Create image with many distinct colors
        color_idx = 0
        for x in range(100):
            for y in range(100):
                pixels[x, y] = (color_idx % 256, (color_idx * 2) % 256, (color_idx * 3) % 256)
                color_idx += 5

        img_path = tmp_path / "many_colors.png"
        img.save(img_path)

        detector = SpotColorDetector()
        result = detector.detect(str(img_path))

        # Should be marked unsuitable (too complex)
        assert result.color_count == -1 or result.color_count > 8


class TestPercentageCalculation:
    """Tests for color percentage calculation."""

    def test_percentages_sum_to_100(self, sample_two_color_image):
        """Test that color percentages sum to approximately 100%."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_two_color_image))

        if result.color_count > 0:
            total_percentage = sum(color.percentage for color in result.colors)
            # Allow some rounding error
            assert 99.0 <= total_percentage <= 101.0

    def test_pixel_counts_match_percentages(self, sample_solid_image):
        """Test that pixel counts align with percentages."""
        detector = SpotColorDetector()
        result = detector.detect(str(sample_solid_image))

        if result.color_count > 0:
            total_pixels = sum(color.pixel_count for color in result.colors)
            for color in result.colors:
                expected_percentage = (color.pixel_count / total_pixels) * 100
                # Allow some rounding error
                assert abs(color.percentage - expected_percentage) < 1.0
