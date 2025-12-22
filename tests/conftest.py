"""Pytest configuration and fixtures for SpotColor API tests."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def test_images_dir():
    """Return the path to the test images directory."""
    return Path(__file__).parent.parent / "test_images"


@pytest.fixture
def sample_solid_image(tmp_path):
    """Create a simple solid color test image."""
    img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    img_path = tmp_path / "solid_red.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_two_color_image(tmp_path):
    """Create a two-color test image."""
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    pixels = img.load()

    # Left half red, right half blue
    for x in range(50):
        for y in range(100):
            pixels[x, y] = (255, 0, 0)

    for x in range(50, 100):
        for y in range(100):
            pixels[x, y] = (0, 0, 255)

    img_path = tmp_path / "two_colors.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_gradient_image(tmp_path):
    """Create a gradient image (unsuitable for screen printing)."""
    img = Image.new('RGB', (100, 100))
    pixels = img.load()

    # Create horizontal gradient
    for x in range(100):
        for y in range(100):
            value = int(255 * x / 100)
            pixels[x, y] = (value, value, value)

    img_path = tmp_path / "gradient.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_transparent_image(tmp_path):
    """Create an image with transparency."""
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))
    pixels = img.load()

    # Make center transparent
    for x in range(25, 75):
        for y in range(25, 75):
            pixels[x, y] = (255, 0, 0, 0)

    img_path = tmp_path / "transparent.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_antialiased_image(tmp_path):
    """Create an image with anti-aliased edges."""
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    pixels = img.load()

    # Draw a circle with anti-aliasing
    center_x, center_y = 50, 50
    radius = 30

    for x in range(100):
        for y in range(100):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius - 2:
                pixels[x, y] = (255, 0, 0)
            elif dist < radius:
                # Anti-aliased edge
                blend = (radius - dist) / 2
                r = int(255 * blend + 255 * (1 - blend))
                pixels[x, y] = (r, int(255 * (1 - blend)), int(255 * (1 - blend)))

    img_path = tmp_path / "antialiased.png"
    img.save(img_path)
    return img_path
