"""Tests for the FastAPI application endpoints."""

import pytest
import io
from PIL import Image


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_html(self, client):
        """Test root endpoint returns HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"SpotColor API" in response.content


class TestAnalyzeEndpoint:
    """Tests for the image analysis endpoint."""

    def create_image_bytes(self, color=(255, 0, 0), size=(100, 100)):
        """Helper to create image bytes for upload."""
        img = Image.new('RGB', size, color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes

    def test_analyze_solid_image(self, client):
        """Test analyzing a simple solid color image."""
        img_bytes = self.create_image_bytes()

        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"mode": "conservative", "dark_substrate": "false"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["color_count"] == 1
        assert len(data["colors"]) == 1
        assert data["mode"] == "conservative"

    def test_analyze_with_aggressive_mode(self, client):
        """Test analyzing with aggressive mode."""
        img_bytes = self.create_image_bytes()

        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"mode": "aggressive", "dark_substrate": "false"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "aggressive"

    def test_analyze_with_dark_substrate(self, client):
        """Test analyzing with dark substrate option."""
        img_bytes = self.create_image_bytes()

        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"mode": "conservative", "dark_substrate": "true"}
        )

        assert response.status_code == 200

    def test_analyze_empty_file(self, client):
        """Test analyzing empty file returns error."""
        empty_bytes = io.BytesIO(b"")

        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", empty_bytes, "image/png")},
            data={"mode": "conservative"}
        )

        assert response.status_code == 400

    def test_analyze_invalid_file_type(self, client):
        """Test analyzing non-image file returns error."""
        text_bytes = io.BytesIO(b"not an image")

        response = client.post(
            "/api/analyze",
            files={"file": ("test.txt", text_bytes, "text/plain")},
            data={"mode": "conservative"}
        )

        assert response.status_code == 415

    def test_response_structure(self, client):
        """Test that response has all required fields."""
        img_bytes = self.create_image_bytes()

        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"mode": "conservative"}
        )

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "success" in data
        assert "color_count" in data
        assert "colors" in data
        assert "flags" in data
        assert "confidence" in data
        assert "mode" in data
        assert "unsuitable_reason" in data

        # Check color info structure
        if data["colors"]:
            color = data["colors"][0]
            assert "rgb" in color
            assert "hex" in color
            assert "oklab" in color
            assert "pixel_count" in color
            assert "percentage" in color


class TestDocumentationEndpoints:
    """Tests for API documentation endpoints."""

    def test_swagger_ui_accessible(self, client):
        """Test Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self, client):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "SpotColor API"
