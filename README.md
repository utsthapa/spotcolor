# SpotColor API

> **Automatic spot color detection for screen printing** - Analyze artwork and instantly determine the number of distinct colors needed for accurate screen printing quotes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/yourusername/spotcolor-api/workflows/Tests/badge.svg)](https://github.com/yourusername/spotcolor-api/actions)

## Overview

SpotColor API is a production-ready REST API that automatically detects the number of spot colors in images for screen printing applications. Built with perceptual color science, it uses the OKLab color space and DBSCAN clustering to accurately identify colors as humans perceive them.

### Key Features

- 🎨 **Perceptual Color Detection** - Uses OKLab color space for human-like color perception
- 🔬 **DBSCAN Clustering** - Scientifically groups similar colors together
- ✂️ **Anti-Aliasing Filtering** - Automatically ignores edge artifacts that aren't real colors
- 📊 **Gradient Detection** - Identifies unsuitable images (gradients, photographs)
- ⚡ **Two Detection Modes**:
  - **Conservative**: Slightly overestimates colors for safer quotes
  - **Aggressive**: Consolidates similar colors for competitive pricing
- 🌓 **Dark Substrate Support** - Handles transparency for dark garments
- 📚 **Auto-Generated Docs** - Interactive Swagger UI and ReDoc
- 🚀 **AWS Lambda Ready** - Includes SAM deployment template
- ✅ **Comprehensive Tests** - 54 tests with 73% coverage

### Use Cases

- **Screen Printing Shops** - Automate quote generation
- **E-commerce Platforms** - Print-on-demand integrations
- **Design Tools** - Validate artwork before submission
- **Print Brokers** - Instant pricing calculations

---

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/spotcolor-api.git
cd spotcolor-api

# Install dependencies
pip install -e ".[dev]"

# Run the development server
python -m api.server
```

Visit:
- **API Playground**: http://localhost:8000/docs
- **API Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## API Documentation

### Analyze Image

**Endpoint**: `POST /api/analyze`

**Request**:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@artwork.png" \
  -F "mode=conservative" \
  -F "dark_substrate=false"
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | ✅ Yes | - | PNG image to analyze |
| `mode` | String | ❌ No | `conservative` | Detection mode: `conservative` or `aggressive` |
| `dark_substrate` | Boolean | ❌ No | `false` | Set to `true` for dark garments/substrates |

**Success Response** (200 OK):
```json
{
  "success": true,
  "color_count": 3,
  "colors": [
    {
      "rgb": [255, 255, 255],
      "hex": "#ffffff",
      "oklab": [1.0, 0.0, 0.0],
      "pixel_count": 25000,
      "percentage": 50.0
    },
    {
      "rgb": [255, 107, 107],
      "hex": "#ff6b6b",
      "oklab": [0.7234, 0.1245, 0.0678],
      "pixel_count": 15000,
      "percentage": 30.0
    }
  ],
  "flags": ["has_transparency"],
  "confidence": "high",
  "mode": "conservative",
  "unsuitable_reason": null
}
```

**Error Responses**:

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| `400` | `EMPTY_FILE` | Empty file uploaded |
| `400` | `FILE_TOO_LARGE` | File exceeds 10MB limit |
| `415` | `INVALID_FILE_TYPE` | File is not an image |
| `500` | `ANALYSIS_ERROR` | Internal analysis error |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spotcolor-api.git
cd spotcolor-api

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run server
python -m api.server
```

---

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=screenprint --cov=api --cov-report=html

# Run specific tests
pytest tests/test_detector.py -v
```

**Test Coverage**: 54 tests covering API endpoints, color detection, and utilities.

---

## Deployment

### AWS Lambda (Recommended)

```bash
# Using AWS SAM
cd deploy
./deploy.sh prod
```

### Docker

```bash
docker build -t spotcolor-api .
docker run -p 8000:8000 spotcolor-api
```

### Traditional Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Detection Algorithm

SpotColor uses a 4-phase detection pipeline:

1. **Transparency Flattening** - Composites RGBA against white/black background
2. **Complexity Detection** - Identifies gradients and photographic content (returns `-1` if unsuitable)
3. **Anti-Aliasing Filtering** - Removes edge artifacts using Sobel detection
4. **Color Clustering** - Groups colors using DBSCAN in OKLab space

### Why OKLab?

OKLab is a perceptually uniform color space, meaning equal distances correspond to equal perceptual differences. This provides more accurate color detection than RGB or traditional LAB space.

---

## Project Structure

```
spotcolor-api/
├── api/                  # FastAPI application
├── screenprint/          # Core detection library
├── tests/                # Test suite (54 tests)
├── deploy/               # AWS deployment configs
├── .github/workflows/    # GitHub Actions CI/CD
└── test_images/          # Test dataset
```

---

## CLI Usage

```bash
# Detect colors in an image
spot-colors image.png

# Use aggressive mode
spot-colors image.png --mode aggressive

# Dark substrate mode
spot-colors image.png --dark-substrate
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether analysis completed |
| `color_count` | int | Number of colors (`-1` if unsuitable) |
| `colors` | array | Detected colors with RGB, hex, OKLab values |
| `flags` | array | Warnings (transparency, AA filtering, etc.) |
| `confidence` | string | `high`, `medium`, or `low` |
| `unsuitable_reason` | string | Explanation if image is unsuitable |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure tests pass (`pytest`)
5. Format code (`black .`)
6. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the screen printing community**
