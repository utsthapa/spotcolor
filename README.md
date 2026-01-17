# SpotColor

SpotColor is a FastAPI service and CLI that counts spot colors in PNG artwork for screen printing quotes. It clusters colors in OKLab space, filters anti-aliased edges, and flags gradients or photographic images that are unsuitable for spot printing.

## Features
- OKLab + DBSCAN clustering tuned for screen print artwork
- Anti-alias filtering and transparency handling (light or dark substrates)
- Conservative and aggressive detection modes for quote flexibility
- FastAPI docs at `/docs` and `/redoc` plus a small HTML landing page
- CLI entry point `spot-colors` for local inspection

## Quick start
```bash
pip install -e ".[dev]"
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

### Analyze an image
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@artwork.png" \
  -F "mode=conservative" \
  -F "dark_substrate=false"
```

Example response:
```json
{
  "success": true,
  "color_count": 3,
  "colors": [
    {"rgb":[255,255,255],"hex":"#ffffff","oklab":[1.0,0.0,0.0],"pixel_count":25000,"percentage":50.0},
    {"rgb":[255,107,107],"hex":"#ff6b6b","oklab":[0.72,0.12,0.07],"pixel_count":15000,"percentage":30.0}
  ],
  "flags": ["has_transparency"],
  "confidence": "high",
  "mode": "conservative",
  "unsuitable_reason": null
}
```

### CLI
```bash
spot-colors artwork.png --mode aggressive --dark-substrate
```

## Testing and CI
- Run tests locally: `pytest -q`
- Regression tests include the full `test_images/manifest.json` dataset to guard against detector drift.
- Coverage: `pytest --cov=screenprint --cov=api`
- GitHub Actions workflow `.github/workflows/test.yml` runs tests and lint on pushes and pull requests.

## Project layout
```
api/                        FastAPI app, models, and warmup endpoint
screenprint/                Detection pipeline and CLI
tests/                      Unit tests for API, detector, and utilities
.github/workflows/test.yml  CI pipeline
Dockerfile                  Container for running the API
pyproject.toml              Package metadata and scripts
requirements.txt            Runtime dependencies
LICENSE                     MIT license
```

## License
MIT
