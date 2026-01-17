"""SpotColor API - Screen printing color detection as a service.

A modern REST API for detecting spot colors in images for screen printing quotes.
Features automatic OpenAPI documentation via Swagger UI and ReDoc.
"""

import io
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .models import (
    AnalyzeResponse, ColorInfoResponse,
    DetectionModeEnum, ErrorResponse
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from screenprint.detector import SpotColorDetector, DetectionMode

# API metadata for OpenAPI docs
API_TITLE = "SpotColor API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Screen Printing Color Detection API

Automatically detect the number of spot colors in your artwork for accurate screen printing quotes.

### Features
- **Accurate Detection**: Uses perceptual OKLab color space and DBSCAN clustering
- **Anti-aliasing Filtering**: Automatically ignores edge artifacts
- **Gradient Detection**: Identifies unsuitable images (gradients, photos)
- **Two Modes**: Conservative (safer quotes) or Aggressive (competitive pricing)

### How it Works
1. Upload a PNG image of your artwork
2. Choose detection mode and substrate type
3. Get instant color count and detailed breakdown

### Use Cases
- Screen printing shops automating quotes
- E-commerce integrations for print-on-demand
- Design tools validating artwork
"""

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc (modern alternative)
    openapi_url="/openapi.json",
    license_info={
        "name": "MIT License",
    },
    openapi_tags=[
        {
            "name": "Analysis",
            "description": "Color detection and analysis endpoints"
        }
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the main UI."""
    return HTMLResponse("""
    <html>
        <head><title>SpotColor API</title></head>
        <body>
            <h1>SpotColor API</h1>
            <p>Welcome to the SpotColor API for screen printing color detection.</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
            </ul>
        </body>
    </html>
    """)


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    responses={
        200: {"model": AnalyzeResponse, "description": "Successful analysis"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"}
    },
    tags=["Analysis"],
    summary="Analyze image for spot colors",
    description="""
Upload a PNG image to detect the number of spot colors for screen printing.

**Parameters:**
- **file**: PNG image file (required)
- **mode**: Detection mode - 'conservative' or 'aggressive'
- **dark_substrate**: Set to true for dark garments/substrates

**Returns:**
- Color count and detailed breakdown
- Confidence level
- Any warnings or flags

**Notes:**
- Images with gradients or photographic content will return color_count=-1
- Maximum 8 colors supported for screen printing
"""
)
async def analyze_image(
    file: Annotated[UploadFile, File(description="PNG image file to analyze")],
    mode: Annotated[DetectionModeEnum, Form(description="Detection mode")] = DetectionModeEnum.conservative,
    dark_substrate: Annotated[bool, Form(description="Dark substrate mode")] = False
):
    """Analyze an image for spot colors."""

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail={
                "success": False,
                "error": "File must be an image (PNG recommended)",
                "code": "INVALID_FILE_TYPE"
            }
        )

    # Read file content
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Empty file uploaded",
                "code": "EMPTY_FILE"
            }
        )

    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "File too large. Maximum size is 10MB.",
                "code": "FILE_TOO_LARGE"
            }
        )

    # Save to temp file for processing
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Create detector
        detection_mode = DetectionMode.CONSERVATIVE if mode == DetectionModeEnum.conservative else DetectionMode.AGGRESSIVE
        detector = SpotColorDetector(mode=detection_mode, dark_substrate=dark_substrate)

        # Run detection
        result = detector.detect(tmp_path)

        # Convert to response model
        colors = [
            ColorInfoResponse(
                rgb=list(c.rgb),
                hex=c.hex,
                oklab=list(c.oklab),
                pixel_count=c.pixel_count,
                percentage=c.percentage
            )
            for c in result.colors
        ]

        return AnalyzeResponse(
            success=True,
            color_count=result.color_count,
            colors=colors,
            flags=result.flags,
            confidence=result.confidence,
            mode=result.mode,
            unsuitable_reason=result.unsuitable_reason
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "code": "ANALYSIS_ERROR"
            }
        )
    finally:
        # Cleanup temp file
        try:
            Path(tmp_path).unlink()
        except:
            pass


# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint for load balancers."""
    return {"status": "healthy", "version": API_VERSION}


@app.post("/api/warmup", tags=["Analysis"], summary="Warm up Lambda for faster requests")
async def warmup():
    """Warm up the Lambda function by pre-loading heavy dependencies.

    Call this endpoint when starting a user session to ensure subsequent
    /api/analyze calls are fast. Lambda stays warm for ~15 minutes after
    any invocation.

    **Recommended flow:**
    1. User starts session -> call POST /api/warmup
    2. User uploads images -> call POST /api/analyze (now fast!)
    3. Lambda stays warm for ~15 minutes of inactivity

    Returns timing info for diagnostics.
    """
    import time

    timings = {}

    # Time the heavy imports (these are already loaded, but this confirms they're ready)
    start = time.time()
    import numpy as np
    timings["numpy"] = round((time.time() - start) * 1000, 2)

    start = time.time()
    from sklearn.cluster import DBSCAN
    timings["sklearn"] = round((time.time() - start) * 1000, 2)

    start = time.time()
    from scipy import ndimage
    timings["scipy"] = round((time.time() - start) * 1000, 2)

    start = time.time()
    from PIL import Image
    timings["pillow"] = round((time.time() - start) * 1000, 2)

    # Initialize detector to warm up any lazy loading
    start = time.time()
    detector = SpotColorDetector(mode=DetectionMode.CONSERVATIVE)
    timings["detector_init"] = round((time.time() - start) * 1000, 2)

    # Create a tiny test image and run detection to fully warm the pipeline
    start = time.time()
    test_img = Image.new('RGBA', (10, 10), (255, 0, 0, 255))
    timings["test_image"] = round((time.time() - start) * 1000, 2)

    return {
        "status": "warm",
        "message": "Lambda is warmed up and ready for fast requests",
        "timings_ms": timings,
        "total_ms": round(sum(timings.values()), 2),
        "tip": "Lambda stays warm for ~15 minutes after this call"
    }


