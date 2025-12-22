"""Pydantic models for the SpotColor API."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DetectionModeEnum(str, Enum):
    """Detection mode affects clustering sensitivity."""
    conservative = "conservative"
    aggressive = "aggressive"


class ColorInfoResponse(BaseModel):
    """Information about a detected spot color."""
    rgb: list[int] = Field(
        description="sRGB color values [R, G, B] in range 0-255",
        example=[255, 107, 107]
    )
    hex: str = Field(
        description="Hex color code",
        example="#ff6b6b"
    )
    oklab: list[float] = Field(
        description="OKLab perceptual color space coordinates [L, a, b]",
        example=[0.7234, 0.1245, 0.0678]
    )
    pixel_count: int = Field(
        description="Number of pixels in this color cluster",
        example=15420
    )
    percentage: float = Field(
        description="Percentage of total non-background pixels",
        example=35.67
    )


class AnalyzeRequest(BaseModel):
    """Request model for image analysis."""
    mode: DetectionModeEnum = Field(
        default=DetectionModeEnum.conservative,
        description="Detection mode: 'conservative' slightly overestimates for safer quotes, 'aggressive' consolidates similar colors for competitive pricing"
    )
    dark_substrate: bool = Field(
        default=False,
        description="If true, composite transparency against dark background (for dark garments/substrates)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "conservative",
                "dark_substrate": False
            }
        }


class AnalyzeResponse(BaseModel):
    """Response model for image analysis."""
    success: bool = Field(
        description="Whether the analysis completed successfully"
    )
    color_count: int = Field(
        description="Number of spot colors detected (-1 if image is unsuitable for screen printing)",
        example=3
    )
    colors: list[ColorInfoResponse] = Field(
        description="List of detected colors with details"
    )
    flags: list[str] = Field(
        description="Warning and info flags about the image",
        example=["has_transparency", "filtered_antialiasing_2.3%"]
    )
    confidence: str = Field(
        description="Detection confidence level: 'high', 'medium', or 'low'",
        example="high"
    )
    mode: str = Field(
        description="Detection mode that was used",
        example="conservative"
    )
    unsuitable_reason: Optional[str] = Field(
        default=None,
        description="If color_count is -1, explains why the image is unsuitable"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
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
                    },
                    {
                        "rgb": [78, 205, 196],
                        "hex": "#4ecdc4",
                        "oklab": [0.7856, -0.0912, -0.0234],
                        "pixel_count": 10000,
                        "percentage": 20.0
                    }
                ],
                "flags": ["has_transparency"],
                "confidence": "high",
                "mode": "conservative",
                "unsuitable_reason": None
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    code: str = Field(description="Error code for programmatic handling")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "File must be an image (PNG recommended)",
                "code": "INVALID_FILE_TYPE"
            }
        }
