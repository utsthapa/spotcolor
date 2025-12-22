"""Screen Printing Color Detection System.

Detects perceptually distinct spot colors in PNG images for screen printing quotes.
"""

from .detector import SpotColorDetector, DetectionResult, DetectionMode

__version__ = "1.0.0"
__all__ = ["SpotColorDetector", "DetectionResult", "DetectionMode"]
