"""Command-line interface for spot color detection."""

import argparse
import json
import sys
from pathlib import Path

from .detector import SpotColorDetector, DetectionMode


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Detect spot colors in PNG images for screen printing quotes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.png --mode aggressive
  %(prog)s image.png --mode conservative --dark-substrate
  %(prog)s image.png --output results.json

Output JSON format:
  {
    "color_count": 3,           // Number of spot colors (-1 if unsuitable)
    "colors": [...],            // Detailed color information
    "flags": [...],             // Warning/info flags
    "confidence": "high",       // Detection confidence level
    "mode": "conservative",     // Detection mode used
    "unsuitable_reason": null   // Reason if color_count is -1
  }
        """
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to PNG image file"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Detection mode: 'conservative' (default) slightly overestimates for safer quotes, "
             "'aggressive' consolidates similar colors for competitive pricing"
    )

    parser.add_argument(
        "-d", "--dark-substrate",
        action="store_true",
        help="Composite transparency against dark background (for dark garments)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file path (default: print to stdout)"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only output the color count number"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Include additional diagnostic information"
    )

    args = parser.parse_args()

    # Validate image path
    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not args.image.suffix.lower() == ".png":
        print(f"Warning: File does not have .png extension: {args.image}", file=sys.stderr)

    # Create detector
    mode = DetectionMode.CONSERVATIVE if args.mode == "conservative" else DetectionMode.AGGRESSIVE
    detector = SpotColorDetector(mode=mode, dark_substrate=args.dark_substrate)

    # Run detection
    try:
        result = detector.detect(args.image)
    except Exception as e:
        print(f"Error during detection: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.quiet:
        print(result.color_count)
    else:
        result_dict = result.to_dict()

        if args.verbose:
            result_dict["input_file"] = str(args.image.absolute())
            result_dict["dark_substrate"] = args.dark_substrate

        indent = 2 if args.pretty else None
        json_output = json.dumps(result_dict, indent=indent)

        if args.output:
            args.output.write_text(json_output)
            print(f"Results written to {args.output}")
        else:
            print(json_output)

    # Exit with non-zero if unsuitable
    if result.color_count == -1:
        sys.exit(2)


if __name__ == "__main__":
    main()
