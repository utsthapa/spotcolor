"""Regression tests that run the detector against the bundled image manifest."""

import json
from pathlib import Path

import pytest

from screenprint.detector import DetectionMode, SpotColorDetector


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "test_images" / "manifest.json"


def _load_cases():
    if not MANIFEST_PATH.exists():
        return []
    data = json.loads(MANIFEST_PATH.read_text())
    cases = []
    for item in data:
        image_path = ROOT / Path(item["path"])
        if not image_path.exists():
            continue

        # Conservative expectation
        cases.append(
            pytest.param(
                image_path,
                item["expected_colors"],
                DetectionMode.CONSERVATIVE,
                id=f"{item['name']}_conservative",
            )
        )

        # Aggressive expectation, if provided
        if "expected_colors_aggressive" in item:
            cases.append(
                pytest.param(
                    image_path,
                    item["expected_colors_aggressive"],
                    DetectionMode.AGGRESSIVE,
                    id=f"{item['name']}_aggressive",
                )
            )
    return cases


@pytest.mark.parametrize("image_path,expected,mode", _load_cases())
def test_manifest_images_match_expected_color_counts(image_path, expected, mode):
    detector = SpotColorDetector(mode=mode)
    result = detector.detect(str(image_path))
    assert result.color_count == expected, f"{image_path.name} ({mode.value})"
