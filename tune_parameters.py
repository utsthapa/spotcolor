"""Parameter tuning script for screen print color detection.

Tests various parameter combinations to find optimal settings.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN

from screenprint.color_utils import srgb_to_oklab
from screenprint.detector import SpotColorDetector, DetectionMode, DetectionResult


@dataclass
class TuningResult:
    params: dict
    conservative_accuracy: float
    aggressive_accuracy: float
    combined_accuracy: float
    false_positives: int  # Gradients/photos detected as valid
    false_negatives: int  # Valid logos rejected


class TunableDetector(SpotColorDetector):
    """Detector with tunable parameters for experimentation."""

    def __init__(self, mode, **kwargs):
        super().__init__(mode)
        self.tune_params = kwargs

    def _detect_complexity(self, pixels, aa_mask):
        h, w = pixels.shape[:2]
        total_pixels = h * w

        # Get tuning parameters with defaults
        min_unique_for_gradient = self.tune_params.get('min_unique_for_gradient', 50)
        max_l_gap_threshold = self.tune_params.get('max_l_gap_threshold', 0.08)
        small_grad_threshold = self.tune_params.get('small_grad_threshold', 0.5)
        zero_grad_threshold = self.tune_params.get('zero_grad_threshold', 0.90)
        min_l_range = self.tune_params.get('min_l_range', 0.15)

        aa_percentage = aa_mask.sum() / total_pixels
        has_significant_aa = aa_percentage > 0.02

        flat_pixels = pixels.reshape(-1, 3)
        flat_aa_mask = aa_mask.reshape(-1)

        non_aa_pixels = flat_pixels[~flat_aa_mask]
        if len(non_aa_pixels) < 100:
            non_aa_pixels = flat_pixels

        unique_colors = np.unique(non_aa_pixels, axis=0)
        unique_count = len(unique_colors)

        if unique_count <= 20:
            return False, None

        # Convert to grayscale for gradient analysis
        gray = np.mean(pixels.astype(np.float32), axis=2)

        # Zero gradient ratio - key differentiator
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        grad_x_flat = grad_x.flatten()
        grad_y_flat = grad_y.flatten()

        zero_grad = ((grad_x_flat == 0).sum() + (grad_y_flat == 0).sum())
        total_grad = len(grad_x_flat) + len(grad_y_flat)
        zero_grad_ratio = zero_grad / total_grad

        # If high zero gradient ratio, it's likely a solid-color logo
        if zero_grad_ratio > zero_grad_threshold:
            return False, None

        # Small gradient ratio
        small_grad_x = ((grad_x_flat > 0.5) & (grad_x_flat < 10)).sum()
        small_grad_y = ((grad_y_flat > 0.5) & (grad_y_flat < 10)).sum()
        small_grad_ratio = (small_grad_x + small_grad_y) / total_grad

        # Check luminance distribution
        if unique_count >= min_unique_for_gradient:
            oklab_colors = srgb_to_oklab(unique_colors.astype(np.float64))
            l_values = oklab_colors[:, 0]
            l_range = l_values.max() - l_values.min()

            l_sorted = np.sort(l_values)
            l_gaps = np.diff(l_sorted)
            max_gap = l_gaps.max() if len(l_gaps) > 0 else 0

            # Gradient: many colors, small gaps, spread over a range
            if l_range > min_l_range and max_gap < max_l_gap_threshold:
                return True, f"Gradient detected (max_gap={max_gap:.4f}, unique={unique_count})"

            # High small gradient ratio with many colors
            if small_grad_ratio > small_grad_threshold and unique_count > 80 and not has_significant_aa:
                return True, f"Gradient detected (small_grad_ratio={small_grad_ratio:.2%})"

        # Photographic detection
        if unique_count > 3000:
            oklab_colors = srgb_to_oklab(unique_colors.astype(np.float64))
            l_values = oklab_colors[:, 0]

            hist, _ = np.histogram(l_values, bins=20, range=(0, 1))
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized[hist_normalized > 0] *
                             np.log2(hist_normalized[hist_normalized > 0]))

            a_range = oklab_colors[:, 1].max() - oklab_colors[:, 1].min()
            b_range = oklab_colors[:, 2].max() - oklab_colors[:, 2].min()

            if entropy > 3.0 and unique_count > 5000:
                return True, "Photographic content detected"

        return False, None


def run_test_with_params(params: dict) -> TuningResult:
    """Run tests with specific parameters."""
    with open("test_images/manifest.json") as f:
        manifest = json.load(f)

    conservative_correct = 0
    aggressive_correct = 0
    false_positives = 0
    false_negatives = 0

    for test in manifest:
        path = test["path"]
        expected = test["expected_colors"]

        for mode in [DetectionMode.CONSERVATIVE, DetectionMode.AGGRESSIVE]:
            detector = TunableDetector(mode, **params)
            result = detector.detect(path)
            actual = result.color_count

            correct = (actual == expected) or (expected == -1 and actual == -1)

            if mode == DetectionMode.CONSERVATIVE:
                if correct:
                    conservative_correct += 1
            else:
                if correct:
                    aggressive_correct += 1

            # Track error types
            if expected == -1 and actual != -1:
                false_positives += 1
            elif expected != -1 and actual == -1:
                false_negatives += 1

    total = len(manifest)
    return TuningResult(
        params=params,
        conservative_accuracy=conservative_correct / total,
        aggressive_accuracy=aggressive_correct / total,
        combined_accuracy=(conservative_correct + aggressive_correct) / (2 * total),
        false_positives=false_positives // 2,  # Counted twice for two modes
        false_negatives=false_negatives // 2
    )


def grid_search():
    """Perform grid search over parameter combinations."""
    print("Starting parameter grid search...")
    print("=" * 80)

    # Reduced parameter ranges based on analysis
    param_grid = {
        'max_l_gap_threshold': [0.08, 0.10, 0.12],
        'small_grad_threshold': [0.4, 0.6],
        'zero_grad_threshold': [0.90, 0.95],
        'min_unique_for_gradient': [50, 70],
        'min_l_range': [0.12, 0.18],
    }

    best_result = None
    best_accuracy = 0
    results = []

    # Test all combinations
    from itertools import product
    keys = list(param_grid.keys())
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    print(f"Testing {total_combos} parameter combinations...")

    for i, values in enumerate(product(*param_grid.values())):
        params = dict(zip(keys, values))

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{total_combos}")

        result = run_test_with_params(params)
        results.append(result)

        if result.combined_accuracy > best_accuracy:
            best_accuracy = result.combined_accuracy
            best_result = result

    print("\n" + "=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 80)

    # Sort by combined accuracy
    results.sort(key=lambda r: r.combined_accuracy, reverse=True)

    for i, r in enumerate(results[:10]):
        print(f"\n#{i+1}: Combined={r.combined_accuracy:.1%}, "
              f"Con={r.conservative_accuracy:.1%}, Agg={r.aggressive_accuracy:.1%}")
        print(f"    FP={r.false_positives}, FN={r.false_negatives}")
        print(f"    Params: {r.params}")

    print("\n" + "=" * 80)
    print("BEST RESULT")
    print("=" * 80)
    print(f"Combined accuracy: {best_result.combined_accuracy:.1%}")
    print(f"Conservative: {best_result.conservative_accuracy:.1%}")
    print(f"Aggressive: {best_result.aggressive_accuracy:.1%}")
    print(f"False positives: {best_result.false_positives}")
    print(f"False negatives: {best_result.false_negatives}")
    print(f"Parameters: {best_result.params}")

    return best_result


if __name__ == "__main__":
    best = grid_search()
