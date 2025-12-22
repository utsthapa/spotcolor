"""Analyze failing test cases to understand why detection is wrong."""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from screenprint.color_utils import srgb_to_oklab

def analyze_image(path: str):
    """Analyze image characteristics."""
    img = Image.open(path).convert("RGB")
    pixels = np.array(img)
    h, w = pixels.shape[:2]

    flat_pixels = pixels.reshape(-1, 3)
    unique_colors = np.unique(flat_pixels, axis=0)
    unique_count = len(unique_colors)

    # Convert to grayscale for gradient analysis
    gray = np.mean(pixels.astype(np.float32), axis=2)

    # Compute gradients
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))

    grad_x_flat = grad_x.flatten()
    grad_y_flat = grad_y.flatten()

    # Gradient statistics
    small_grad_x = ((grad_x_flat > 0.5) & (grad_x_flat < 10)).sum()
    small_grad_y = ((grad_y_flat > 0.5) & (grad_y_flat < 10)).sum()
    total_grad_pixels = len(grad_x_flat) + len(grad_y_flat)
    small_grad_ratio = (small_grad_x + small_grad_y) / total_grad_pixels

    # Zero gradients (perfectly flat regions)
    zero_grad_x = (grad_x_flat == 0).sum()
    zero_grad_y = (grad_y_flat == 0).sum()
    zero_grad_ratio = (zero_grad_x + zero_grad_y) / total_grad_pixels

    # OKLab analysis
    oklab_colors = srgb_to_oklab(unique_colors.astype(np.float64))
    l_values = oklab_colors[:, 0]
    l_range = l_values.max() - l_values.min()

    l_sorted = np.sort(l_values)
    l_gaps = np.diff(l_sorted)
    max_gap = l_gaps.max() if len(l_gaps) > 0 else 0
    avg_gap = l_gaps.mean() if len(l_gaps) > 0 else 0

    # Chromatic range
    a_range = oklab_colors[:, 1].max() - oklab_colors[:, 1].min()
    b_range = oklab_colors[:, 2].max() - oklab_colors[:, 2].min()

    # Histogram entropy
    hist, _ = np.histogram(l_values, bins=20, range=(0, 1))
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized[hist_normalized > 0] *
                      np.log2(hist_normalized[hist_normalized > 0]))

    return {
        "size": f"{w}x{h}",
        "unique_colors": unique_count,
        "small_grad_ratio": round(small_grad_ratio, 4),
        "zero_grad_ratio": round(zero_grad_ratio, 4),
        "l_range": round(l_range, 4),
        "max_l_gap": round(max_gap, 4),
        "avg_l_gap": round(avg_gap, 4),
        "a_range": round(a_range, 4),
        "b_range": round(b_range, 4),
        "entropy": round(entropy, 4),
    }

def main():
    # Load manifest
    with open("test_images/manifest.json") as f:
        manifest = json.load(f)

    # Find gradient and photographic images (expected -1)
    reject_images = [m for m in manifest if m["expected_colors"] == -1]

    print("=" * 80)
    print("ANALYSIS OF IMAGES THAT SHOULD BE REJECTED (expected_colors = -1)")
    print("=" * 80)

    for item in reject_images:
        path = item["path"]
        name = item["name"]
        category = item["category"]

        analysis = analyze_image(path)

        print(f"\n{category}/{name}:")
        print(f"  Size: {analysis['size']}")
        print(f"  Unique colors: {analysis['unique_colors']}")
        print(f"  Small gradient ratio: {analysis['small_grad_ratio']:.2%}")
        print(f"  Zero gradient ratio: {analysis['zero_grad_ratio']:.2%}")
        print(f"  L range: {analysis['l_range']:.3f}")
        print(f"  Max L gap: {analysis['max_l_gap']:.4f}")
        print(f"  Avg L gap: {analysis['avg_l_gap']:.4f}")
        print(f"  A range: {analysis['a_range']:.3f}, B range: {analysis['b_range']:.3f}")
        print(f"  Entropy: {analysis['entropy']:.3f}")

    print("\n" + "=" * 80)
    print("ANALYSIS OF VALID LOGO IMAGES (for comparison)")
    print("=" * 80)

    # Sample some valid images for comparison
    valid_images = [m for m in manifest if m["expected_colors"] > 0 and m["expected_colors"] <= 4][:10]

    for item in valid_images:
        path = item["path"]
        name = item["name"]
        category = item["category"]
        expected = item["expected_colors"]

        analysis = analyze_image(path)

        print(f"\n{category}/{name} (expected {expected} colors):")
        print(f"  Size: {analysis['size']}")
        print(f"  Unique colors: {analysis['unique_colors']}")
        print(f"  Small gradient ratio: {analysis['small_grad_ratio']:.2%}")
        print(f"  Zero gradient ratio: {analysis['zero_grad_ratio']:.2%}")
        print(f"  L range: {analysis['l_range']:.3f}")
        print(f"  Max L gap: {analysis['max_l_gap']:.4f}")
        print(f"  Avg L gap: {analysis['avg_l_gap']:.4f}")

if __name__ == "__main__":
    main()
