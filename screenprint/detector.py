"""Core spot color detection for screen printing.

Implements a multi-phase detection pipeline:
1. Flatten transparency (composite against white)
2. Detect gradients and complexity (early exit if unsuitable)
3. Filter anti-aliasing artifacts using spatial coherence
4. Cluster colors in OKLab space using DBSCAN
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN

from .color_utils import srgb_to_oklab, oklab_to_srgb


class DetectionMode(Enum):
    """Detection mode affects clustering sensitivity."""
    CONSERVATIVE = "conservative"  # Slightly overestimate colors for safer quotes
    AGGRESSIVE = "aggressive"      # Consolidate similar colors for competitive pricing


@dataclass
class ColorInfo:
    """Information about a detected spot color."""
    rgb: tuple[int, int, int]          # sRGB color (0-255)
    hex: str                           # Hex color code
    oklab: tuple[float, float, float]  # OKLab coordinates
    pixel_count: int                   # Number of pixels in this cluster
    percentage: float                  # Percentage of total non-background pixels

    def to_dict(self) -> dict:
        return {
            "rgb": list(self.rgb),
            "hex": self.hex,
            "oklab": [round(v, 4) for v in self.oklab],
            "pixel_count": self.pixel_count,
            "percentage": round(self.percentage, 2)
        }


@dataclass
class DetectionResult:
    """Result of spot color detection."""
    color_count: int                    # Number of spot colors (-1 if unsuitable)
    colors: list[ColorInfo]             # Detected colors
    flags: list[str]                    # Warning/info flags
    confidence: str                     # "high", "medium", "low"
    mode: str                           # Detection mode used
    unsuitable_reason: Optional[str] = None  # Reason if color_count is -1

    def to_dict(self) -> dict:
        return {
            "color_count": self.color_count,
            "colors": [c.to_dict() for c in self.colors],
            "flags": self.flags,
            "confidence": self.confidence,
            "mode": self.mode,
            "unsuitable_reason": self.unsuitable_reason
        }


class SpotColorDetector:
    """Detects spot colors in images for screen printing.

    Implements a multi-phase pipeline to accurately count perceptually
    distinct colors while filtering out anti-aliasing artifacts and
    detecting unsuitable images (gradients, photos).
    """

    # DBSCAN parameters by mode
    PARAMS = {
        DetectionMode.CONSERVATIVE: {"epsilon": 0.025, "min_pts_divisor": 800},
        DetectionMode.AGGRESSIVE: {"epsilon": 0.055, "min_pts_divisor": 200},
    }

    # Thresholds for complexity detection
    MAX_SPOT_COLORS = 8                 # Maximum colors for screen printing

    # Anti-aliasing detection
    AA_EDGE_THRESHOLD = 0.1             # Edge detection sensitivity
    AA_MIN_REGION_SIZE = 5              # Minimum pixels for a "real" color region
    MAX_PIXELS = 1_000_000              # Safety cap to keep processing time predictable
    UNIQUE_COLOR_QUANTIZE_THRESHOLD = 50_000   # Above this, quantize to reduce work
    UNIQUE_COLOR_FAST_FAIL_THRESHOLD = 120_000 # Above this, treat as photographic
    QUANTIZE_COLORS = 256

    def __init__(
        self,
        mode: DetectionMode = DetectionMode.CONSERVATIVE,
        dark_substrate: bool = False,
        background_color: tuple[int, int, int] = (255, 255, 255),
        max_pixels: int | None = None
    ):
        """Initialize the detector.

        Args:
            mode: Detection mode (conservative or aggressive)
            dark_substrate: If True, composite transparency against dark background
            background_color: Background color for transparency compositing
            max_pixels: Optional cap for pixel count; images above this are downscaled
        """
        self.mode = mode
        self.dark_substrate = dark_substrate
        self.background_color = (0, 0, 0) if dark_substrate else background_color
        self.max_pixels = max_pixels or self.MAX_PIXELS
        self._flags: list[str] = []

    def detect(self, image_path: str | Path) -> DetectionResult:
        """Detect spot colors in an image.

        Args:
            image_path: Path to PNG image

        Returns:
            DetectionResult with color count, colors, flags, and confidence
        """
        self._flags = []
        image_path = Path(image_path)

        # Load image
        try:
            img = Image.open(image_path)
        except Exception as e:
            return self._unsuitable_result(f"Failed to load image: {e}")

        # Convert to RGBA if needed
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Resize oversized inputs to keep processing within API/Gateway time limits
        original_size = img.size
        total_pixels = original_size[0] * original_size[1]
        if self.max_pixels and total_pixels > self.max_pixels:
            scale = (self.max_pixels / total_pixels) ** 0.5
            new_size = (
                max(1, int(original_size[0] * scale)),
                max(1, int(original_size[1] * scale))
            )
            img = img.resize(new_size, Image.LANCZOS)
            self._flags.append(
                f"resized_for_performance_{original_size[0]}x{original_size[1]}_to_{new_size[0]}x{new_size[1]}"
            )

        # Phase 1: Flatten transparency
        flattened, has_transparency, has_semi_transparent = self._flatten_transparency(img)

        if has_transparency:
            self._flags.append("has_transparency")
        if has_semi_transparent:
            self._flags.append("has_semi_transparent_pixels")

        # Fast unique-color scan to avoid pathological runtime on photographic images
        color_scan = flattened.getcolors(maxcolors=self.UNIQUE_COLOR_FAST_FAIL_THRESHOLD + 1)
        unique_count = len(color_scan) if color_scan is not None else self.UNIQUE_COLOR_FAST_FAIL_THRESHOLD + 1

        if unique_count > self.UNIQUE_COLOR_FAST_FAIL_THRESHOLD:
            return self._unsuitable_result(
                f"Image contains too many unique colors ({unique_count}); likely photographic or gradient"
            )

        # Quantize very high-color images to keep AA/DBSCAN steps performant
        if unique_count > self.UNIQUE_COLOR_QUANTIZE_THRESHOLD:
            flattened = flattened.convert("P", palette=Image.ADAPTIVE, colors=self.QUANTIZE_COLORS).convert("RGB")
            self._flags.append(
                f"quantized_for_performance_{unique_count}_colors_to_{self.QUANTIZE_COLORS}"
            )

        # Convert to numpy array (RGB, 0-255)
        pixels = np.array(flattened)

        # Phase 3: Filter anti-aliasing artifacts (do this BEFORE complexity detection)
        core_pixels, aa_mask, aa_colors_removed = self._filter_antialiasing(pixels)

        if aa_mask.sum() > 0:
            aa_percentage = (aa_mask.sum() / aa_mask.size) * 100
            self._flags.append(f"filtered_antialiasing_{aa_percentage:.1f}%")

        # Phase 2: Detect gradients and complexity (after AA filtering)
        is_complex, complexity_reason = self._detect_complexity(core_pixels, aa_mask)
        if is_complex:
            return self._unsuitable_result(complexity_reason)

        # Phase 4: Cluster colors in OKLab space (using filtered pixels)
        colors, confidence = self._cluster_colors(core_pixels, aa_mask)

        # Check if too many colors
        if len(colors) > self.MAX_SPOT_COLORS:
            return self._unsuitable_result(
                f"Too many distinct colors ({len(colors)}) for screen printing"
            )

        return DetectionResult(
            color_count=len(colors),
            colors=colors,
            flags=self._flags,
            confidence=confidence,
            mode=self.mode.value
        )

    def _flatten_transparency(
        self, img: Image.Image
    ) -> tuple[Image.Image, bool, bool]:
        """Phase 1: Flatten transparency by compositing against background.

        Args:
            img: RGBA image

        Returns:
            Tuple of (flattened RGB image, has_any_transparency, has_semi_transparent)
        """
        rgba = np.array(img)
        alpha = rgba[:, :, 3]

        has_transparency = (alpha < 255).any()
        has_semi_transparent = ((alpha > 0) & (alpha < 255)).any()

        if not has_transparency:
            # No transparency, just convert to RGB
            return img.convert("RGB"), False, False

        # Composite against background
        bg = np.array(self.background_color, dtype=np.float32)
        fg = rgba[:, :, :3].astype(np.float32)
        alpha_norm = alpha.astype(np.float32) / 255.0

        # Alpha compositing: result = fg * alpha + bg * (1 - alpha)
        composited = (
            fg * alpha_norm[:, :, np.newaxis] +
            bg * (1 - alpha_norm[:, :, np.newaxis])
        )

        composited = np.clip(composited, 0, 255).astype(np.uint8)
        return Image.fromarray(composited, "RGB"), has_transparency, has_semi_transparent

    def _detect_complexity(
        self, pixels: NDArray[np.uint8], aa_mask: NDArray[np.bool_]
    ) -> tuple[bool, Optional[str]]:
        """Phase 2: Detect if image is too complex for screen printing.

        Checks for:
        - Gradients (smooth color transitions)
        - Photographic content (too many unique colors in non-AA regions)
        - High color variance in local neighborhoods

        Args:
            pixels: RGB pixel array (H, W, 3)
            aa_mask: Boolean mask of anti-aliased pixels to exclude

        Returns:
            Tuple of (is_complex, reason_if_complex)
        """
        h, w = pixels.shape[:2]
        total_pixels = h * w

        # Check AA percentage
        aa_percentage = aa_mask.sum() / total_pixels

        # Get unique colors from full image for analysis
        flat_pixels = pixels.reshape(-1, 3)
        flat_aa_mask = aa_mask.reshape(-1)

        # Get unique colors from full image
        all_unique_colors = np.unique(flat_pixels, axis=0)
        all_unique_count = len(all_unique_colors)

        # Get unique colors excluding AA pixels
        non_aa_pixels = flat_pixels[~flat_aa_mask]

        # CRITICAL: If almost all pixels are marked as AA (>50%), this is a strong
        # signal of gradient/photographic content - real logos have solid core regions
        if aa_percentage > 0.50 and all_unique_count > 50:
            # Check if the colors form a continuous distribution (gradient signature)
            oklab_colors = srgb_to_oklab(all_unique_colors.astype(np.float64))
            l_values = oklab_colors[:, 0]
            l_range = l_values.max() - l_values.min()

            if all_unique_count > 100:
                # Many colors with no solid regions = gradient or photo
                l_sorted = np.sort(l_values)
                l_gaps = np.diff(l_sorted)
                avg_gap = l_gaps.mean() if len(l_gaps) > 0 else 0

                # Check for continuous color distribution
                if avg_gap < 0.01 and l_range > 0.2:
                    return True, f"Image contains smooth gradient ({all_unique_count} colors, {aa_percentage*100:.0f}% non-solid)"

                # Check for noisy/photographic content with many scattered colors
                if all_unique_count > 200:
                    a_values = oklab_colors[:, 1]
                    b_values = oklab_colors[:, 2]
                    a_range = a_values.max() - a_values.min()
                    b_range = b_values.max() - b_values.min()
                    # Wide color distribution = photo-like
                    if a_range > 0.15 or b_range > 0.15:
                        return True, f"Image appears to be photographic ({all_unique_count} colors)"

        if len(non_aa_pixels) < 100:
            # Almost all pixels are AA - use full image
            non_aa_pixels = flat_pixels

        unique_colors = np.unique(non_aa_pixels, axis=0)
        unique_count = len(unique_colors)

        # Quick check: if very few unique colors in non-AA regions, it's not a gradient
        if unique_count <= 30:
            return False, None

        # Method 1: Check for smooth gradients by analyzing unique color count
        # True gradients have many unique colors forming a continuous range
        if unique_count > 50:
            # Convert to OKLab and check color distribution
            oklab_colors = srgb_to_oklab(unique_colors.astype(np.float64))

            # Check if colors form a continuous line/surface in color space
            l_values = oklab_colors[:, 0]
            a_values = oklab_colors[:, 1]
            b_values = oklab_colors[:, 2]

            l_range = l_values.max() - l_values.min()
            a_range = a_values.max() - a_values.min()
            b_range = b_values.max() - b_values.min()

            # Sort by L and check for continuous distribution
            l_sorted = np.sort(l_values)
            l_gaps = np.diff(l_sorted)
            max_gap = l_gaps.max() if len(l_gaps) > 0 else 0
            avg_gap = l_gaps.mean() if len(l_gaps) > 0 else 0

            # A gradient has small, consistent gaps between luminance values
            # A logo with distinct colors has large gaps
            if unique_count > 150 and l_range > 0.3:
                if max_gap < 0.03 and avg_gap < 0.005:
                    return True, f"Image contains smooth gradient ({unique_count} unique colors)"

            # Check for color-space gradients (like red->blue where L is constant but a/b change)
            # Sort by a or b channel and check for continuous distribution
            if unique_count > 50 and (a_range > 0.2 or b_range > 0.2):
                a_sorted = np.sort(a_values)
                b_sorted = np.sort(b_values)
                a_gaps = np.diff(a_sorted)
                b_gaps = np.diff(b_sorted)
                a_avg_gap = a_gaps.mean() if len(a_gaps) > 0 else 1
                b_avg_gap = b_gaps.mean() if len(b_gaps) > 0 else 1
                a_max_gap = a_gaps.max() if len(a_gaps) > 0 else 1
                b_max_gap = b_gaps.max() if len(b_gaps) > 0 else 1

                # Continuous distribution in a or b channel = color gradient
                if (a_range > 0.2 and a_avg_gap < 0.01 and a_max_gap < 0.03):
                    return True, f"Image contains color gradient ({unique_count} colors, a-channel range {a_range:.2f})"
                if (b_range > 0.2 and b_avg_gap < 0.01 and b_max_gap < 0.03):
                    return True, f"Image contains color gradient ({unique_count} colors, b-channel range {b_range:.2f})"

            # Lower threshold for detecting subtle gradients
            if unique_count > 50 and l_range > 0.1:
                if max_gap < 0.02 and avg_gap < 0.005:
                    return True, f"Image contains subtle gradient ({unique_count} unique colors)"

        # Method 2: Spatial gradient detection using derivative analysis
        gray = np.mean(pixels.astype(np.float32), axis=2)

        # Compute image gradients
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))

        grad_x_flat = grad_x.flatten()
        grad_y_flat = grad_y.flatten()

        # Count pixels with small but non-zero gradients (characteristic of smooth gradients)
        small_grad_x = ((grad_x_flat > 0.3) & (grad_x_flat < 15)).sum()
        small_grad_y = ((grad_y_flat > 0.3) & (grad_y_flat < 15)).sum()

        # Check EACH direction separately - a horizontal gradient only has gradients in x
        x_small_ratio = small_grad_x / len(grad_x_flat) if len(grad_x_flat) > 0 else 0
        y_small_ratio = small_grad_y / len(grad_y_flat) if len(grad_y_flat) > 0 else 0

        # If EITHER direction has dominant small gradients, it's likely a gradient image
        if unique_count > 50:
            if x_small_ratio > 0.7 or y_small_ratio > 0.7:
                max_ratio = max(x_small_ratio, y_small_ratio)
                return True, f"Image contains gradients ({max_ratio*100:.1f}% gradient in one direction)"

        # Also check combined ratio with lower threshold
        total_grad_pixels = len(grad_x_flat) + len(grad_y_flat)
        small_grad_ratio = (small_grad_x + small_grad_y) / total_grad_pixels if total_grad_pixels > 0 else 0

        if small_grad_ratio >= 0.4 and unique_count > 100:
            return True, f"Image contains gradients ({small_grad_ratio*100:.1f}% gradient area)"

        # Method 3: Check for photographic content
        if unique_count > 3000:
            oklab_colors = srgb_to_oklab(unique_colors.astype(np.float64))
            l_values = oklab_colors[:, 0]

            hist, _ = np.histogram(l_values, bins=20, range=(0, 1))
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized[hist_normalized > 0] *
                             np.log2(hist_normalized[hist_normalized > 0]))

            a_range = oklab_colors[:, 1].max() - oklab_colors[:, 1].min()
            b_range = oklab_colors[:, 2].max() - oklab_colors[:, 2].min()

            if entropy > 3.5 and unique_count > 5000 and (a_range > 0.25 or b_range > 0.25):
                return True, "Image appears to be photographic content"

        return False, None

    def _filter_antialiasing(
        self, pixels: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_], int]:
        """Phase 3: Filter anti-aliasing artifacts using spatial coherence.

        Anti-aliasing creates edge pixels that blend between adjacent colors.
        These shouldn't count as separate colors. We detect them by:
        1. Finding edges using Sobel operator
        2. Identifying pixels near edges that don't form coherent regions
        3. Marking these as anti-aliasing artifacts

        Args:
            pixels: RGB pixel array (H, W, 3)

        Returns:
            Tuple of (pixels, boolean mask of AA pixels, count of unique AA colors)
        """
        h, w = pixels.shape[:2]
        total_pixels = h * w

        # Convert to grayscale for edge detection
        gray = np.mean(pixels.astype(np.float32), axis=2)

        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize and threshold
        max_edge = edge_magnitude.max()
        if max_edge > 0:
            edge_magnitude = edge_magnitude / max_edge
        edge_mask = edge_magnitude > self.AA_EDGE_THRESHOLD

        # Dilate edge mask to catch pixels near edges
        edge_dilated = ndimage.binary_dilation(edge_mask, iterations=2)

        # Find unique colors and their regions
        flat_pixels = pixels.reshape(-1, 3)
        unique_colors, inverse_indices, counts = np.unique(
            flat_pixels, axis=0, return_inverse=True, return_counts=True
        )
        color_map = inverse_indices.reshape(h, w)

        # On large/high-color images, skip expensive per-color component analysis.
        # Using the edge mask directly keeps runtime bounded while still excluding edges.
        if len(unique_colors) > 128 or total_pixels > 1_000_000:
            aa_mask = edge_dilated
            return pixels, aa_mask, 0

        # Identify core colors: those that form significant coherent regions
        # away from edges
        core_color_mask = np.zeros(len(unique_colors), dtype=bool)
        aa_mask = np.zeros((h, w), dtype=bool)

        for color_idx in range(len(unique_colors)):
            color_mask = color_map == color_idx
            pixel_count = counts[color_idx]

            if pixel_count == 0:
                continue

            # Check if this color exists significantly away from edges
            away_from_edges = color_mask & ~edge_dilated
            away_from_edges_count = away_from_edges.sum()

            # Check connected components
            labeled, num_features = ndimage.label(color_mask)
            if num_features == 0:
                continue

            component_sizes = ndimage.sum(color_mask, labeled, range(1, num_features + 1))
            max_component_size = max(component_sizes) if len(component_sizes) > 0 else 0

            # A color is a "core" color (not AA) if:
            # 1. It has significant presence away from edges, OR
            # 2. It forms large coherent regions, OR
            # 3. It covers a significant percentage of the image
            is_core = (
                away_from_edges_count > 20 or
                max_component_size > 100 or
                pixel_count > total_pixels * 0.02
            )

            if is_core:
                core_color_mask[color_idx] = True
            else:
                # Mark as anti-aliasing
                aa_mask |= color_mask

        aa_colors_count = (~core_color_mask).sum()
        return pixels, aa_mask, aa_colors_count

    def _cluster_colors(
        self, pixels: NDArray[np.uint8], aa_mask: NDArray[np.bool_]
    ) -> tuple[list[ColorInfo], str]:
        """Phase 4: Cluster colors in OKLab space using DBSCAN.

        Args:
            pixels: RGB pixel array (H, W, 3)
            aa_mask: Boolean mask of anti-aliased pixels

        Returns:
            Tuple of (list of ColorInfo, confidence level)
        """
        h, w = pixels.shape[:2]
        total_pixels = h * w

        # Get unique colors and their counts, excluding AA pixels for clustering
        flat_pixels = pixels.reshape(-1, 3)
        flat_aa_mask = aa_mask.reshape(-1)

        # Use non-AA pixels for finding core colors
        non_aa_pixels = flat_pixels[~flat_aa_mask]

        if len(non_aa_pixels) == 0:
            # All pixels are AA - use all pixels
            non_aa_pixels = flat_pixels

        unique_colors, inverse_indices, counts = np.unique(
            non_aa_pixels, axis=0, return_inverse=True, return_counts=True
        )

        if len(unique_colors) == 0:
            return [], "low"

        # Convert to OKLab
        oklab_colors = srgb_to_oklab(unique_colors.astype(np.float64))

        # Get DBSCAN parameters for this mode
        params = self.PARAMS[self.mode]
        epsilon = params["epsilon"]

        # Adjust min_samples based on image size and unique color count
        # For images with few unique colors, use min_samples=1
        if len(unique_colors) < 50:
            min_samples = 1
        else:
            min_samples = max(1, len(non_aa_pixels) // params["min_pts_divisor"])
            # Cap min_samples to avoid requiring too many samples
            min_samples = min(min_samples, 10)

        # Run DBSCAN
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(oklab_colors)

        # Group colors by cluster
        cluster_colors: dict[int, list[tuple[NDArray, int]]] = {}
        for idx, label in enumerate(labels):
            if label not in cluster_colors:
                cluster_colors[label] = []
            cluster_colors[label].append((unique_colors[idx], counts[idx]))

        # Handle noise points (label -1)
        noise_colors = cluster_colors.pop(-1, [])

        if noise_colors:
            if self.mode == DetectionMode.AGGRESSIVE:
                # Assign noise to nearest cluster
                if cluster_colors:
                    cluster_centroids = {}
                    for label, colors_counts in cluster_colors.items():
                        colors_arr = np.array([c for c, _ in colors_counts])
                        counts_arr = np.array([cnt for _, cnt in colors_counts])
                        centroid = np.average(
                            srgb_to_oklab(colors_arr.astype(np.float64)),
                            axis=0,
                            weights=counts_arr
                        )
                        cluster_centroids[label] = centroid

                    for color, count in noise_colors:
                        color_oklab = srgb_to_oklab(color.astype(np.float64).reshape(1, 3))[0]
                        min_dist = float('inf')
                        nearest_label = list(cluster_colors.keys())[0]
                        for label, centroid in cluster_centroids.items():
                            dist = np.linalg.norm(color_oklab - centroid)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_label = label
                        cluster_colors[nearest_label].append((color, count))
                else:
                    # No clusters, treat all noise as one cluster
                    cluster_colors[0] = noise_colors
            else:
                # Conservative mode: significant noise points become their own color
                total_non_aa = len(non_aa_pixels)
                for i, (color, count) in enumerate(noise_colors):
                    # Only include if it represents significant coverage
                    if count > total_non_aa * 0.005:  # 0.5% threshold
                        new_label = max(cluster_colors.keys(), default=-1) + 1 + i
                        cluster_colors[new_label] = [(color, count)]

        if not cluster_colors:
            return [], "low"

        # Build ColorInfo list
        colors: list[ColorInfo] = []
        for label, colors_counts in cluster_colors.items():
            colors_arr = np.array([c for c, _ in colors_counts])
            counts_arr = np.array([cnt for _, cnt in colors_counts])

            total_count = counts_arr.sum()

            # Weighted average color (in sRGB)
            avg_rgb = np.average(colors_arr, axis=0, weights=counts_arr)
            avg_rgb = np.clip(avg_rgb, 0, 255).astype(np.uint8)

            # Convert to OKLab for storage
            avg_oklab = srgb_to_oklab(avg_rgb.astype(np.float64).reshape(1, 3))[0]

            rgb_tuple = tuple(int(v) for v in avg_rgb)
            hex_code = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

            colors.append(ColorInfo(
                rgb=rgb_tuple,  # type: ignore
                hex=hex_code,
                oklab=tuple(float(v) for v in avg_oklab),  # type: ignore
                pixel_count=int(total_count),
                percentage=(total_count / total_pixels) * 100
            ))

        # Sort by pixel count (most prominent first)
        colors.sort(key=lambda c: c.pixel_count, reverse=True)

        # Merge very similar colors in post-processing for aggressive mode
        if self.mode == DetectionMode.AGGRESSIVE and len(colors) > 1:
            colors = self._merge_similar_colors(colors, threshold=0.08)

        # Determine confidence
        confidence = self._assess_confidence(colors, total_pixels)

        return colors, confidence

    def _merge_similar_colors(
        self, colors: list[ColorInfo], threshold: float
    ) -> list[ColorInfo]:
        """Merge colors that are very similar in OKLab space."""
        if len(colors) <= 1:
            return colors

        merged = []
        used = set()

        for i, c1 in enumerate(colors):
            if i in used:
                continue

            # Find all colors similar to this one
            similar_group = [(c1.rgb, c1.pixel_count)]
            used.add(i)

            for j, c2 in enumerate(colors):
                if j in used:
                    continue

                dist = np.linalg.norm(
                    np.array(c1.oklab) - np.array(c2.oklab)
                )
                if dist < threshold:
                    similar_group.append((c2.rgb, c2.pixel_count))
                    used.add(j)

            # Merge the similar colors
            total_count = sum(cnt for _, cnt in similar_group)
            avg_rgb = np.average(
                [rgb for rgb, _ in similar_group],
                axis=0,
                weights=[cnt for _, cnt in similar_group]
            )
            avg_rgb = np.clip(avg_rgb, 0, 255).astype(np.uint8)
            avg_oklab = srgb_to_oklab(avg_rgb.astype(np.float64).reshape(1, 3))[0]

            rgb_tuple = tuple(int(v) for v in avg_rgb)
            hex_code = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

            merged.append(ColorInfo(
                rgb=rgb_tuple,  # type: ignore
                hex=hex_code,
                oklab=tuple(float(v) for v in avg_oklab),  # type: ignore
                pixel_count=total_count,
                percentage=colors[0].percentage  # Will recalculate
            ))

        # Recalculate percentages
        total = sum(c.pixel_count for c in merged)
        for c in merged:
            c.percentage = (c.pixel_count / total) * 100 if total > 0 else 0

        merged.sort(key=lambda c: c.pixel_count, reverse=True)
        return merged

    def _assess_confidence(
        self, colors: list[ColorInfo], total_pixels: int
    ) -> str:
        """Assess confidence level of the detection.

        Args:
            colors: Detected colors
            total_pixels: Total pixel count

        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if not colors:
            return "low"

        # Check color coverage
        total_coverage = sum(c.percentage for c in colors)

        # Check minimum color size
        min_coverage = min(c.percentage for c in colors)

        # Check color separation in OKLab
        if len(colors) >= 2:
            oklabs = np.array([c.oklab for c in colors])
            min_separation = float('inf')
            for i in range(len(oklabs)):
                for j in range(i + 1, len(oklabs)):
                    dist = np.linalg.norm(oklabs[i] - oklabs[j])
                    min_separation = min(min_separation, dist)
        else:
            min_separation = 1.0

        # Scoring
        score = 0

        if total_coverage > 95:
            score += 2
        elif total_coverage > 80:
            score += 1

        if min_coverage > 5:
            score += 2
        elif min_coverage > 1:
            score += 1

        if min_separation > 0.1:
            score += 2
        elif min_separation > 0.05:
            score += 1

        if len(self._flags) == 0:
            score += 1
        elif "has_semi_transparent_pixels" in self._flags:
            score -= 1

        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def _unsuitable_result(self, reason: str) -> DetectionResult:
        """Create a result for unsuitable images."""
        return DetectionResult(
            color_count=-1,
            colors=[],
            flags=self._flags,
            confidence="low",
            mode=self.mode.value,
            unsuitable_reason=reason
        )
