"""Generate test images for screen print color detection testing.

Creates ~100 test images across various categories:
- Simple logos (solid colors, no AA)
- Logos with anti-aliasing
- Gradients (linear, radial)
- Transparency variations
- Photographic-like content
- Edge cases
"""

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Output directory
OUTPUT_DIR = Path("test_images")

# Test image manifest - stores expected results
MANIFEST: list[dict] = []


def save_image(img: Image.Image, category: str, name: str, expected_colors: int,
               notes: str = "", expected_flags: list[str] | None = None):
    """Save image and add to manifest."""
    category_dir = OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{name}.png"
    filepath = category_dir / filename

    img.save(filepath, "PNG")

    MANIFEST.append({
        "path": str(filepath),
        "category": category,
        "name": name,
        "expected_colors": expected_colors,
        "expected_flags": expected_flags or [],
        "notes": notes
    })

    print(f"  Created: {filepath}")


def create_solid_color_images():
    """Create simple solid color test images."""
    print("\nGenerating solid color images...")

    # Single solid color
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    save_image(img, "solid", "single_red", 1, "Single solid red color")

    # Two colors - split
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 200], fill=(0, 0, 255))
    save_image(img, "solid", "two_colors_split", 2, "Blue and white split")

    # Three colors - stripes
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 200], fill=(255, 0, 0))
    draw.rectangle([100, 0, 200, 200], fill=(0, 255, 0))
    save_image(img, "solid", "three_colors_stripes", 3, "RGB stripes")

    # Four colors - quadrants
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 100], fill=(255, 0, 0))
    draw.rectangle([100, 0, 200, 100], fill=(0, 255, 0))
    draw.rectangle([0, 100, 100, 200], fill=(0, 0, 255))
    draw.rectangle([100, 100, 200, 200], fill=(255, 255, 0))
    save_image(img, "solid", "four_colors_quadrants", 4, "RGBY quadrants")

    # Five colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    img = Image.new("RGB", (250, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        draw.rectangle([i*50, 0, (i+1)*50, 200], fill=color)
    save_image(img, "solid", "five_colors_stripes", 5, "5 color stripes")

    # Six colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        draw.rectangle([i*50, 0, (i+1)*50, 200], fill=color)
    save_image(img, "solid", "six_colors_stripes", 6, "6 color stripes")

    # Eight colors (max allowed)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        draw.rectangle([i*50, 0, (i+1)*50, 200], fill=color)
    save_image(img, "solid", "eight_colors_max", 8, "8 colors - maximum allowed")

    # Similar colors (should merge in aggressive mode)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 200], fill=(255, 0, 0))      # Red
    draw.rectangle([100, 0, 200, 200], fill=(250, 10, 10))  # Slightly different red
    save_image(img, "solid", "similar_reds", 1,  # They're close enough to merge
               "Two similar reds - should merge")

    # More distinct reds (should NOT merge)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 200], fill=(255, 0, 0))      # Bright red
    draw.rectangle([100, 0, 200, 200], fill=(180, 0, 0))    # Dark red
    save_image(img, "solid", "distinct_reds", 2,
               "Two distinct reds - should NOT merge")


def create_logo_shapes():
    """Create logo-like shapes with solid colors."""
    print("\nGenerating logo shapes...")

    # Circle on background
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 170, 170], fill=(0, 100, 200))
    save_image(img, "logos", "circle_on_white", 2, "Blue circle on white")

    # Multiple shapes
    img = Image.new("RGB", (300, 200), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 40, 100, 160], fill=(255, 0, 0))
    draw.rectangle([110, 40, 190, 160], fill=(0, 150, 0))
    draw.polygon([(200, 160), (250, 40), (290, 160)], fill=(0, 0, 200))
    save_image(img, "logos", "shapes_three_colors", 4, "Circle, square, triangle on gray")

    # Nested shapes
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 180, 180], fill=(0, 0, 150))
    draw.ellipse([50, 50, 150, 150], fill=(255, 255, 255))
    draw.ellipse([80, 80, 120, 120], fill=(255, 0, 0))
    save_image(img, "logos", "nested_circles", 3, "Nested circles - blue, white, red")

    # Text-like blocks (simulating text without actual text)
    img = Image.new("RGB", (300, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Simulate text with rectangles
    for i in range(5):
        x = 20 + i * 55
        draw.rectangle([x, 30, x + 40, 70], fill=(30, 30, 30))
    save_image(img, "logos", "text_blocks", 2, "Simulated text - black on white")

    # Star shape
    img = Image.new("RGB", (200, 200), (20, 20, 80))
    draw = ImageDraw.Draw(img)
    # 5-pointed star
    cx, cy, r = 100, 100, 70
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        radius = r if i % 2 == 0 else r * 0.4
        points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
    draw.polygon(points, fill=(255, 215, 0))
    save_image(img, "logos", "gold_star_on_navy", 2, "Gold star on navy background")


def create_antialiased_images():
    """Create images with anti-aliased edges."""
    print("\nGenerating anti-aliased images...")

    # Anti-aliased circle (PIL's default is aliased, we need to do this manually)
    # Create at higher resolution and scale down
    scale = 4
    img_large = Image.new("RGB", (200 * scale, 200 * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img_large)
    draw.ellipse([30*scale, 30*scale, 170*scale, 170*scale], fill=(200, 50, 50))
    img = img_large.resize((200, 200), Image.LANCZOS)
    save_image(img, "antialiased", "circle_aa", 2,
               "Anti-aliased circle - should detect 2 colors",
               expected_flags=["filtered_antialiasing"])

    # Anti-aliased diagonal line
    scale = 4
    img_large = Image.new("RGB", (200 * scale, 200 * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img_large)
    draw.line([(20*scale, 180*scale), (180*scale, 20*scale)], fill=(0, 0, 0), width=8*scale)
    img = img_large.resize((200, 200), Image.LANCZOS)
    save_image(img, "antialiased", "diagonal_line_aa", 2,
               "Anti-aliased diagonal line",
               expected_flags=["filtered_antialiasing"])

    # Anti-aliased text-like shapes
    scale = 4
    img_large = Image.new("RGB", (300 * scale, 100 * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img_large)
    # Curved shapes to simulate text
    for i in range(4):
        x = 30*scale + i * 70*scale
        draw.ellipse([x, 20*scale, x + 50*scale, 80*scale], fill=(50, 50, 50))
    img = img_large.resize((300, 100), Image.LANCZOS)
    save_image(img, "antialiased", "smooth_shapes_aa", 2,
               "Smooth shapes with AA",
               expected_flags=["filtered_antialiasing"])

    # Multiple colors with AA edges
    scale = 4
    img_large = Image.new("RGB", (300 * scale, 200 * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img_large)
    draw.ellipse([20*scale, 40*scale, 100*scale, 160*scale], fill=(255, 50, 50))
    draw.ellipse([110*scale, 40*scale, 190*scale, 160*scale], fill=(50, 255, 50))
    draw.ellipse([200*scale, 40*scale, 280*scale, 160*scale], fill=(50, 50, 255))
    img = img_large.resize((300, 200), Image.LANCZOS)
    save_image(img, "antialiased", "three_circles_aa", 4,
               "Three AA circles on white",
               expected_flags=["filtered_antialiasing"])

    # Heavy AA (more blur)
    img_large = Image.new("RGB", (200 * 8, 200 * 8), (255, 255, 255))
    draw = ImageDraw.Draw(img_large)
    draw.ellipse([30*8, 30*8, 170*8, 170*8], fill=(0, 100, 180))
    img = img_large.resize((200, 200), Image.LANCZOS)
    save_image(img, "antialiased", "circle_heavy_aa", 2,
               "Circle with heavy anti-aliasing",
               expected_flags=["filtered_antialiasing"])


def create_gradient_images():
    """Create gradient images that should be rejected."""
    print("\nGenerating gradient images...")

    # Linear gradient horizontal
    img = Image.new("RGB", (200, 200))
    for x in range(200):
        r = int(255 * x / 199)
        for y in range(200):
            img.putpixel((x, y), (r, 0, 0))
    save_image(img, "gradients", "linear_horizontal", -1,
               "Linear gradient - should be rejected")

    # Linear gradient vertical
    img = Image.new("RGB", (200, 200))
    for y in range(200):
        g = int(255 * y / 199)
        for x in range(200):
            img.putpixel((x, y), (0, g, 0))
    save_image(img, "gradients", "linear_vertical", -1,
               "Vertical gradient - should be rejected")

    # Radial gradient
    img = Image.new("RGB", (200, 200))
    cx, cy = 100, 100
    max_dist = math.sqrt(100**2 + 100**2)
    for y in range(200):
        for x in range(200):
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            intensity = int(255 * (1 - min(dist / max_dist, 1)))
            img.putpixel((x, y), (intensity, intensity, 255))
    save_image(img, "gradients", "radial", -1,
               "Radial gradient - should be rejected")

    # Diagonal gradient
    img = Image.new("RGB", (200, 200))
    for y in range(200):
        for x in range(200):
            val = int(255 * (x + y) / 398)
            img.putpixel((x, y), (val, val, val))
    save_image(img, "gradients", "diagonal", -1,
               "Diagonal gradient - should be rejected")

    # Two-color gradient
    img = Image.new("RGB", (200, 200))
    for x in range(200):
        t = x / 199
        r = int(255 * (1 - t))
        b = int(255 * t)
        for y in range(200):
            img.putpixel((x, y), (r, 0, b))
    save_image(img, "gradients", "two_color_gradient", -1,
               "Red to blue gradient - should be rejected")

    # Subtle gradient (wider range, should be detected)
    img = Image.new("RGB", (200, 200))
    for x in range(200):
        val = 80 + int(95 * x / 199)  # Range from 80 to 175
        for y in range(200):
            img.putpixel((x, y), (val, val, val))
    save_image(img, "gradients", "subtle_gradient", -1,
               "Subtle gradient - should be rejected")


def create_transparency_images():
    """Create images with various transparency scenarios."""
    print("\nGenerating transparency images...")

    # Simple transparency - circle on transparent
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 170, 170], fill=(255, 0, 0, 255))
    save_image(img, "transparency", "circle_on_transparent", 2,
               "Red circle on transparent (composites to white bg)",
               expected_flags=["has_transparency"])

    # Multiple opaque shapes on transparent
    img = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 40, 100, 160], fill=(255, 0, 0, 255))
    draw.rectangle([110, 40, 190, 160], fill=(0, 200, 0, 255))
    draw.polygon([(200, 160), (250, 40), (290, 160)], fill=(0, 0, 255, 255))
    save_image(img, "transparency", "shapes_on_transparent", 4,
               "Three shapes on transparent",
               expected_flags=["has_transparency"])

    # Semi-transparent overlay
    img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 200, 100], fill=(0, 0, 255, 255))
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0, 128))  # 50% transparent red
    save_image(img, "transparency", "semi_transparent_overlay", 3,
               "Semi-transparent red circle",
               expected_flags=["has_transparency", "has_semi_transparent_pixels"])

    # Varying transparency levels - creates 5 distinct blue shades when composited
    img = Image.new("RGBA", (250, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, alpha in enumerate([255, 200, 150, 100, 50]):
        x = i * 50
        draw.rectangle([x, 0, x + 50, 200], fill=(0, 0, 200, alpha))
    save_image(img, "transparency", "varying_alpha", 5,
               "Varying transparency - creates 5 distinct blue shades",
               expected_flags=["has_transparency", "has_semi_transparent_pixels"])

    # Transparency with anti-aliasing
    scale = 4
    img_large = Image.new("RGBA", (200 * scale, 200 * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_large)
    draw.ellipse([30*scale, 30*scale, 170*scale, 170*scale], fill=(0, 150, 100, 255))
    img = img_large.resize((200, 200), Image.LANCZOS)
    save_image(img, "transparency", "aa_on_transparent", 2,
               "AA circle on transparent",
               expected_flags=["has_transparency", "has_semi_transparent_pixels", "filtered_antialiasing"])


def create_complex_images():
    """Create complex images that test edge cases."""
    print("\nGenerating complex images...")

    # Checkerboard with larger squares (easier to detect)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                draw.rectangle([i*50, j*50, (i+1)*50, (j+1)*50], fill=(100, 100, 200))
    save_image(img, "complex", "checkerboard", 2, "Checkerboard pattern - 2 colors")

    # Small but visible color region (white bg + gray + red = 3 colors)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 198, 198], fill=(200, 200, 200))
    # Add a small but visible red square
    draw.rectangle([80, 80, 120, 120], fill=(255, 0, 0))
    save_image(img, "complex", "small_accent_region", 3,
               "Gray with red accent (white border visible)")

    # Near-white colors (very similar - will likely merge)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 100, 200], fill=(250, 250, 250))
    draw.rectangle([100, 0, 200, 200], fill=(245, 248, 250))
    save_image(img, "complex", "near_whites", 1,  # So similar they should merge
               "Very similar whites - should merge")

    # High contrast neighbors
    img = Image.new("RGB", (200, 200), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Alternating 1-pixel stripes
    for x in range(0, 200, 2):
        draw.line([(x, 0), (x, 200)], fill=(255, 255, 255))
    save_image(img, "complex", "pixel_stripes", 2,
               "1-pixel alternating stripes")

    # Dithered appearance
    img = Image.new("RGB", (200, 200))
    for y in range(200):
        for x in range(200):
            if (x + y) % 3 == 0:
                img.putpixel((x, y), (255, 0, 0))
            elif (x + y) % 3 == 1:
                img.putpixel((x, y), (0, 255, 0))
            else:
                img.putpixel((x, y), (0, 0, 255))
    save_image(img, "complex", "dithered_rgb", 3, "Dithered RGB pattern")


def create_photographic_images():
    """Create photographic-like images that should be rejected."""
    print("\nGenerating photographic-like images...")

    # Noise (simulates photo grain)
    img = Image.new("RGB", (200, 200))
    pixels = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    save_image(img, "photographic", "noise", -1,
               "Random noise - should be rejected")

    # Blurred noise (less blur to retain complexity)
    pixels = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    save_image(img, "photographic", "blurred_noise", -1,
               "Lightly blurred noise - should be rejected")

    # Realistic color distribution (simulated photo)
    img = Image.new("RGB", (200, 200))
    for y in range(200):
        for x in range(200):
            # Simulate natural color variation
            base_r = 100 + int(50 * math.sin(x / 20))
            base_g = 150 + int(30 * math.cos(y / 25))
            base_b = 120 + int(40 * math.sin((x + y) / 30))
            noise = random.randint(-10, 10)
            r = max(0, min(255, base_r + noise))
            g = max(0, min(255, base_g + noise))
            b = max(0, min(255, base_b + noise))
            img.putpixel((x, y), (r, g, b))
    save_image(img, "photographic", "simulated_photo", -1,
               "Simulated photographic content")

    # Many unique colors (but not gradient)
    img = Image.new("RGB", (200, 200))
    random.seed(42)
    for y in range(200):
        for x in range(200):
            # Random but clustered around a few hues
            hue_choice = random.choice([0, 120, 240])  # R, G, or B dominant
            variation = random.randint(-30, 30)
            if hue_choice == 0:
                img.putpixel((x, y), (200 + variation, 50 + variation//2, 50 + variation//2))
            elif hue_choice == 120:
                img.putpixel((x, y), (50 + variation//2, 200 + variation, 50 + variation//2))
            else:
                img.putpixel((x, y), (50 + variation//2, 50 + variation//2, 200 + variation))
    save_image(img, "photographic", "noisy_colors", -1,
               "Many unique colors with noise")


def create_edge_case_images():
    """Create edge case test images."""
    print("\nGenerating edge case images...")

    # Single pixel
    img = Image.new("RGB", (1, 1), (255, 0, 0))
    save_image(img, "edge_cases", "single_pixel", 1, "Single red pixel")

    # Very small image
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 5, 10], fill=(0, 0, 255))
    save_image(img, "edge_cases", "tiny_image", 2, "10x10 pixel image")

    # Large solid color
    img = Image.new("RGB", (1000, 1000), (128, 0, 255))
    save_image(img, "edge_cases", "large_solid", 1, "Large single color")

    # Almost all one color with small accent
    img = Image.new("RGB", (200, 200), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([90, 90, 110, 110], fill=(255, 200, 0))
    save_image(img, "edge_cases", "dominant_with_accent", 2,
               "Large dark area with small accent")

    # Black and white only
    img = Image.new("RGB", (200, 200), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 170, 170], fill=(255, 255, 255))
    save_image(img, "edge_cases", "black_white", 2, "Pure black and white")

    # Nine colors (just over limit)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    img = Image.new("RGB", (450, 200))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        draw.rectangle([i*50, 0, (i+1)*50, 200], fill=color)
    save_image(img, "edge_cases", "nine_colors", -1,
               "9 colors - exceeds maximum")

    # All transparent
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    save_image(img, "edge_cases", "fully_transparent", 1,
               "Fully transparent - composites to white",
               expected_flags=["has_transparency"])

    # Grayscale steps (4 distinct levels - more achievable)
    img = Image.new("RGB", (200, 200))
    draw = ImageDraw.Draw(img)
    grays = [0, 85, 170, 255]  # Well-spaced grays
    for i, gray in enumerate(grays):
        draw.rectangle([i*50, 0, (i+1)*50, 200], fill=(gray, gray, gray))
    save_image(img, "edge_cases", "grayscale_steps", 4,
               "4 distinct gray levels")


def create_realistic_logo_simulations():
    """Create more realistic logo simulations."""
    print("\nGenerating realistic logo simulations...")

    # Sports team style logo
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Background shape
    draw.polygon([(50, 180), (150, 20), (250, 180)], fill=(0, 50, 100))
    # Inner element
    draw.ellipse([100, 60, 200, 140], fill=(255, 200, 0))
    # Outline effect
    draw.polygon([(52, 178), (150, 22), (248, 178)], outline=(200, 200, 200), width=3)
    save_image(img, "realistic", "sports_logo", 4,
               "Sports team style logo")

    # Corporate logo style
    img = Image.new("RGB", (300, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Icon
    draw.rectangle([20, 20, 80, 80], fill=(0, 100, 200))
    # "Text" blocks
    for i in range(4):
        draw.rectangle([100 + i*50, 35, 140 + i*50, 65], fill=(50, 50, 50))
    save_image(img, "realistic", "corporate_logo", 3,
               "Corporate logo with icon and text")

    # Badge/emblem style
    scale = 4
    img_large = Image.new("RGB", (200 * scale, 200 * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img_large)
    # Outer circle
    draw.ellipse([10*scale, 10*scale, 190*scale, 190*scale], fill=(0, 50, 100))
    # Inner circle
    draw.ellipse([20*scale, 20*scale, 180*scale, 180*scale], fill=(255, 255, 255))
    # Center emblem
    draw.ellipse([60*scale, 60*scale, 140*scale, 140*scale], fill=(200, 0, 50))
    img = img_large.resize((200, 200), Image.LANCZOS)
    save_image(img, "realistic", "badge_emblem", 3,
               "Badge/emblem with AA",
               expected_flags=["filtered_antialiasing"])

    # Vintage style
    img = Image.new("RGB", (300, 200), (245, 235, 220))  # Cream background
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 270, 170], outline=(80, 40, 20), width=5)
    draw.rectangle([50, 50, 250, 150], fill=(180, 50, 30))
    # "Text" area
    draw.rectangle([70, 70, 230, 130], fill=(245, 235, 220))
    save_image(img, "realistic", "vintage_style", 3,
               "Vintage style logo")

    # Minimalist logo
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 160, 160], fill=(0, 0, 0))
    draw.rectangle([70, 85, 130, 115], fill=(255, 255, 255))
    save_image(img, "realistic", "minimalist", 2,
               "Minimalist black and white logo")


def create_additional_tests():
    """Create additional test images for better coverage."""
    print("\nGenerating additional test images...")

    # Concentric rectangles
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], fill=(0, 0, 128))
    draw.rectangle([40, 40, 160, 160], fill=(255, 255, 255))
    draw.rectangle([60, 60, 140, 140], fill=(0, 0, 128))
    draw.rectangle([80, 80, 120, 120], fill=(255, 255, 255))
    save_image(img, "additional", "concentric_rectangles", 2,
               "Concentric rectangles - 2 colors")

    # Polka dots
    img = Image.new("RGB", (200, 200), (255, 200, 200))
    draw = ImageDraw.Draw(img)
    for i in range(4):
        for j in range(4):
            x, y = 25 + i * 50, 25 + j * 50
            draw.ellipse([x-15, y-15, x+15, y+15], fill=(200, 50, 50))
    save_image(img, "additional", "polka_dots", 2, "Polka dot pattern")

    # Horizontal stripes (7 colors)
    img = Image.new("RGB", (200, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 0, 0), (255, 127, 0), (255, 255, 0),
        (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)
    ]
    for i, color in enumerate(colors):
        draw.rectangle([0, i*40, 200, (i+1)*40], fill=color)
    save_image(img, "additional", "rainbow_stripes", 7, "Rainbow stripes - 7 colors")

    # Simple icon
    img = Image.new("RGB", (200, 200), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # House shape
    draw.polygon([(100, 30), (30, 100), (170, 100)], fill=(180, 60, 60))  # Roof
    draw.rectangle([50, 100, 150, 180], fill=(100, 80, 60))  # Body
    draw.rectangle([80, 130, 120, 180], fill=(200, 180, 100))  # Door
    save_image(img, "additional", "simple_house_icon", 4, "Simple house icon")

    # Arrow shape
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 20), (180, 100), (140, 100), (140, 180),
                  (60, 180), (60, 100), (20, 100)], fill=(0, 150, 0))
    save_image(img, "additional", "arrow_shape", 2, "Arrow shape")

    # Cross/plus shape
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 20, 120, 180], fill=(200, 0, 0))
    draw.rectangle([20, 80, 180, 120], fill=(200, 0, 0))
    save_image(img, "additional", "cross_shape", 2, "Cross/plus shape")

    # Rings/Olympic style
    img = Image.new("RGB", (300, 180), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    colors = [(0, 129, 188), (0, 0, 0), (237, 51, 78), (252, 177, 49), (0, 157, 87)]
    positions = [(30, 20), (110, 20), (190, 20), (70, 60), (150, 60)]
    for (x, y), color in zip(positions, colors):
        draw.ellipse([x, y, x+60, y+60], outline=color, width=8)
    save_image(img, "additional", "olympic_rings", 6, "Olympic-style rings")

    # Triangle pattern
    img = Image.new("RGB", (200, 200), (50, 50, 100))
    draw = ImageDraw.Draw(img)
    draw.polygon([(0, 200), (100, 0), (200, 200)], fill=(200, 200, 50))
    save_image(img, "additional", "triangle_bg", 2, "Triangle on background")

    # Grid of small squares (single color)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(5):
        for j in range(5):
            draw.rectangle([i*40+5, j*40+5, i*40+35, j*40+35], fill=(70, 70, 180))
    save_image(img, "additional", "grid_squares", 2, "Grid of squares")

    # Diagonal stripes
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(-200, 400, 40):
        draw.polygon([(i, 0), (i+20, 0), (i+220, 200), (i+200, 200)], fill=(0, 100, 100))
    save_image(img, "additional", "diagonal_stripes", 2, "Diagonal stripes")

    # Hexagon
    img = Image.new("RGB", (200, 200), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    cx, cy, r = 100, 100, 70
    points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=(255, 100, 50))
    save_image(img, "additional", "hexagon", 2, "Hexagon shape")

    # Three overlapping circles (no transparency, just touching)
    img = Image.new("RGB", (250, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 40, 120, 140], fill=(255, 100, 100))
    draw.ellipse([70, 40, 170, 140], fill=(100, 255, 100))
    draw.ellipse([120, 40, 220, 140], fill=(100, 100, 255))
    save_image(img, "additional", "three_overlapping_circles", 4,
               "Three overlapping circles")

    # Banner/ribbon shape
    img = Image.new("RGB", (300, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(0, 30), (30, 50), (0, 70), (270, 70), (300, 50), (270, 30)],
                 fill=(220, 50, 50))
    save_image(img, "additional", "ribbon_banner", 2, "Ribbon banner shape")

    # Shield shape
    img = Image.new("RGB", (200, 250), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 230), (20, 80), (20, 20), (180, 20), (180, 80)],
                 fill=(0, 80, 160))
    draw.polygon([(100, 200), (40, 80), (40, 40), (160, 40), (160, 80)],
                 fill=(255, 215, 0))
    save_image(img, "additional", "shield_shape", 3, "Shield shape")

    # Zigzag pattern (solid colors, no curves)
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    draw = ImageDraw.Draw(img)
    # Simple zigzag stripes
    for y_start in range(0, 200, 50):
        points = [(0, y_start), (50, y_start + 25), (100, y_start),
                  (150, y_start + 25), (200, y_start),
                  (200, y_start + 25), (150, y_start + 50), (100, y_start + 25),
                  (50, y_start + 50), (0, y_start + 25)]
        draw.polygon(points, fill=(50, 100, 150))
    save_image(img, "additional", "zigzag_pattern", 2, "Zigzag stripe pattern")

    # Bullseye target
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (255, 255, 255)]
    for i, radius in enumerate([90, 70, 50, 30, 10]):
        color = colors[i % 2]
        draw.ellipse([100-radius, 100-radius, 100+radius, 100+radius], fill=color)
    save_image(img, "additional", "bullseye_target", 2, "Bullseye target")

    # Speech bubble (gray bg + white bubble + black outline = may show as 2-3)
    img = Image.new("RGB", (200, 180), (200, 200, 200))  # More distinct gray
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 180, 120], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    draw.polygon([(60, 115), (80, 115), (50, 160)], fill=(255, 255, 255))
    draw.polygon([(60, 118), (75, 118), (55, 150)], fill=(255, 255, 255))
    save_image(img, "additional", "speech_bubble", 3, "Speech bubble")

    # Heart shape
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Approximate heart with circles and triangle
    draw.ellipse([40, 50, 110, 120], fill=(220, 20, 60))
    draw.ellipse([90, 50, 160, 120], fill=(220, 20, 60))
    draw.polygon([(40, 95), (160, 95), (100, 170)], fill=(220, 20, 60))
    save_image(img, "additional", "heart_shape", 2, "Heart shape")

    # Starburst
    img = Image.new("RGB", (200, 200), (30, 30, 80))
    draw = ImageDraw.Draw(img)
    cx, cy = 100, 100
    for i in range(12):
        angle = i * math.pi / 6
        x1 = cx + 30 * math.cos(angle)
        y1 = cy + 30 * math.sin(angle)
        x2 = cx + 80 * math.cos(angle)
        y2 = cy + 80 * math.sin(angle)
        draw.line([(cx, cy), (x2, y2)], fill=(255, 220, 50), width=8)
    draw.ellipse([70, 70, 130, 130], fill=(255, 220, 50))
    save_image(img, "additional", "starburst", 2, "Starburst pattern")

    # Price tag shape
    img = Image.new("RGB", (200, 150), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(20, 20), (160, 20), (180, 75), (160, 130), (20, 130)],
                 fill=(255, 100, 0))
    draw.ellipse([25, 65, 45, 85], fill=(255, 255, 255))
    save_image(img, "additional", "price_tag", 2, "Price tag shape")


def create_print_specific_tests():
    """Create tests specific to screen printing scenarios."""
    print("\nGenerating print-specific tests...")

    # Halftone pattern (simulated)
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(0, 200, 8):
        for x in range(0, 200, 8):
            # Varying dot sizes create tonal variation
            size = 2 + int(4 * (x + y) / 400)
            draw.ellipse([x, y, x + size, y + size], fill=(0, 0, 0))
    save_image(img, "print_specific", "halftone", 2,
               "Halftone pattern - should be 2 colors")

    # Simulated CMY separation (3 stripes, no background showing)
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Cyan-ish
    draw.rectangle([0, 0, 100, 200], fill=(0, 180, 220))
    # Magenta-ish
    draw.rectangle([100, 0, 200, 200], fill=(220, 0, 180))
    # Yellow
    draw.rectangle([200, 0, 300, 200], fill=(255, 230, 0))
    save_image(img, "print_specific", "cmy_separation", 3,
               "CMY color separation - 3 colors")

    # White ink on dark (simulated)
    img = Image.new("RGB", (200, 200), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 160, 160], fill=(255, 255, 255))
    draw.rectangle([70, 70, 130, 130], fill=(255, 200, 0))
    save_image(img, "print_specific", "white_on_dark", 3,
               "White and color on dark - test dark substrate mode")

    # Registration marks (common in print files) - thicker marks
    img = Image.new("RGB", (220, 220), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Main design
    draw.rectangle([40, 40, 180, 180], fill=(0, 100, 200))
    # Registration marks in corners - thicker
    for x, y in [(5, 5), (195, 5), (5, 195), (195, 195)]:
        draw.ellipse([x, y, x+20, y+20], outline=(0, 0, 0), width=3)
        draw.line([(x+10, y), (x+10, y+20)], fill=(0, 0, 0), width=2)
        draw.line([(x, y+10), (x+20, y+10)], fill=(0, 0, 0), width=2)
    save_image(img, "print_specific", "with_reg_marks", 3,
               "Design with registration marks")


def main():
    """Generate all test images."""
    print("Screen Print Color Detection - Test Image Generator")
    print("=" * 50)

    # Clear output directory
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Generate all categories
    create_solid_color_images()
    create_logo_shapes()
    create_antialiased_images()
    create_gradient_images()
    create_transparency_images()
    create_complex_images()
    create_photographic_images()
    create_edge_case_images()
    create_realistic_logo_simulations()
    create_additional_tests()
    create_print_specific_tests()

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(MANIFEST, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Generated {len(MANIFEST)} test images")
    print(f"Manifest saved to: {manifest_path}")

    # Summary by category
    categories = {}
    for item in MANIFEST:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nImages by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
