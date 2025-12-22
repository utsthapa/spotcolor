"""Test harness for screen print color detection.

Runs all test images and shows accuracy metrics.
"""

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from screenprint import SpotColorDetector, DetectionMode


@dataclass
class TestResult:
    """Result of a single test."""
    path: str
    category: str
    name: str
    expected_colors: int
    actual_colors: int
    expected_flags: list[str]
    actual_flags: list[str]
    confidence: str
    passed: bool
    mode: str
    time_ms: float
    error: str | None = None


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load test manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def run_single_test(
    test_case: dict,
    detector: SpotColorDetector,
    mode_name: str
) -> TestResult:
    """Run a single test case."""
    path = test_case["path"]
    expected = test_case["expected_colors"]
    expected_flags = test_case.get("expected_flags", [])

    start_time = time.perf_counter()
    try:
        result = detector.detect(path)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Check if passed
        # For color count: exact match or within tolerance for edge cases
        color_match = result.color_count == expected

        # For -1 (unsuitable), any -1 result is correct
        if expected == -1 and result.color_count == -1:
            color_match = True

        # Check flags (expected flags should be present, extra flags are OK)
        flags_match = all(
            any(exp_flag in actual_flag for actual_flag in result.flags)
            for exp_flag in expected_flags
        )

        passed = color_match  # Primary criterion is color count

        return TestResult(
            path=path,
            category=test_case["category"],
            name=test_case["name"],
            expected_colors=expected,
            actual_colors=result.color_count,
            expected_flags=expected_flags,
            actual_flags=result.flags,
            confidence=result.confidence,
            passed=passed,
            mode=mode_name,
            time_ms=elapsed_ms
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return TestResult(
            path=path,
            category=test_case["category"],
            name=test_case["name"],
            expected_colors=expected,
            actual_colors=-999,
            expected_flags=expected_flags,
            actual_flags=[],
            confidence="error",
            passed=False,
            mode=mode_name,
            time_ms=elapsed_ms,
            error=str(e)
        )


def print_results_table(results: list[TestResult], title: str):
    """Print results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    # Group by category
    by_category: dict[str, list[TestResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    for category in sorted(by_category.keys()):
        cat_results = by_category[category]
        passed = sum(1 for r in cat_results if r.passed)
        total = len(cat_results)

        print(f"\n  {category.upper()} ({passed}/{total} passed)")
        print(f"  {'-' * 70}")

        for r in cat_results:
            status = "PASS" if r.passed else "FAIL"
            color_info = f"exp={r.expected_colors:2d} got={r.actual_colors:2d}"

            # Truncate name if too long
            name = r.name[:30].ljust(30)

            if r.error:
                print(f"    [{status}] {name} ERROR: {r.error[:30]}")
            elif r.passed:
                print(f"    [{status}] {name} {color_info} ({r.time_ms:.1f}ms)")
            else:
                print(f"    [{status}] {name} {color_info} ({r.time_ms:.1f}ms) <<<")


def print_summary(conservative_results: list[TestResult], aggressive_results: list[TestResult]):
    """Print summary statistics."""
    print(f"\n{'=' * 80}")
    print(" SUMMARY")
    print(f"{'=' * 80}")

    for mode_name, results in [("Conservative", conservative_results), ("Aggressive", aggressive_results)]:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        accuracy = (passed / total * 100) if total > 0 else 0

        # By category
        by_cat: dict[str, tuple[int, int]] = {}
        for r in results:
            if r.category not in by_cat:
                by_cat[r.category] = (0, 0)
            p, t = by_cat[r.category]
            by_cat[r.category] = (p + (1 if r.passed else 0), t + 1)

        # Timing
        total_time = sum(r.time_ms for r in results)
        avg_time = total_time / len(results) if results else 0

        print(f"\n  {mode_name} Mode:")
        print(f"    Overall: {passed}/{total} ({accuracy:.1f}%)")
        print(f"    Avg time: {avg_time:.1f}ms per image")
        print(f"    Total time: {total_time/1000:.2f}s")

        print(f"\n    By category:")
        for cat in sorted(by_cat.keys()):
            p, t = by_cat[cat]
            pct = (p / t * 100) if t > 0 else 0
            bar = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
            print(f"      {cat:20s} [{bar}] {p:2d}/{t:2d} ({pct:5.1f}%)")

    # Comparison
    print(f"\n  Mode Comparison:")
    con_acc = sum(1 for r in conservative_results if r.passed) / len(conservative_results) * 100
    agg_acc = sum(1 for r in aggressive_results if r.passed) / len(aggressive_results) * 100
    print(f"    Conservative: {con_acc:.1f}%")
    print(f"    Aggressive:   {agg_acc:.1f}%")

    # Cases where modes differ
    differ = []
    for c, a in zip(conservative_results, aggressive_results):
        if c.actual_colors != a.actual_colors:
            differ.append((c, a))

    if differ:
        print(f"\n  Cases where modes differ ({len(differ)} total):")
        for c, a in differ[:10]:  # Show first 10
            print(f"    {c.name}: conservative={c.actual_colors}, aggressive={a.actual_colors}")
        if len(differ) > 10:
            print(f"    ... and {len(differ) - 10} more")


def print_failures(results: list[TestResult], mode_name: str):
    """Print detailed info about failures."""
    failures = [r for r in results if not r.passed]
    if not failures:
        return

    print(f"\n{'=' * 80}")
    print(f" FAILURES - {mode_name} Mode ({len(failures)} failures)")
    print(f"{'=' * 80}")

    for r in failures:
        print(f"\n  {r.category}/{r.name}")
        print(f"    Path: {r.path}")
        print(f"    Expected: {r.expected_colors} colors")
        print(f"    Got: {r.actual_colors} colors")
        print(f"    Confidence: {r.confidence}")
        print(f"    Flags: {r.actual_flags}")
        if r.error:
            print(f"    Error: {r.error}")


def main():
    """Run all tests."""
    print("Screen Print Color Detection - Test Harness")
    print("=" * 50)

    # Find manifest
    manifest_path = Path("test_images/manifest.json")
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run generate_test_images.py first to create test images.")
        sys.exit(1)

    # Load manifest
    test_cases = load_manifest(manifest_path)
    print(f"Loaded {len(test_cases)} test cases")

    # Create detectors
    conservative_detector = SpotColorDetector(mode=DetectionMode.CONSERVATIVE)
    aggressive_detector = SpotColorDetector(mode=DetectionMode.AGGRESSIVE)

    # Run tests
    print("\nRunning conservative mode tests...")
    conservative_results = []
    for i, test_case in enumerate(test_cases):
        result = run_single_test(test_case, conservative_detector, "conservative")
        conservative_results.append(result)
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_cases)}...")

    print("\nRunning aggressive mode tests...")
    aggressive_results = []
    for i, test_case in enumerate(test_cases):
        result = run_single_test(test_case, aggressive_detector, "aggressive")
        aggressive_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_cases)}...")

    # Print results
    print_results_table(conservative_results, "CONSERVATIVE MODE RESULTS")
    print_results_table(aggressive_results, "AGGRESSIVE MODE RESULTS")

    # Print summary
    print_summary(conservative_results, aggressive_results)

    # Print failures
    print_failures(conservative_results, "Conservative")
    print_failures(aggressive_results, "Aggressive")

    # Final verdict
    con_passed = sum(1 for r in conservative_results if r.passed)
    agg_passed = sum(1 for r in aggressive_results if r.passed)
    total = len(test_cases)

    print(f"\n{'=' * 80}")
    print(" FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"  Conservative: {con_passed}/{total} tests passed ({con_passed/total*100:.1f}%)")
    print(f"  Aggressive:   {agg_passed}/{total} tests passed ({agg_passed/total*100:.1f}%)")

    # Exit code
    if con_passed == total and agg_passed == total:
        print("\n  All tests passed!")
        return 0
    else:
        print(f"\n  {total - con_passed + total - agg_passed} total failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())
