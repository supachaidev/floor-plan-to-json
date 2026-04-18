"""
vectorizer.py
-------------
Read a rasterised floor plan from ./dataset and recover clean, axis-aligned
wall segments using OpenCV's Probabilistic Hough Line Transform.

Pipeline:
    1. Load image as grayscale and threshold black pixels into a binary mask.
    2. Morphological opening to strip thin door-leaf / arc strokes (~1-2px)
       while preserving thick wall strokes (~6px).
    3. cv2.HoughLinesP -> fragmented raw line segments.
    4. Filter to near-horizontal / near-vertical segments (walls are
       axis-aligned in this dataset).
    5. Cluster collinear segments (shared row/column within a small tolerance)
       and merge each cluster into a single [min, max] span, bridging any
       door-sized gap.
    6. Print the final endpoints (x0, y0) -> (x1, y1) to the terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
BINARY_THRESHOLD = 127        # grayscale cutoff between wall (black) and empty (white)
OPEN_KERNEL = 3               # morphological opening removes strokes thinner than this
ANGLE_TOL_DEG = 3.0           # max deviation from 0/90 deg to accept as axis-aligned
COLLINEAR_TOL = 5             # px row/column tolerance for collinear clustering
MERGE_GAP = 80                # px gap bridged when merging collinear segments (> DOOR_WIDTH)

# HoughLinesP parameters tuned for 1024x768 plans with 6px-thick walls.
HOUGH_THRESHOLD = 80
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 20


Segment = Tuple[int, int, int, int]             # (x1, y1, x2, y2)
AxisSeg = Tuple[int, int, int]                  # (shared_coord, lo, hi)
Wall = Tuple[Tuple[int, int], Tuple[int, int]]  # ((x0, y0), (x1, y1))


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #
def _wall_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of thick wall strokes: 255 = wall pixel, 0 = empty."""
    _, mask = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # Opening with a small kernel nibbles away thin door-leaf / arc strokes
    # but leaves the thicker wall strokes intact.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL, OPEN_KERNEL))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------------------------------- #
# Hough detection
# --------------------------------------------------------------------------- #
def _hough_segments(mask: np.ndarray) -> List[Segment]:
    """Run Probabilistic Hough on the wall mask and return raw line segments."""
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in line[0]) for line in lines]


def _axis_of(seg: Segment) -> str | None:
    """Return 'h' for near-horizontal, 'v' for near-vertical, None otherwise."""
    x1, y1, x2, y2 = seg
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx == 0 and dy == 0:
        return None
    angle = np.degrees(np.arctan2(dy, dx))   # 0 = horizontal, 90 = vertical
    if angle < ANGLE_TOL_DEG:
        return "h"
    if angle > 90 - ANGLE_TOL_DEG:
        return "v"
    return None


# --------------------------------------------------------------------------- #
# Collinear clustering + merging
# --------------------------------------------------------------------------- #
def _cluster_and_merge(
    segs: List[AxisSeg],
    row_tol: int = COLLINEAR_TOL,
    gap_tol: int = MERGE_GAP,
) -> List[AxisSeg]:
    """Given axis-aligned segments (row, lo, hi), group ones sharing a row
    (within row_tol of the cluster's running mean) and union their [lo, hi]
    spans when gaps are <= gap_tol. Rows are weighted by span length so the
    cluster's representative coordinate converges to the dominant wall line."""
    if not segs:
        return []

    segs_sorted = sorted(segs, key=lambda s: (s[0], s[1]))
    # Each cluster: [r_sum, w_sum, intervals]
    clusters: List[List] = []
    for row, lo, hi in segs_sorted:
        weight = hi - lo + 1
        if clusters:
            r_sum, w_sum, intervals = clusters[-1]
            if abs(row - r_sum / w_sum) <= row_tol:
                clusters[-1] = [r_sum + row * weight, w_sum + weight, intervals + [(lo, hi)]]
                continue
        clusters.append([float(row) * weight, float(weight), [(lo, hi)]])

    merged: List[AxisSeg] = []
    for r_sum, w_sum, intervals in clusters:
        row = int(round(r_sum / w_sum))
        intervals.sort()
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi + gap_tol:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append((row, cur_lo, cur_hi))
                cur_lo, cur_hi = lo, hi
        merged.append((row, cur_lo, cur_hi))
    return merged


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def vectorize(image_path: Path) -> List[Wall]:
    """Extract clean wall endpoints from a floor-plan image."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask = _wall_mask(gray)
    raw = _hough_segments(mask)

    horizontals: List[AxisSeg] = []
    verticals: List[AxisSeg] = []
    for seg in raw:
        axis = _axis_of(seg)
        x1, y1, x2, y2 = seg
        if axis == "h":
            horizontals.append(((y1 + y2) // 2, min(x1, x2), max(x1, x2)))
        elif axis == "v":
            verticals.append(((x1 + x2) // 2, min(y1, y2), max(y1, y2)))

    walls: List[Wall] = []
    for y, lo, hi in _cluster_and_merge(horizontals):
        walls.append(((lo, y), (hi, y)))
    for x, lo, hi in _cluster_and_merge(verticals):
        walls.append(((x, lo), (x, hi)))
    return walls


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_walls(walls: List[Wall], image_path: Path) -> None:
    print(f"\n{image_path.name}: detected {len(walls)} wall segments")
    print("-" * 64)
    if not walls:
        print("  (no walls found)")
        return
    for i, ((x0, y0), (x1, y1)) in enumerate(walls):
        orient = "H" if y0 == y1 else "V"
        length = abs(x1 - x0) + abs(y1 - y0)
        print(
            f"  [{orient}] w{i:02d}: "
            f"({x0:>4}, {y0:>4}) -> ({x1:>4}, {y1:>4})   length={length}px"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        default=str(Path(__file__).parent / "dataset" / "floorplan_000.png"),
        help="Path to floor-plan PNG (default: ./dataset/floorplan_000.png).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    walls = vectorize(image_path)
    _print_walls(walls, image_path)


if __name__ == "__main__":
    main()
