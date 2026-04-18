"""
vectorizer.py
-------------
Read a rasterised floor plan and recover clean axis-aligned wall segments
and door openings using OpenCV's Probabilistic Hough Line Transform.

Wall pipeline:
    1. Threshold the grayscale image to isolate black wall / door strokes.
    2. Morphological opening to strip thin door-leaf / arc strokes (~1-2 px)
       while preserving thick wall strokes (~6 px).
    3. cv2.HoughLinesP -> fragmented raw line segments.
    4. Filter to near-horizontal / near-vertical segments.
    5. Cluster collinear segments (shared row/column within a small tolerance)
       and merge each cluster into one [min, max] span, bridging any
       door-sized gap.

Door pipeline (per merged wall):
    6. Scan the wall mask along the wall's axis. A contiguous run of zero
       pixels in the wall's interior is a candidate door opening.
    7. For each gap endpoint, count contiguous black pixels perpendicular
       to the wall in the raw (un-opened) mask. The endpoint with a
       leaf-length perpendicular stroke is the hinge; swing_to = hinge
       + perpendicular * DOOR_WIDTH.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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

# HoughLinesP parameters tuned for 1024x768 plans with 6 px-thick walls.
HOUGH_THRESHOLD = 80
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 20

# Door-detection tuning.
WALL_THICKNESS_PX = 6         # matches generator.WALL_THICKNESS
DOOR_WIDTH_PX = 60            # matches generator.DOOR_WIDTH
DOOR_GAP_MIN = 40             # min wall-gap size to consider as a door opening
DOOR_GAP_MAX = 90             # max wall-gap size to consider as a door opening
DOOR_END_MARGIN = 20          # ignore gaps this close to a wall endpoint (avoid corners)
LEAF_MIN_LENGTH = 45          # min contiguous perpendicular black pixels for a leaf
LEAF_LEADING_WHITE = 10       # tolerate up to this many white pixels before the leaf begins
LEAF_MAX_STEPS = DOOR_WIDTH_PX + 10
HINGE_SEARCH_OFFSETS = (1, 2, 3)  # try hinges this far outside the detected gap


Segment = Tuple[int, int, int, int]             # (x1, y1, x2, y2)
AxisSeg = Tuple[int, int, int]                  # (shared_coord, lo, hi)
Point2 = Tuple[int, int]
Wall = Tuple[Point2, Point2]                    # ((x0, y0), (x1, y1))


class Door:
    """A detected door opening on a specific wall."""

    __slots__ = ("wall_index", "start", "end", "hinge", "swing_to")

    def __init__(
        self,
        wall_index: int,
        start: Point2,
        end: Point2,
        hinge: Point2,
        swing_to: Point2,
    ) -> None:
        self.wall_index = wall_index
        self.start = start
        self.end = end
        self.hinge = hinge
        self.swing_to = swing_to

    def as_dict(self) -> dict:
        return {
            "wall_index": self.wall_index,
            "start": self.start,
            "end": self.end,
            "hinge": self.hinge,
            "swing_to": self.swing_to,
        }


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #
def _binary_mask(gray: np.ndarray) -> np.ndarray:
    """Raw inverse threshold: 255 where pixel is dark (wall OR door stroke)."""
    _, mask = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return mask


def _wall_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Morphological open on the raw mask — strips thin door strokes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL, OPEN_KERNEL))
    return cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)


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


def _axis_of(seg: Segment) -> Optional[str]:
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
# Door detection
# --------------------------------------------------------------------------- #
def _coverage_along_wall(
    wall_mask: np.ndarray, wall: Wall, tol: int = 3
) -> Tuple[List[bool], List[int]]:
    """For each integer coordinate along the wall's axis, return True if the
    wall mask has any non-zero pixel within +/- tol rows/cols (so we tolerate
    minor drift in the merged wall's row estimate). Returns (coverage,
    axis_coords)."""
    (x1, y1), (x2, y2) = wall
    h_img, w_img = wall_mask.shape

    coverage: List[bool] = []
    if y1 == y2:  # horizontal
        y = y1
        lo, hi = min(x1, x2), max(x1, x2)
        y_band = slice(max(0, y - tol), min(h_img, y + tol + 1))
        for x in range(lo, hi + 1):
            if 0 <= x < w_img:
                coverage.append(bool(wall_mask[y_band, x].any()))
            else:
                coverage.append(False)
        return coverage, list(range(lo, hi + 1))
    else:  # vertical
        x = x1
        lo, hi = min(y1, y2), max(y1, y2)
        x_band = slice(max(0, x - tol), min(w_img, x + tol + 1))
        for y in range(lo, hi + 1):
            if 0 <= y < h_img:
                coverage.append(bool(wall_mask[y, x_band].any()))
            else:
                coverage.append(False)
        return coverage, list(range(lo, hi + 1))


def _find_gaps(
    coverage: List[bool], axis_coords: List[int]
) -> List[Tuple[int, int]]:
    """Return interior gaps (lo, hi) in the coverage array whose size falls
    within [DOOR_GAP_MIN, DOOR_GAP_MAX], clipped away from wall endpoints."""
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gap_lo_idx = 0
    for i, covered in enumerate(coverage):
        if not covered and not in_gap:
            in_gap = True
            gap_lo_idx = i
        elif covered and in_gap:
            in_gap = False
            gap_hi_idx = i - 1
            size = gap_hi_idx - gap_lo_idx + 1
            if DOOR_GAP_MIN <= size <= DOOR_GAP_MAX:
                lo_coord = axis_coords[gap_lo_idx]
                hi_coord = axis_coords[gap_hi_idx]
                if (
                    lo_coord > axis_coords[0] + DOOR_END_MARGIN
                    and hi_coord < axis_coords[-1] - DOOR_END_MARGIN
                ):
                    gaps.append((lo_coord, hi_coord))
    return gaps


def _scan_leaf(
    raw_mask: np.ndarray, x0: int, y0: int, dx: int, dy: int
) -> int:
    """Stepping (dx, dy) from (x0, y0), skip up to LEAF_LEADING_WHITE initial
    white pixels (the wall stroke / cleanup zone can absorb a few), then
    count the contiguous run of black pixels that follows. A single white
    pixel ends the count."""
    h_img, w_img = raw_mask.shape
    step = 1
    while step <= LEAF_LEADING_WHITE:
        x = x0 + dx * step
        y = y0 + dy * step
        if not (0 <= x < w_img and 0 <= y < h_img):
            return 0
        if raw_mask[y, x] > 0:
            break
        step += 1
    else:
        return 0

    count = 0
    while step <= LEAF_MAX_STEPS:
        x = x0 + dx * step
        y = y0 + dy * step
        if not (0 <= x < w_img and 0 <= y < h_img):
            break
        if raw_mask[y, x] > 0:
            count += 1
            step += 1
        else:
            break
    return count


def _locate_door(
    raw_mask: np.ndarray, wall: Wall, gap: Tuple[int, int]
) -> Optional[Tuple[Point2, Point2, Point2, Point2]]:
    """Given a wall-mask gap on a specific wall, return
    (start, end, hinge, swing_to) if a leaf stroke is visible in the raw mask.
    Returns None if no leaf meets the length threshold (not a door)."""
    (x1, y1), _ = wall
    orient = "h" if wall[0][1] == wall[1][1] else "v"
    gap_lo, gap_hi = gap

    # The true opening boundaries sit one wall-pixel outside the detected gap
    # (opening erosion slightly shrinks the gap; leaf/arc pixels may sit at the
    # wall line itself). Hinge candidates live just outside each gap endpoint.
    if orient == "h":
        y = y1
        lo_side = [(gap_lo - off, y) for off in HINGE_SEARCH_OFFSETS]
        hi_side = [(gap_hi + off, y) for off in HINGE_SEARCH_OFFSETS]
        perp_dirs = ((0, -1), (0, +1))
    else:
        x = x1
        lo_side = [(x, gap_lo - off) for off in HINGE_SEARCH_OFFSETS]
        hi_side = [(x, gap_hi + off) for off in HINGE_SEARCH_OFFSETS]
        perp_dirs = ((-1, 0), (+1, 0))

    best: Optional[Tuple[int, Point2, Tuple[int, int]]] = None
    for candidates in (lo_side, hi_side):
        for hinge in candidates:
            for (dx, dy) in perp_dirs:
                count = _scan_leaf(raw_mask, hinge[0], hinge[1], dx, dy)
                if best is None or count > best[0]:
                    best = (count, hinge, (dx, dy))

    assert best is not None
    leaf_len, hinge, (dx, dy) = best
    if leaf_len < LEAF_MIN_LENGTH:
        return None

    # Report the full opening span: hinge + non-hinge endpoint, both on the
    # wall line. The non-hinge endpoint sits just outside the opposite gap
    # boundary, matching the hinge's distance from its gap boundary.
    side_offset = HINGE_SEARCH_OFFSETS[0]
    if orient == "h":
        other = (gap_hi + side_offset, y) if hinge[0] < gap_lo else (gap_lo - side_offset, y)
    else:
        other = (x1, gap_hi + side_offset) if hinge[1] < gap_lo else (x1, gap_lo - side_offset)
    start, end = sorted([hinge, other])

    swing_to = (hinge[0] + dx * DOOR_WIDTH_PX, hinge[1] + dy * DOOR_WIDTH_PX)
    return start, end, hinge, swing_to


def _detect_doors(
    wall_mask: np.ndarray, raw_mask: np.ndarray, walls: List[Wall]
) -> List[Door]:
    doors: List[Door] = []
    for idx, wall in enumerate(walls):
        coverage, axis_coords = _coverage_along_wall(wall_mask, wall)
        for gap in _find_gaps(coverage, axis_coords):
            resolved = _locate_door(raw_mask, wall, gap)
            if resolved is None:
                continue
            start, end, hinge, swing_to = resolved
            doors.append(Door(idx, start, end, hinge, swing_to))
    return doors


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _walls_from_hough(wall_mask: np.ndarray) -> List[Wall]:
    raw = _hough_segments(wall_mask)

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


def vectorize(image_path: Path) -> List[Wall]:
    """Extract clean wall endpoints from a floor-plan image."""
    walls, _, _ = vectorize_plan(image_path)
    return walls


def vectorize_plan(
    image_path: Path,
) -> Tuple[List[Wall], List[Door], Tuple[int, int]]:
    """Extract walls, doors, and image size (width, height) from an image."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = gray.shape[:2]

    raw_mask = _binary_mask(gray)
    wall_mask = _wall_mask(raw_mask)

    walls = _walls_from_hough(wall_mask)
    doors = _detect_doors(wall_mask, raw_mask, walls)
    return walls, doors, (w, h)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_plan(
    walls: List[Wall], doors: List[Door], image_path: Path
) -> None:
    print(
        f"\n{image_path.name}: detected {len(walls)} walls, {len(doors)} doors"
    )
    print("-" * 72)
    if walls:
        for i, ((x0, y0), (x1, y1)) in enumerate(walls):
            orient = "H" if y0 == y1 else "V"
            length = abs(x1 - x0) + abs(y1 - y0)
            print(
                f"  [{orient}] w{i:02d}: ({x0:>4}, {y0:>4}) -> ({x1:>4}, {y1:>4})"
                f"   length={length}px"
            )
    else:
        print("  (no walls found)")
    if doors:
        print()
        for i, d in enumerate(doors):
            print(
                f"  [D] d{i:02d} on w{d.wall_index:02d}: "
                f"opening {d.start} -> {d.end}   "
                f"hinge={d.hinge}  swing_to={d.swing_to}"
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
    walls, doors, _ = vectorize_plan(image_path)
    _print_plan(walls, doors, image_path)


if __name__ == "__main__":
    main()
