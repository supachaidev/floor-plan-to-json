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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Non-scale-dependent constants
# --------------------------------------------------------------------------- #
BINARY_THRESHOLD = 127            # grayscale cutoff between wall (black) and empty (white)
ANGLE_TOL_DEG = 3.0               # max deviation from 0/90 deg to accept as axis-aligned
HINGE_SEARCH_OFFSETS = (1, 2, 3)  # try hinges this far outside the detected gap

# Fallback used when wall-thickness estimation fails (empty mask etc.).
# Reference thickness on the generator's images: the generator asks cv2.line
# for 6-px strokes, but anti-aliasing / line endcaps fatten the binarized
# result to ~8 px. All derived-tunable ratios in from_thickness() are
# calibrated against t=8 so the auto path reproduces the original hand-tuned
# constants exactly for generator output.
_DEFAULT_THICKNESS_PX = 8


Segment = Tuple[int, int, int, int]             # (x1, y1, x2, y2)
AxisSeg = Tuple[int, int, int]                  # (shared_coord, lo, hi)
Point2 = Tuple[int, int]
Wall = Tuple[Point2, Point2]                    # ((x0, y0), (x1, y1))


# --------------------------------------------------------------------------- #
# Scale-adaptive tunables
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VectorizerConfig:
    """All pixel-scale vectorizer thresholds, derived from wall stroke width.

    At wall_thickness_px=6 and door_width_px=60, from_thickness() reproduces
    the original generator-tuned defaults exactly. Other scales scale the
    thresholds proportionally so the same pipeline handles plans drawn at
    different resolutions without hand-tuning.
    """
    wall_thickness_px: int
    door_width_px: int
    close_kernel: int
    open_kernel: int
    collinear_tol: int
    merge_gap: int
    hough_threshold: int
    hough_min_len: int
    hough_max_gap: int
    door_gap_min: int
    door_gap_max: int
    door_end_margin: int
    leaf_min_length: int
    leaf_leading_white: int
    leaf_max_steps: int

    @classmethod
    def from_thickness(
        cls,
        wall_thickness_px: int,
        door_width_px: Optional[int] = None,
        image_width_px: Optional[int] = None,
    ) -> "VectorizerConfig":
        """Derive all thresholds from an explicit binary-mask wall thickness.

        door_width_px defaults to max(7.5 x wall thickness, 5% of image width),
        so larger images get proportionally larger door openings. collinear_tol
        also gets an image-width floor so that hand-drawn drift on large images
        (walls meant-to-be-collinear but offset by several px) still merges.

        Ratios are calibrated so from_thickness(8, 60, 1024) reproduces the
        original hand-tuned constants exactly.
        """
        t = max(2, int(wall_thickness_px))
        w = int(image_width_px) if image_width_px is not None else 0
        if door_width_px is not None:
            door = int(door_width_px)
        else:
            door = max(int(round(7.5 * t)), int(round(0.05 * w)))
        # Image-width floor on collinear tolerance so large canvases can tolerate
        # larger absolute drift between intended-collinear hand-drawn segments.
        collinear = max(int(round(0.625 * t)), int(round(0.004 * w)))
        return cls(
            wall_thickness_px=t,
            door_width_px=door,
            # Small closing kernel bridges 1-2 px gaps (pen stipple, rendering
            # holes) without closing real openings like doors or room gaps.
            close_kernel=3,
            # Opening kernel must be smaller than the wall stroke so walls
            # survive while thin door-leaf / arc strokes are erased.
            open_kernel=max(3, t // 3),
            collinear_tol=max(3, collinear),
            merge_gap=max(40, int(round(4.0 * door / 3.0))),
            hough_threshold=max(40, 10 * t),
            hough_min_len=max(20, 5 * t),
            hough_max_gap=max(10, int(round(2.5 * t))),
            door_gap_min=max(20, int(round(2.0 * door / 3.0))),
            door_gap_max=max(30, int(round(1.5 * door))),
            door_end_margin=max(10, int(round(2.5 * t))),
            leaf_min_length=max(20, int(round(0.75 * door))),
            leaf_leading_white=max(5, int(round(1.25 * t))),
            leaf_max_steps=door + 10,
        )

    @classmethod
    def auto(
        cls,
        raw_mask: np.ndarray,
        door_width_px: Optional[int] = None,
    ) -> "VectorizerConfig":
        """Estimate wall thickness from a binary mask and derive a scaled config."""
        t = _estimate_wall_thickness_px(raw_mask)
        image_width_px = int(raw_mask.shape[1]) if raw_mask.ndim >= 2 else None
        return cls.from_thickness(
            t,
            door_width_px=door_width_px,
            image_width_px=image_width_px,
        )


def _estimate_wall_thickness_px(raw_mask: np.ndarray) -> int:
    """Estimate wall stroke thickness in pixels from a binary foreground mask.

    Uses 2 x p99 of the distance transform over foreground pixels: inside a
    stroke of thickness T, distance-to-edge maxes at ~T/2 along the stroke's
    midline. Thin door strokes pull the distribution's lower tail down but
    don't affect p99; corners / T-junctions that slightly exceed T/2 are rare
    enough that p99 stays close to the true wall half-thickness.
    """
    if raw_mask.size == 0 or not np.any(raw_mask):
        return _DEFAULT_THICKNESS_PX
    dt = cv2.distanceTransform(raw_mask, cv2.DIST_L2, 3)
    vals = dt[raw_mask > 0]
    half = float(np.percentile(vals, 99))
    return max(2, int(round(half * 2)))


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


def _bridge_mask(raw_mask: np.ndarray, cfg: VectorizerConfig) -> np.ndarray:
    """Close small axial gaps in the mask (stipple / pen skips / anti-alias
    holes) while leaving large openings like door gaps untouched.

    Horizontal close then vertical close; 1D kernels so a bridge in one axis
    doesn't accidentally leak into the perpendicular axis.
    """
    k = cfg.close_kernel
    if k <= 1:
        return raw_mask
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    m = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, hk)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, vk)


def _wall_mask(raw_mask: np.ndarray, cfg: VectorizerConfig) -> np.ndarray:
    """Morphological open on the raw mask — strips thin door strokes."""
    k = cfg.open_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------------------------------- #
# Hough detection
# --------------------------------------------------------------------------- #
def _hough_segments(mask: np.ndarray, cfg: VectorizerConfig) -> List[Segment]:
    """Run Probabilistic Hough on the wall mask and return raw line segments."""
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_len,
        maxLineGap=cfg.hough_max_gap,
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
    cfg: VectorizerConfig,
) -> List[AxisSeg]:
    """Given axis-aligned segments (row, lo, hi), group ones sharing a row
    (within cfg.collinear_tol of the cluster's running mean) and union their
    [lo, hi] spans when gaps are <= cfg.merge_gap. Rows are weighted by span
    length so the cluster's representative coordinate converges to the
    dominant wall line."""
    if not segs:
        return []
    row_tol = cfg.collinear_tol
    gap_tol = cfg.merge_gap

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
    coverage: List[bool], axis_coords: List[int], cfg: VectorizerConfig
) -> List[Tuple[int, int]]:
    """Return interior gaps (lo, hi) in the coverage array whose size falls
    within [cfg.door_gap_min, cfg.door_gap_max], clipped away from wall
    endpoints by cfg.door_end_margin."""
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
            if cfg.door_gap_min <= size <= cfg.door_gap_max:
                lo_coord = axis_coords[gap_lo_idx]
                hi_coord = axis_coords[gap_hi_idx]
                if (
                    lo_coord > axis_coords[0] + cfg.door_end_margin
                    and hi_coord < axis_coords[-1] - cfg.door_end_margin
                ):
                    gaps.append((lo_coord, hi_coord))
    return gaps


def _scan_leaf(
    raw_mask: np.ndarray, x0: int, y0: int, dx: int, dy: int, cfg: VectorizerConfig
) -> int:
    """Stepping (dx, dy) from (x0, y0), skip up to cfg.leaf_leading_white
    initial white pixels (the wall stroke / cleanup zone can absorb a few),
    then count the contiguous run of black pixels that follows. A single
    white pixel ends the count."""
    h_img, w_img = raw_mask.shape
    step = 1
    while step <= cfg.leaf_leading_white:
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
    while step <= cfg.leaf_max_steps:
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
    raw_mask: np.ndarray, wall: Wall, gap: Tuple[int, int], cfg: VectorizerConfig
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
                count = _scan_leaf(raw_mask, hinge[0], hinge[1], dx, dy, cfg)
                if best is None or count > best[0]:
                    best = (count, hinge, (dx, dy))

    assert best is not None
    leaf_len, hinge, (dx, dy) = best
    side_offset = HINGE_SEARCH_OFFSETS[0]

    if leaf_len >= cfg.leaf_min_length:
        # Confident CAD-style: a clear perpendicular leaf is drawn at a hinge
        # just outside one gap boundary. Use it to orient the opening.
        if orient == "h":
            other = (gap_hi + side_offset, y) if hinge[0] < gap_lo else (gap_lo - side_offset, y)
        else:
            other = (x1, gap_hi + side_offset) if hinge[1] < gap_lo else (x1, gap_lo - side_offset)
        start, end = sorted([hinge, other])
    else:
        # Gap-only fallback: no perpendicular leaf (e.g. hand-drawn arc doors).
        # Use gap boundaries as the opening span and the gap midpoint as a
        # nominal hinge; swing_to keeps the best-scoring perpendicular even if
        # it only saw a stub of ink.
        if orient == "h":
            start = (gap_lo - side_offset, y)
            end = (gap_hi + side_offset, y)
            hinge = ((gap_lo + gap_hi) // 2, y)
        else:
            start = (x1, gap_lo - side_offset)
            end = (x1, gap_hi + side_offset)
            hinge = (x1, (gap_lo + gap_hi) // 2)

    swing_to = (hinge[0] + dx * cfg.door_width_px, hinge[1] + dy * cfg.door_width_px)
    return start, end, hinge, swing_to


def _detect_doors(
    wall_mask: np.ndarray,
    raw_mask: np.ndarray,
    walls: List[Wall],
    cfg: VectorizerConfig,
) -> List[Door]:
    doors: List[Door] = []
    for idx, wall in enumerate(walls):
        coverage, axis_coords = _coverage_along_wall(wall_mask, wall)
        for gap in _find_gaps(coverage, axis_coords, cfg):
            resolved = _locate_door(raw_mask, wall, gap, cfg)
            if resolved is None:
                continue
            start, end, hinge, swing_to = resolved
            doors.append(Door(idx, start, end, hinge, swing_to))
    return doors


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _filter_stub_walls(walls: List[Wall], cfg: VectorizerConfig) -> List[Wall]:
    """Drop short perpendicular stubs (door-post flanks, drawing artifacts)
    whose length is below the expected door width and that have exactly one
    endpoint touching another wall's line."""
    if len(walls) < 2:
        return walls
    length_threshold = cfg.door_width_px
    tol = max(cfg.collinear_tol, cfg.wall_thickness_px)

    def _on_wall(px: int, py: int, other: Wall) -> bool:
        (x1, y1), (x2, y2) = other
        if y1 == y2:
            return (
                abs(py - y1) <= tol
                and min(x1, x2) - tol <= px <= max(x1, x2) + tol
            )
        return (
            abs(px - x1) <= tol
            and min(y1, y2) - tol <= py <= max(y1, y2) + tol
        )

    kept: List[Wall] = []
    for w in walls:
        (x1, y1), (x2, y2) = w
        length = abs(x2 - x1) if y1 == y2 else abs(y2 - y1)
        if length < length_threshold:
            others = [o for o in walls if o is not w]
            p1_on = any(_on_wall(x1, y1, o) for o in others)
            p2_on = any(_on_wall(x2, y2, o) for o in others)
            if p1_on != p2_on:
                continue
        kept.append(w)
    return kept


def _walls_from_hough(wall_mask: np.ndarray, cfg: VectorizerConfig) -> List[Wall]:
    raw = _hough_segments(wall_mask, cfg)

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
    for y, lo, hi in _cluster_and_merge(horizontals, cfg):
        walls.append(((lo, y), (hi, y)))
    for x, lo, hi in _cluster_and_merge(verticals, cfg):
        walls.append(((x, lo), (x, hi)))
    return walls


def vectorize(
    image_path: Path,
    wall_thickness_px: Optional[int] = None,
    door_width_px: Optional[int] = None,
) -> List[Wall]:
    """Extract clean wall endpoints from a floor-plan image.

    If wall_thickness_px is None, it is estimated from the image.
    """
    walls, _, _, _ = vectorize_plan(
        image_path,
        wall_thickness_px=wall_thickness_px,
        door_width_px=door_width_px,
    )
    return walls


def vectorize_plan(
    image_path: Path,
    wall_thickness_px: Optional[int] = None,
    door_width_px: Optional[int] = None,
) -> Tuple[List[Wall], List[Door], Tuple[int, int], VectorizerConfig]:
    """Extract walls, doors, image size (width, height), and the config used.

    If wall_thickness_px is None, the stroke thickness is estimated from the
    raw binary mask and all other tunables are derived from it.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = gray.shape[:2]

    raw_mask = _binary_mask(gray)
    if wall_thickness_px is not None:
        cfg = VectorizerConfig.from_thickness(
            wall_thickness_px,
            door_width_px=door_width_px,
            image_width_px=w,
        )
    else:
        cfg = VectorizerConfig.auto(raw_mask, door_width_px=door_width_px)

    bridged_mask = _bridge_mask(raw_mask, cfg)
    wall_mask = _wall_mask(bridged_mask, cfg)

    walls = _walls_from_hough(wall_mask, cfg)
    walls = _filter_stub_walls(walls, cfg)
    doors = _detect_doors(wall_mask, raw_mask, walls, cfg)
    return walls, doors, (w, h), cfg


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_plan(
    walls: List[Wall], doors: List[Door], image_path: Path, cfg: VectorizerConfig
) -> None:
    print(
        f"\n{image_path.name}: detected {len(walls)} walls, {len(doors)} doors"
        f"  [wall={cfg.wall_thickness_px}px, door={cfg.door_width_px}px]"
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
    parser.add_argument(
        "--wall-thickness",
        type=int,
        default=None,
        help="Override wall stroke thickness (px). Auto-estimated if omitted.",
    )
    parser.add_argument(
        "--door-width",
        type=int,
        default=None,
        help="Override door opening width (px). Defaults to 10 x wall thickness.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    walls, doors, _, cfg = vectorize_plan(
        image_path,
        wall_thickness_px=args.wall_thickness,
        door_width_px=args.door_width,
    )
    _print_plan(walls, doors, image_path, cfg)


if __name__ == "__main__":
    main()
