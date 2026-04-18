"""
generator.py
------------
Procedurally generates 2D CAD-style floor plans and exports image / ground-truth
JSON pairs into ./dataset.

Pipeline:
    1. Build a rectangular outer boundary.
    2. Recursively split it via Binary Space Partitioning (BSP) into 2-4 rooms.
    3. Place a door on each interior wall (and one exterior front door).
    4. Rasterise: thick black wall strokes on a white background, with a clean
       white gap for each door.
    5. Emit a JSON file describing every wall and door with start / end
       coordinates in image-pixel space.

The JSON is intentionally simple ("ground truth"-flavored, RoomPlan-inspired):

    {
      "image_size":      {"width": int, "height": int},
      "wall_thickness":  int,
      "rooms":           [{"id": "r0", "polygon": [[x, y], ...],
                           "bbox": {x_min,y_min,x_max,y_max}, "area": float,
                           "centroid": [x, y]}],
      "walls":           [{"id": "w0", "type": "exterior"|"interior",
                           "start": {x,y}, "end": {x,y}, "length": float}],
      "doors":           [{"id": "d0", "wall_id": "wN",
                           "type": "exterior"|"interior",
                           "start": {x,y}, "end": {x,y}, "width": int,
                           "hinge":    {x,y},   # pivot endpoint of the opening
                           "swing_to": {x,y}}]  # door-leaf tip when open
    }
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, box


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
IMG_WIDTH = 1024
IMG_HEIGHT = 768
MARGIN = 60                # padding between boundary and image edge
WALL_THICKNESS = 6         # px
DOOR_WIDTH = 60            # px (size of the gap left in a wall)
MIN_ROOM_DIM = 160         # min room side length, ensures BSP cuts are valid
DOOR_END_BUFFER = 30       # min distance between a door and the nearest corner
DOOR_LEAF_THICKNESS = 2    # px, line showing the door in open position
DOOR_ARC_THICKNESS = 1     # px, quarter-circle swing arc


Rect = Tuple[int, int, int, int]   # (x0, y0, x1, y1) inclusive of corners


# --------------------------------------------------------------------------- #
# Floor-plan generation (BSP)
# --------------------------------------------------------------------------- #
@dataclass
class _Wall:
    id: str
    type: str           # "exterior" | "interior"
    start: Tuple[int, int]
    end: Tuple[int, int]


@dataclass
class _Door:
    id: str
    wall_id: str
    type: str           # "exterior" | "interior"
    start: Tuple[int, int]
    end: Tuple[int, int]
    hinge: Tuple[int, int]      # pivot endpoint of the opening
    swing_to: Tuple[int, int]   # door-leaf tip when the door is open


_SWING_DELTAS: Dict[str, Tuple[int, int]] = {
    "+x": (DOOR_WIDTH, 0),
    "-x": (-DOOR_WIDTH, 0),
    "+y": (0, DOOR_WIDTH),
    "-y": (0, -DOOR_WIDTH),
}


def _build_door(
    rng: random.Random,
    door_id: str,
    wall_id: str,
    door_type: str,
    opening_a: Tuple[int, int],
    opening_b: Tuple[int, int],
    swing_options: List[str],
) -> _Door:
    """Pick a random hinge end and swing direction, returning a fully-specified door."""
    hinge = rng.choice([opening_a, opening_b])
    swing = rng.choice(swing_options)
    dx, dy = _SWING_DELTAS[swing]
    swing_to = (hinge[0] + dx, hinge[1] + dy)
    return _Door(door_id, wall_id, door_type, opening_a, opening_b, hinge, swing_to)


def _subtract_forbidden(
    lo: int, hi: int, forbidden: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return sub-ranges of [lo, hi] after removing each forbidden (f_lo, f_hi)."""
    if lo > hi:
        return []
    clamped = sorted(
        (max(f_lo, lo), min(f_hi, hi))
        for f_lo, f_hi in forbidden
        if f_hi >= lo and f_lo <= hi
    )
    merged: List[List[int]] = []
    for f_lo, f_hi in clamped:
        if merged and f_lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], f_hi)
        else:
            merged.append([f_lo, f_hi])
    result: List[Tuple[int, int]] = []
    cursor = lo
    for f_lo, f_hi in merged:
        if f_lo > cursor:
            result.append((cursor, f_lo - 1))
        cursor = f_hi + 1
    if cursor <= hi:
        result.append((cursor, hi))
    return [(a, b) for a, b in result if a <= b]


def _pick_from_ranges(
    rng: random.Random, ranges: List[Tuple[int, int]]
) -> int | None:
    """Uniform-random draw from the union of [a, b] integer ranges, or None if empty."""
    if not ranges:
        return None
    total = sum(b - a + 1 for a, b in ranges)
    r = rng.randint(0, total - 1)
    for a, b in ranges:
        if r <= b - a:
            return a + r
        r -= b - a + 1
    return None  # unreachable


def _doors_on_line(
    doors: List[_Door], axis: str, value: int
) -> List[Tuple[int, int]]:
    """Return door-opening intervals for doors that lie on the given axis-aligned line.
    axis='x' -> vertical line x=value (returns y-intervals);
    axis='y' -> horizontal line y=value (returns x-intervals)."""
    intervals: List[Tuple[int, int]] = []
    for d in doors:
        (sx, sy), (ex, ey) = d.start, d.end
        if axis == "x" and sx == ex == value:
            intervals.append(tuple(sorted([sy, ey])))
        elif axis == "y" and sy == ey == value:
            intervals.append(tuple(sorted([sx, ex])))
    return intervals


def _pick_split_coord(
    rng: random.Random, rect: Rect, axis: str, doors: List[_Door]
) -> int | None:
    """Pick a split coordinate that doesn't place the new wall's endpoints inside
    any existing door opening along the perpendicular edges of `rect`."""
    x0, y0, x1, y1 = rect
    if axis == "v":
        lo, hi = x0 + MIN_ROOM_DIM, x1 - MIN_ROOM_DIM
        forbidden = _doors_on_line(doors, "y", y0) + _doors_on_line(doors, "y", y1)
    else:
        lo, hi = y0 + MIN_ROOM_DIM, y1 - MIN_ROOM_DIM
        forbidden = _doors_on_line(doors, "x", x0) + _doors_on_line(doors, "x", x1)
    return _pick_from_ranges(rng, _subtract_forbidden(lo, hi, forbidden))


def _find_split(
    rng: random.Random, rects: List[Rect], doors: List[_Door]
) -> Tuple[int, str, int] | None:
    """Search for a (rect_idx, axis, coord) triple that yields a valid, collision-free split."""
    for i, r in enumerate(rects):
        w, h = r[2] - r[0], r[3] - r[1]
        axes: List[str] = []
        if w >= 2 * MIN_ROOM_DIM:
            axes.append("v")
        if h >= 2 * MIN_ROOM_DIM:
            axes.append("h")
        if not axes:
            continue
        # Prefer the longer side, with occasional variety.
        if len(axes) == 2:
            preferred = "v" if w >= h else "h"
            if rng.random() < 0.25:
                preferred = "h" if preferred == "v" else "v"
            axes = [preferred, "v" if preferred == "h" else "h"]
        for axis in axes:
            coord = _pick_split_coord(rng, r, axis, doors)
            if coord is not None:
                return i, axis, coord
    return None


def _bsp_split(
    rng: random.Random, num_rooms: int
) -> Tuple[List[Rect], List[_Wall], List[_Door]]:
    """Split the outer boundary into `num_rooms` rectangles via greedy BSP.

    Returns (rooms, interior_walls, interior_doors).
    """
    bx0, by0 = MARGIN, MARGIN
    bx1, by1 = IMG_WIDTH - MARGIN, IMG_HEIGHT - MARGIN

    rects: List[Rect] = [(bx0, by0, bx1, by1)]
    walls: List[_Wall] = []
    doors: List[_Door] = []

    next_wall = 4   # exterior walls occupy w0..w3
    next_door = 1   # exterior front door is d0

    while len(rects) < num_rooms:
        # Largest first, so the biggest rooms get carved up before smaller ones.
        rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        split = _find_split(rng, rects, doors)
        if split is None:
            break  # no rect can be subdivided without colliding with an existing door
        chosen_idx, axis, coord = split
        x0, y0, x1, y1 = rects.pop(chosen_idx)

        if axis == "v":
            split_x = coord
            wall = _Wall(
                id=f"w{next_wall}",
                type="interior",
                start=(split_x, y0),
                end=(split_x, y1),
            )
            door_y0 = rng.randint(
                y0 + DOOR_END_BUFFER, y1 - DOOR_END_BUFFER - DOOR_WIDTH
            )
            door = _build_door(
                rng,
                door_id=f"d{next_door}",
                wall_id=wall.id,
                door_type="interior",
                opening_a=(split_x, door_y0),
                opening_b=(split_x, door_y0 + DOOR_WIDTH),
                swing_options=["+x", "-x"],
            )
            rects.append((x0, y0, split_x, y1))
            rects.append((split_x, y0, x1, y1))
        else:  # axis == "h"
            split_y = coord
            wall = _Wall(
                id=f"w{next_wall}",
                type="interior",
                start=(x0, split_y),
                end=(x1, split_y),
            )
            door_x0 = rng.randint(
                x0 + DOOR_END_BUFFER, x1 - DOOR_END_BUFFER - DOOR_WIDTH
            )
            door = _build_door(
                rng,
                door_id=f"d{next_door}",
                wall_id=wall.id,
                door_type="interior",
                opening_a=(door_x0, split_y),
                opening_b=(door_x0 + DOOR_WIDTH, split_y),
                swing_options=["+y", "-y"],
            )
            rects.append((x0, y0, x1, split_y))
            rects.append((x0, split_y, x1, y1))

        walls.append(wall)
        doors.append(door)
        next_wall += 1
        next_door += 1

    return rects, walls, doors


def _exterior_walls() -> List[_Wall]:
    bx0, by0 = MARGIN, MARGIN
    bx1, by1 = IMG_WIDTH - MARGIN, IMG_HEIGHT - MARGIN
    return [
        _Wall("w0", "exterior", (bx0, by0), (bx1, by0)),  # top
        _Wall("w1", "exterior", (bx1, by0), (bx1, by1)),  # right
        _Wall("w2", "exterior", (bx0, by1), (bx1, by1)),  # bottom
        _Wall("w3", "exterior", (bx0, by0), (bx0, by1)),  # left
    ]


def _t_junctions_on_exterior(interior_walls: List[_Wall]) -> Dict[str, List[int]]:
    """Map each exterior wall id to the coordinates of T-junctions where an
    interior wall endpoint meets it."""
    top_y, bot_y = MARGIN, IMG_HEIGHT - MARGIN
    left_x, right_x = MARGIN, IMG_WIDTH - MARGIN
    result: Dict[str, List[int]] = {"w0": [], "w1": [], "w2": [], "w3": []}
    for wall in interior_walls:
        for (px, py) in (wall.start, wall.end):
            if py == top_y:
                result["w0"].append(px)
            if py == bot_y:
                result["w2"].append(px)
            if px == left_x:
                result["w3"].append(py)
            if px == right_x:
                result["w1"].append(py)
    return result


def _exterior_door(
    rng: random.Random,
    walls: List[_Wall],
    t_junctions: Dict[str, List[int]],
) -> _Door:
    """Cut a single front-door opening into one of the exterior walls, swinging
    inward and avoiding any interior-wall T-junction on that exterior wall."""
    order = list(walls)
    rng.shuffle(order)
    for wall in order:
        sx, sy = wall.start
        ex, ey = wall.end
        tj = t_junctions.get(wall.id, [])
        # The door occupies [d_lo, d_lo + DOOR_WIDTH]; blocking any T-junction
        # t means d_lo must avoid [t - DOOR_WIDTH, t].
        forbidden = [(t - DOOR_WIDTH, t) for t in tj]
        if sy == ey:                                    # horizontal exterior wall
            a, b = sorted([sx, ex])
            start_lo = a + DOOR_END_BUFFER + 20
            start_hi = b - DOOR_END_BUFFER - 20 - DOOR_WIDTH
            d_lo = _pick_from_ranges(rng, _subtract_forbidden(start_lo, start_hi, forbidden))
            if d_lo is None:
                continue
            inward = "+y" if sy == MARGIN else "-y"
            return _build_door(
                rng, "d0", wall.id, "exterior",
                (d_lo, sy), (d_lo + DOOR_WIDTH, sy),
                swing_options=[inward],
            )
        else:                                           # vertical exterior wall
            a, b = sorted([sy, ey])
            start_lo = a + DOOR_END_BUFFER + 20
            start_hi = b - DOOR_END_BUFFER - 20 - DOOR_WIDTH
            d_lo = _pick_from_ranges(rng, _subtract_forbidden(start_lo, start_hi, forbidden))
            if d_lo is None:
                continue
            inward = "+x" if sx == MARGIN else "-x"
            return _build_door(
                rng, "d0", wall.id, "exterior",
                (sx, d_lo), (sx, d_lo + DOOR_WIDTH),
                swing_options=[inward],
            )
    raise RuntimeError("No exterior wall has room for a front door without a T-junction collision.")


def generate_floor_plan(seed: int | None = None) -> Dict[str, Any]:
    """Build a floor plan and return its ground-truth dictionary."""
    rng = random.Random(seed)

    num_rooms = rng.randint(2, 4)
    room_rects, interior_walls, interior_doors = _bsp_split(rng, num_rooms)

    walls = _exterior_walls() + interior_walls
    t_junctions = _t_junctions_on_exterior(interior_walls)
    doors = [_exterior_door(rng, _exterior_walls(), t_junctions)] + interior_doors

    # Build rooms (use shapely for area / centroid).
    rooms = []
    for i, (x0, y0, x1, y1) in enumerate(room_rects):
        poly: Polygon = box(x0, y0, x1, y1)
        cx, cy = poly.centroid.coords[0]
        rooms.append({
            "id": f"r{i}",
            "polygon": [[int(x), int(y)] for (x, y) in poly.exterior.coords[:-1]],
            "bbox": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
            "area": round(poly.area, 2),
            "centroid": [round(cx, 2), round(cy, 2)],
        })

    def _wall_dict(w: _Wall) -> Dict[str, Any]:
        length = LineString([w.start, w.end]).length
        return {
            "id": w.id,
            "type": w.type,
            "start": {"x": int(w.start[0]), "y": int(w.start[1])},
            "end":   {"x": int(w.end[0]),   "y": int(w.end[1])},
            "length": round(length, 2),
        }

    def _door_dict(d: _Door) -> Dict[str, Any]:
        return {
            "id": d.id,
            "wall_id": d.wall_id,
            "type": d.type,
            "start": {"x": int(d.start[0]), "y": int(d.start[1])},
            "end":   {"x": int(d.end[0]),   "y": int(d.end[1])},
            "width": DOOR_WIDTH,
            "hinge":    {"x": int(d.hinge[0]),    "y": int(d.hinge[1])},
            "swing_to": {"x": int(d.swing_to[0]), "y": int(d.swing_to[1])},
        }

    return {
        "image_size": {"width": IMG_WIDTH, "height": IMG_HEIGHT},
        "wall_thickness": WALL_THICKNESS,
        "rooms": rooms,
        "walls": [_wall_dict(w) for w in walls],
        "doors": [_door_dict(d) for d in doors],
    }


# --------------------------------------------------------------------------- #
# Rasterisation
# --------------------------------------------------------------------------- #
def render(plan: Dict[str, Any]) -> np.ndarray:
    """Rasterise the floor plan to a uint8 BGR image."""
    h, w = plan["image_size"]["height"], plan["image_size"]["width"]
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Group doors by the wall they belong to so we can subtract their gaps.
    doors_by_wall: Dict[str, List[Dict[str, Any]]] = {}
    for d in plan["doors"]:
        doors_by_wall.setdefault(d["wall_id"], []).append(d)

    thickness = plan["wall_thickness"]

    for wall in plan["walls"]:
        sx, sy = wall["start"]["x"], wall["start"]["y"]
        ex, ey = wall["end"]["x"], wall["end"]["y"]
        wall_doors = doors_by_wall.get(wall["id"], [])

        if sy == ey:                                  # horizontal wall
            y = sy
            a, b = sorted([sx, ex])
            gaps = sorted(
                (min(d["start"]["x"], d["end"]["x"]),
                 max(d["start"]["x"], d["end"]["x"]))
                for d in wall_doors
            )
            for seg_lo, seg_hi in _subtract_gaps(a, b, gaps):
                cv2.line(img, (seg_lo, y), (seg_hi, y), (0, 0, 0), thickness)
        else:                                         # vertical wall
            x = sx
            a, b = sorted([sy, ey])
            gaps = sorted(
                (min(d["start"]["y"], d["end"]["y"]),
                 max(d["start"]["y"], d["end"]["y"]))
                for d in wall_doors
            )
            for seg_lo, seg_hi in _subtract_gaps(a, b, gaps):
                cv2.line(img, (x, seg_lo), (x, seg_hi), (0, 0, 0), thickness)

    # Overpaint each door gap with a white rectangle to ensure a clean
    # opening (cv2.line caps can otherwise nibble into the gap).
    pad = thickness  # full clearance around the gap
    for d in plan["doors"]:
        sx, sy = d["start"]["x"], d["start"]["y"]
        ex, ey = d["end"]["x"], d["end"]["y"]
        if sy == ey:
            a, b = sorted([sx, ex])
            cv2.rectangle(img, (a, sy - pad), (b, sy + pad), (255, 255, 255), -1)
        else:
            a, b = sorted([sy, ey])
            cv2.rectangle(img, (sx - pad, a), (sx + pad, b), (255, 255, 255), -1)

    # Draw door symbols (leaf + swing arc) on top of the clean gap.
    for d in plan["doors"]:
        _draw_door_symbol(img, d)

    return img


def _draw_door_symbol(img: np.ndarray, door: Dict[str, Any]) -> None:
    """Draw the open door leaf and its quarter-circle swing arc."""
    hx, hy = door["hinge"]["x"], door["hinge"]["y"]
    sx, sy = door["swing_to"]["x"], door["swing_to"]["y"]

    # The closed-position tip is the opening endpoint that is NOT the hinge.
    if (door["start"]["x"], door["start"]["y"]) == (hx, hy):
        cx, cy = door["end"]["x"], door["end"]["y"]
    else:
        cx, cy = door["start"]["x"], door["start"]["y"]

    # Door leaf, drawn in the open position.
    cv2.line(img, (hx, hy), (sx, sy), (0, 0, 0), DOOR_LEAF_THICKNESS, cv2.LINE_AA)

    # Quarter-circle arc swept by the door tip from closed to open.
    radius = int(round(math.hypot(sx - hx, sy - hy)))
    a_closed = math.degrees(math.atan2(cy - hy, cx - hx)) % 360
    a_open = math.degrees(math.atan2(sy - hy, sx - hx)) % 360
    # Pick the 90-degree sweep (not the 270-degree complement).
    if (a_open - a_closed) % 360 > 180:
        a_start, a_end = a_open, a_closed
    else:
        a_start, a_end = a_closed, a_open
    # cv2.ellipse interprets end<start as the long way round, so unwrap.
    if a_end < a_start:
        a_end += 360
    cv2.ellipse(
        img, (hx, hy), (radius, radius), 0, a_start, a_end,
        (0, 0, 0), DOOR_ARC_THICKNESS, cv2.LINE_AA,
    )


def _subtract_gaps(
    a: int, b: int, gaps: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return [a, b] minus each (gap_lo, gap_hi) segment, as a list of pieces."""
    segments: List[Tuple[int, int]] = []
    cursor = a
    for g_lo, g_hi in gaps:
        if g_lo > cursor:
            segments.append((cursor, min(g_lo, b)))
        cursor = max(cursor, g_hi)
    if cursor < b:
        segments.append((cursor, b))
    return segments


# --------------------------------------------------------------------------- #
# Dataset driver
# --------------------------------------------------------------------------- #
def build_dataset(out_dir: Path, n: int = 10, base_seed: int = 1000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        seed = base_seed + i
        plan = generate_floor_plan(seed=seed)
        img = render(plan)

        stem = f"floorplan_{i:03d}"
        img_path = out_dir / f"{stem}.png"
        json_path = out_dir / f"{stem}.json"

        cv2.imwrite(str(img_path), img)
        with open(json_path, "w") as f:
            json.dump(plan, f, indent=2)

        print(
            f"[{i + 1:>2}/{n}] {stem}: "
            f"{len(plan['rooms'])} rooms, "
            f"{len(plan['walls'])} walls, "
            f"{len(plan['doors'])} doors  -> {img_path.name}, {json_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--num-samples", type=int, default=10,
        help="Number of image / JSON pairs to generate (default: 10).",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path,
        default=Path(__file__).parent / "dataset",
        help="Output directory (default: ./dataset).",
    )
    parser.add_argument(
        "--seed", type=int, default=1000,
        help="Base RNG seed; sample i uses seed+i (default: 1000).",
    )
    args = parser.parse_args()
    build_dataset(args.out_dir, n=args.num_samples, base_seed=args.seed)


if __name__ == "__main__":
    main()
