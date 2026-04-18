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


def _choose_axis(rng: random.Random, w: int, h: int) -> str | None:
    """Decide whether to split vertically ('v') or horizontally ('h')."""
    can_v = w >= 2 * MIN_ROOM_DIM
    can_h = h >= 2 * MIN_ROOM_DIM
    if can_v and can_h:
        # bias toward splitting the longer side, with a little randomness
        preferred = "v" if w >= h else "h"
        if rng.random() < 0.25:
            return "h" if preferred == "v" else "v"
        return preferred
    if can_v:
        return "v"
    if can_h:
        return "h"
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
        # Pick the largest splittable rect.
        rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        chosen_idx = None
        for i, r in enumerate(rects):
            if _choose_axis(rng, r[2] - r[0], r[3] - r[1]) is not None:
                chosen_idx = i
                break
        if chosen_idx is None:
            break  # no rect is large enough to subdivide further

        x0, y0, x1, y1 = rects.pop(chosen_idx)
        axis = _choose_axis(rng, x1 - x0, y1 - y0)

        if axis == "v":
            split_x = rng.randint(x0 + MIN_ROOM_DIM, x1 - MIN_ROOM_DIM)
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
            split_y = rng.randint(y0 + MIN_ROOM_DIM, y1 - MIN_ROOM_DIM)
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


def _exterior_door(rng: random.Random, walls: List[_Wall]) -> _Door:
    """Cut a single front-door opening into one of the exterior walls (swings inward)."""
    wall = rng.choice(walls)
    sx, sy = wall.start
    ex, ey = wall.end
    if sy == ey:                                        # horizontal exterior wall
        a, b = sorted([sx, ex])
        d_lo = rng.randint(a + DOOR_END_BUFFER + 20, b - DOOR_END_BUFFER - 20 - DOOR_WIDTH)
        # top wall (y == MARGIN) swings down (+y); bottom wall swings up (-y)
        inward = "+y" if sy == MARGIN else "-y"
        return _build_door(
            rng, "d0", wall.id, "exterior",
            (d_lo, sy), (d_lo + DOOR_WIDTH, sy),
            swing_options=[inward],
        )
    else:                                               # vertical exterior wall
        a, b = sorted([sy, ey])
        d_lo = rng.randint(a + DOOR_END_BUFFER + 20, b - DOOR_END_BUFFER - 20 - DOOR_WIDTH)
        # left wall (x == MARGIN) swings right (+x); right wall swings left (-x)
        inward = "+x" if sx == MARGIN else "-x"
        return _build_door(
            rng, "d0", wall.id, "exterior",
            (sx, d_lo), (sx, d_lo + DOOR_WIDTH),
            swing_options=[inward],
        )


def generate_floor_plan(seed: int | None = None) -> Dict[str, Any]:
    """Build a floor plan and return its ground-truth dictionary."""
    rng = random.Random(seed)

    num_rooms = rng.randint(2, 4)
    room_rects, interior_walls, interior_doors = _bsp_split(rng, num_rooms)

    walls = _exterior_walls() + interior_walls
    doors = [_exterior_door(rng, _exterior_walls())] + interior_doors

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
