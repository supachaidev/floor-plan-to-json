"""
roomplan_mapper.py
------------------
Map axis-aligned 2D wall / door coordinates (pixel space) into a 3D Apple
RoomPlan CapturedRoom JSON payload.

Output shape (mirrors the encoded form of RoomCaptureSession.CapturedRoom):

    {
      "identifier": UUID,
      "version":    1,
      "walls":    [Surface, ...],
      "doors":    [Surface, ...],
      "windows":  [],
      "openings": [],
      "objects":  [],
      "story":    0
    }

Each Surface:

    {
      "identifier":       UUID,
      "category":         "wall" | "door" | "window" | "opening",
      "confidence":       "high" | "medium" | "low",
      "dimensions":       [width_m, height_m, depth_m],
      "transform":        [16 floats, column-major simd_float4x4],
      "curve":            null,
      "completedEdges":   ["top", "bottom", "left", "right"],
      "parentIdentifier": UUID (doors/windows/openings only)
    }

Coordinate conventions:
    - 2D input: +x right, +y down, integer pixels.
    - 3D RoomPlan: +x right, +y up, +z forward (right-handed, y-up).
    - The 2D image center maps to the world origin; pixel +y maps to world +z.
    - A wall's local axes are: x = length (width), y = up (height), z = thickness
      (depth). Horizontal 2D walls use identity rotation; vertical 2D walls are
      rotated +90 deg around world-y so their length runs along world-z.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_CEILING_M = 2.5
DEFAULT_DOOR_HEIGHT_M = 2.0
DEFAULT_WALL_THICKNESS_PX = 6
DEFAULT_PIXELS_PER_METER = 100.0    # 1024 px image -> 10.24 m across

Point2 = Tuple[int, int]
Wall2D = Tuple[Point2, Point2]
Matrix4 = List[List[float]]          # row-major 4x4


# --------------------------------------------------------------------------- #
# Matrix helpers
# --------------------------------------------------------------------------- #
def _identity4() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rot_y(theta: float) -> Matrix4:
    """Homogeneous rotation around the world-y axis."""
    c, s = math.cos(theta), math.sin(theta)
    return [
        [ c,   0.0,  s,   0.0],
        [ 0.0, 1.0,  0.0, 0.0],
        [-s,   0.0,  c,   0.0],
        [ 0.0, 0.0,  0.0, 1.0],
    ]


def _with_translation(m: Matrix4, tx: float, ty: float, tz: float) -> Matrix4:
    out = [row[:] for row in m]
    out[0][3] = tx
    out[1][3] = ty
    out[2][3] = tz
    return out


def _column_major_flat(m: Matrix4) -> List[float]:
    """simd_float4x4 stores columns contiguously — flatten accordingly."""
    return [round(m[r][c], 6) for c in range(4) for r in range(4)]


# --------------------------------------------------------------------------- #
# Surface builder
# --------------------------------------------------------------------------- #
def _new_uuid() -> str:
    return str(uuid.uuid4()).upper()


def _surface(
    category: str,
    dimensions_m: Tuple[float, float, float],
    transform: Matrix4,
    parent_identifier: Optional[str] = None,
) -> Dict[str, Any]:
    surface: Dict[str, Any] = {
        "identifier":     _new_uuid(),
        "category":       category,
        "confidence":     "high",
        "dimensions":     [round(d, 4) for d in dimensions_m],
        "transform":      _column_major_flat(transform),
        "curve":          None,
        "completedEdges": ["top", "bottom", "left", "right"],
    }
    if parent_identifier is not None:
        surface["parentIdentifier"] = parent_identifier
    return surface


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _orient(wall: Wall2D) -> str:
    (_, y1), (_, y2) = wall
    return "h" if y1 == y2 else "v"


def _segment_center_and_length_px(
    start: Point2, end: Point2, orientation: str
) -> Tuple[float, float, float]:
    """Return (center_x_px, center_y_px, length_px) for an axis-aligned segment."""
    x1, y1 = start
    x2, y2 = end
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    length_px = float(abs(x2 - x1)) if orientation == "h" else float(abs(y2 - y1))
    return cx, cy, length_px


def _pixel_to_world(
    cx_px: float,
    cy_px: float,
    image_size_px: Tuple[int, int],
    scale_m_per_px: float,
) -> Tuple[float, float]:
    """Recenter pixel coords around the image center, convert to meters.
    Returns (world_x, world_z). Pixel +y maps to world +z (depth)."""
    img_w, img_h = image_size_px
    world_x = (cx_px - img_w / 2.0) * scale_m_per_px
    world_z = (cy_px - img_h / 2.0) * scale_m_per_px
    return world_x, world_z


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_captured_room(
    walls_2d: List[Wall2D],
    doors_2d: List[Dict[str, Any]],
    *,
    image_size_px: Tuple[int, int],
    ceiling_m: float = DEFAULT_CEILING_M,
    door_height_m: float = DEFAULT_DOOR_HEIGHT_M,
    wall_thickness_px: int = DEFAULT_WALL_THICKNESS_PX,
    pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
) -> Dict[str, Any]:
    """Build a CapturedRoom-compatible dict from axis-aligned 2D coordinates.

    doors_2d items must be of the form:
        {"wall_index": int, "start": (x, y), "end": (x, y)}
    where wall_index points into walls_2d.
    """
    scale = 1.0 / pixels_per_meter
    thickness_m = wall_thickness_px * scale

    wall_entries: List[Dict[str, Any]] = []
    for wall in walls_2d:
        orient = _orient(wall)
        cx_px, cy_px, length_px = _segment_center_and_length_px(
            wall[0], wall[1], orient
        )
        world_x, world_z = _pixel_to_world(cx_px, cy_px, image_size_px, scale)
        world_y = ceiling_m / 2.0

        rot = _identity4() if orient == "h" else _rot_y(math.pi / 2.0)
        transform = _with_translation(rot, world_x, world_y, world_z)
        dimensions = (length_px * scale, ceiling_m, thickness_m)
        wall_entries.append(_surface("wall", dimensions, transform))

    door_entries: List[Dict[str, Any]] = []
    for door in doors_2d:
        idx = door["wall_index"]
        if not 0 <= idx < len(walls_2d):
            continue
        parent = wall_entries[idx]
        parent_wall = walls_2d[idx]
        orient = _orient(parent_wall)

        cx_px, cy_px, length_px = _segment_center_and_length_px(
            door["start"], door["end"], orient
        )
        world_x, world_z = _pixel_to_world(cx_px, cy_px, image_size_px, scale)
        world_y = door_height_m / 2.0

        rot = _identity4() if orient == "h" else _rot_y(math.pi / 2.0)
        transform = _with_translation(rot, world_x, world_y, world_z)
        dimensions = (length_px * scale, door_height_m, thickness_m)
        door_entries.append(
            _surface("door", dimensions, transform, parent["identifier"])
        )

    return {
        "identifier": _new_uuid(),
        "version":    1,
        "walls":      wall_entries,
        "doors":      door_entries,
        "windows":    [],
        "openings":   [],
        "objects":    [],
        "story":      0,
    }
