"""
visualize.py
------------
Render a RoomPlan CapturedRoom JSON as 3D boxes in matplotlib.

Each wall / door is reconstructed from its `dimensions` (width, height, depth)
and `transform` (column-major simd_float4x4), then drawn as an axis-aligned
box in the surface's local frame, transformed into world space.

Axis mapping:
    world +x -> matplotlib x  (right)
    world +z -> matplotlib y  (forward/back — floor plane)
    world +y -> matplotlib z  (up)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Corner indices for the 6 faces of a box (cube topology).
_FACES = [
    [0, 1, 2, 3],   # -z
    [4, 5, 6, 7],   # +z
    [0, 1, 5, 4],   # -y
    [2, 3, 7, 6],   # +y
    [0, 3, 7, 4],   # -x
    [1, 2, 6, 5],   # +x
]


def _transform_from_flat(flat_16: List[float]) -> np.ndarray:
    """Reshape a 16-float column-major list into a 4x4 row-major numpy matrix."""
    m = np.asarray(flat_16, dtype=float).reshape(4, 4).T  # column-major -> row-major
    return m


def _box_corners_local(dimensions: Tuple[float, float, float]) -> np.ndarray:
    """Return the 8 corners of a centered box in homogeneous coords, shape (4, 8)."""
    w, h, d = dimensions
    hw, hh, hd = w / 2.0, h / 2.0, d / 2.0
    corners = np.array([
        [-hw, -hh, -hd],
        [ hw, -hh, -hd],
        [ hw,  hh, -hd],
        [-hw,  hh, -hd],
        [-hw, -hh,  hd],
        [ hw, -hh,  hd],
        [ hw,  hh,  hd],
        [-hw,  hh,  hd],
    ], dtype=float)
    homogeneous = np.hstack([corners, np.ones((8, 1))])
    return homogeneous.T  # (4, 8)


def _world_corners(surface: Dict[str, Any]) -> np.ndarray:
    m = _transform_from_flat(surface["transform"])
    local = _box_corners_local(tuple(surface["dimensions"]))
    world = (m @ local)[:3, :].T       # (8, 3)
    return world


def _to_mpl(points: np.ndarray) -> np.ndarray:
    """Swap world Y and Z so matplotlib's default Z-up renders world-y as up."""
    return points[:, [0, 2, 1]]


def _draw_box(ax, surface: Dict[str, Any], face_color: str, edge_color: str) -> np.ndarray:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    corners_world = _world_corners(surface)
    corners_mpl = _to_mpl(corners_world)
    polys = [corners_mpl[face] for face in _FACES]
    collection = Poly3DCollection(
        polys,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidths=0.5,
        alpha=0.75,
    )
    ax.add_collection3d(collection)
    return corners_mpl


def _set_equal_aspect(ax, all_points: np.ndarray) -> None:
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    half = (maxs - mins).max() / 2.0 * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def view_room(json_path: Path, save_path: Optional[Path] = None) -> None:
    """Render a CapturedRoom JSON. If save_path is given, write a PNG instead
    of opening an interactive window."""
    import matplotlib.pyplot as plt

    data = json.loads(Path(json_path).read_text())
    walls = data.get("walls", [])
    doors = data.get("doors", [])
    windows = data.get("windows", [])
    openings = data.get("openings", [])

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_corners: List[np.ndarray] = []
    for w in walls:
        all_corners.append(_draw_box(ax, w, "#d8d8d8", "#333333"))
    for d in doors:
        all_corners.append(_draw_box(ax, d, "#e57373", "#b71c1c"))
    for w in windows:
        all_corners.append(_draw_box(ax, w, "#64b5f6", "#0d47a1"))
    for o in openings:
        all_corners.append(_draw_box(ax, o, "#fff59d", "#f57f17"))

    if all_corners:
        _set_equal_aspect(ax, np.vstack(all_corners))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)   [floor plane]")
    ax.set_zlabel("Y (m)   [up]")
    ax.set_title(
        f"{Path(json_path).name}   "
        f"{len(walls)} walls, {len(doors)} doors, "
        f"{len(windows)} windows, {len(openings)} openings"
    )
    ax.view_init(elev=25, azim=-60)

    # Legend proxies (Poly3DCollection doesn't get picked up by default).
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#d8d8d8", edgecolor="#333333", label="wall"),
        Patch(facecolor="#e57373", edgecolor="#b71c1c", label="door"),
    ]
    if windows:
        legend_items.append(Patch(facecolor="#64b5f6", edgecolor="#0d47a1", label="window"))
    if openings:
        legend_items.append(Patch(facecolor="#fff59d", edgecolor="#f57f17", label="opening"))
    ax.legend(handles=legend_items, loc="upper right")

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {save_path}")
