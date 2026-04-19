"""
main.py
-------
CLI entry point for the floor-plan vectorization engine.

Commands:
    evaluate <dataset-dir>
        For every *.png in the directory, run the vectorizer and compare the
        predicted wall coordinates against the matching *.json ground truth.
        Reports Mean Squared Error on wall endpoint positions (px^2) and on
        wall lengths (px^2), both per-sample and aggregated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from roomplan_mapper import (
    DEFAULT_CEILING_M,
    DEFAULT_PIXELS_PER_METER,
    build_captured_room,
)
from vectorizer import Wall, vectorize, vectorize_plan


# --------------------------------------------------------------------------- #
# Wall helpers
# --------------------------------------------------------------------------- #
def _normalized(wall: Wall) -> Wall:
    """Order endpoints so the smaller (x, y) comes first (stable comparison)."""
    (x1, y1), (x2, y2) = wall
    if (x2, y2) < (x1, y1):
        return ((x2, y2), (x1, y1))
    return ((x1, y1), (x2, y2))


def _orientation(wall: Wall) -> str:
    (_, y1), (_, y2) = wall
    return "h" if y1 == y2 else "v"


def _length(wall: Wall) -> float:
    # Walls are axis-aligned, so |dx| + |dy| equals the Euclidean length.
    (x1, y1), (x2, y2) = wall
    return float(abs(x2 - x1) + abs(y2 - y1))


def _load_ground_truth(json_path: Path) -> List[Wall]:
    data = json.loads(json_path.read_text())
    walls: List[Wall] = []
    for w in data["walls"]:
        p1 = (w["start"]["x"], w["start"]["y"])
        p2 = (w["end"]["x"], w["end"]["y"])
        walls.append(_normalized((p1, p2)))
    return walls


# --------------------------------------------------------------------------- #
# Prediction <-> ground-truth matching
# --------------------------------------------------------------------------- #
def _match_walls(
    gt: List[Wall], pred: List[Wall]
) -> Tuple[List[Tuple[Wall, Wall]], List[Wall], List[Wall]]:
    """Greedy match within each orientation by shared-axis proximity.

    Returns (matched_pairs, unmatched_gt, unmatched_pred).
    """
    matched: List[Tuple[Wall, Wall]] = []
    unmatched_gt: List[Wall] = []

    for axis in ("h", "v"):
        gt_set = [w for w in gt if _orientation(w) == axis]
        pred_set = [w for w in pred if _orientation(w) == axis]
        coord_idx = 1 if axis == "h" else 0  # horizontal shares y, vertical shares x

        remaining = list(pred_set)
        for g in gt_set:
            if not remaining:
                unmatched_gt.append(g)
                continue
            g_coord = g[0][coord_idx]
            nearest = min(remaining, key=lambda p: abs(p[0][coord_idx] - g_coord))
            matched.append((g, nearest))
            remaining.remove(nearest)

    matched_pred_ids = {id(p) for _, p in matched}
    unmatched_pred = [p for p in pred if id(p) not in matched_pred_ids]
    return matched, unmatched_gt, unmatched_pred


# --------------------------------------------------------------------------- #
# MSE accumulation
# --------------------------------------------------------------------------- #
class _MseAccumulator:
    """Collects raw squared errors so aggregate MSE is computed correctly
    (mean over all errors, not mean of per-sample means)."""

    def __init__(self) -> None:
        self.pos_sq: List[float] = []
        self.len_sq: List[float] = []

    def add(self, gt: Wall, pred: Wall) -> Tuple[float, float]:
        (gx1, gy1), (gx2, gy2) = gt
        (px1, py1), (px2, py2) = pred
        e1 = (px1 - gx1) ** 2 + (py1 - gy1) ** 2
        e2 = (px2 - gx2) ** 2 + (py2 - gy2) ** 2
        el = (_length(pred) - _length(gt)) ** 2
        self.pos_sq.extend([e1, e2])
        self.len_sq.append(el)
        return (e1 + e2) / 2.0, el

    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def pos_mse(self) -> float:
        return self.mean(self.pos_sq)

    @property
    def len_mse(self) -> float:
        return self.mean(self.len_sq)


# --------------------------------------------------------------------------- #
# Per-sample evaluation
# --------------------------------------------------------------------------- #
def _evaluate_sample(
    image_path: Path, json_path: Path, agg: _MseAccumulator
) -> Dict[str, float]:
    pred = [_normalized(w) for w in vectorize(image_path)]
    gt = _load_ground_truth(json_path)
    matched, unmatched_gt, unmatched_pred = _match_walls(gt, pred)

    sample = _MseAccumulator()
    for g, p in matched:
        sample.add(g, p)
        agg.add(g, p)

    return {
        "gt_count": len(gt),
        "pred_count": len(pred),
        "matched": len(matched),
        "missed": len(unmatched_gt),
        "extra": len(unmatched_pred),
        "pos_mse": sample.pos_mse,
        "len_mse": sample.len_mse,
    }


# --------------------------------------------------------------------------- #
# Human-readable explanation
# --------------------------------------------------------------------------- #
# Walls are drawn with a ~6 px stroke, so any endpoint error below that is
# essentially inside the line itself — the best Hough can resolve without
# sub-pixel refinement.
WALL_THICKNESS_PX = 6


def _print_guide() -> None:
    print(
        "\nWhat this measures"
        "\n  Each predicted wall is paired with a ground-truth wall of the same"
        "\n  orientation, by closest shared-axis coordinate (y for horizontals,"
        "\n  x for verticals). The paired walls are then compared geometrically."
        "\n"
        "\nColumn guide"
        "\n  gt       walls in the ground-truth JSON"
        "\n  pred     walls returned by the vectorizer"
        "\n  match    successfully paired walls"
        "\n  miss     GT walls with no matching prediction  (false negatives)"
        "\n  extra    predictions with no GT partner        (false positives)"
        "\n  pos MSE  mean squared endpoint distance, in px^2  (lower is better)"
        "\n  len MSE  mean squared wall-length difference, in px^2 (lower is better)"
        "\n"
        "\nReading the numbers"
        f"\n  Take sqrt(MSE) to convert back to pixels. Walls are ~{WALL_THICKNESS_PX} px"
        "\n  thick, so endpoint RMSE at or below that is sub-stroke accuracy."
    )


def _rate(pos_rmse: float, thickness: int = WALL_THICKNESS_PX) -> str:
    if pos_rmse <= thickness:
        return "excellent (sub-stroke)"
    if pos_rmse <= 2 * thickness:
        return "good (within 2x stroke)"
    if pos_rmse <= 4 * thickness:
        return "fair (noticeable offset)"
    return "poor (walls materially misplaced)"


def _print_interpretation(
    agg: _MseAccumulator, samples: List[Dict[str, float]]
) -> None:
    if not samples:
        return
    pos_rmse = agg.pos_mse ** 0.5
    len_rmse = agg.len_mse ** 0.5

    print("\nInterpretation")
    print(
        f"  On average, each predicted endpoint lands ~{pos_rmse:.1f} px from ground truth,"
        f"\n  and each predicted wall length is off by ~{len_rmse:.1f} px."
        f"\n  Overall rating: {_rate(pos_rmse)}."
    )

    topo = [s for s in samples if s["missed"] > 0 or s["extra"] > 0]
    # Flag samples whose RMSE is at least 3x the dataset-wide RMSE (and not already tiny).
    outlier_cutoff = max(2 * WALL_THICKNESS_PX, 3 * pos_rmse)
    outliers = [s for s in samples if s["pos_mse"] ** 0.5 > outlier_cutoff]

    if topo:
        names = ", ".join(s["name"] for s in topo)
        print(f"  Topology errors (missing or hallucinated walls): {names}")
    if outliers:
        names = ", ".join(
            f"{s['name']} (RMSE={s['pos_mse'] ** 0.5:.1f} px)" for s in outliers
        )
        print(f"  Geometric outliers (endpoint error >> average): {names}")
    if not topo and not outliers:
        print("  No topology errors or geometric outliers flagged.")


# --------------------------------------------------------------------------- #
# Dataset driver
# --------------------------------------------------------------------------- #
def evaluate(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {dataset_dir}")

    images = sorted(dataset_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No *.png images found in {dataset_dir}")

    print(f"\nEvaluating {len(images)} sample(s) in {dataset_dir}")
    _print_guide()

    header = (
        f"\n{'sample':<20}{'gt':>4}{'pred':>6}{'match':>7}"
        f"{'miss':>6}{'extra':>7}{'pos MSE':>14}{'len MSE':>14}"
    )
    print(header)
    print("-" * (len(header) - 1))  # -1 because of the leading newline

    agg = _MseAccumulator()
    samples: List[Dict[str, float]] = []
    for img_path in images:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  [skip] {img_path.name}: no matching .json ground truth")
            continue
        s = _evaluate_sample(img_path, json_path, agg)
        s["name"] = img_path.stem
        samples.append(s)
        print(
            f"{img_path.stem:<20}{s['gt_count']:>4}{s['pred_count']:>6}"
            f"{s['matched']:>7}{s['missed']:>6}{s['extra']:>7}"
            f"{s['pos_mse']:>14.3f}{s['len_mse']:>14.3f}"
        )

    print("-" * (len(header) - 1))
    if not samples:
        print("No samples evaluated.")
        return

    print(
        f"{'AGGREGATE':<20}{'':>4}{'':>6}{len(agg.pos_sq) // 2:>7}{'':>6}{'':>7}"
        f"{agg.pos_mse:>14.3f}{agg.len_mse:>14.3f}"
    )
    print(
        f"\nOverall position MSE: {agg.pos_mse:.3f} px^2"
        f"   (RMSE = {agg.pos_mse ** 0.5:.3f} px over {len(agg.pos_sq)} endpoints)"
    )
    print(
        f"Overall length   MSE: {agg.len_mse:.3f} px^2"
        f"   (RMSE = {agg.len_mse ** 0.5:.3f} px over {len(agg.len_sq)} walls)"
    )

    _print_interpretation(agg, samples)


# --------------------------------------------------------------------------- #
# Convert: 2D vectors -> Apple RoomPlan CapturedRoom JSON
# --------------------------------------------------------------------------- #
def convert(
    image_path: Path,
    out_path: Optional[Path],
    ceiling_m: float,
    pixels_per_meter: float,
    wall_thickness_px: Optional[int] = None,
    door_width_px: Optional[int] = None,
) -> None:
    pred_walls_raw, pred_doors, image_size, cfg = vectorize_plan(
        image_path,
        wall_thickness_px=wall_thickness_px,
        door_width_px=door_width_px,
    )
    pred_walls = [_normalized(w) for w in pred_walls_raw]

    # Re-map door wall_index from the raw wall list to the normalized list.
    # _normalized reorders endpoints but preserves position, so wall order is
    # unchanged — the indices stay valid.
    doors = [
        {
            "wall_index": d.wall_index,
            "start": d.start,
            "end": d.end,
        }
        for d in pred_doors
    ]

    room = build_captured_room(
        pred_walls,
        doors,
        image_size_px=image_size,
        ceiling_m=ceiling_m,
        pixels_per_meter=pixels_per_meter,
    )
    serialized = json.dumps(room, indent=2)

    if out_path is None:
        print(serialized)
    else:
        out_path.write_text(serialized)
        print(
            f"Wrote {out_path}  "
            f"({len(room['walls'])} walls, {len(room['doors'])} doors, "
            f"ceiling={ceiling_m} m, {pixels_per_meter} px/m, "
            f"wall={cfg.wall_thickness_px}px, door={cfg.door_width_px}px)"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floor-plan-engine",
        description="Floor-plan vectorization engine CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser(
        "evaluate",
        help="Measure vectorizer accuracy against ground-truth JSONs.",
    )
    p_eval.add_argument(
        "dataset",
        type=Path,
        help="Directory containing paired *.png and *.json samples.",
    )

    p_conv = sub.add_parser(
        "convert",
        help="Vectorize an image and emit Apple RoomPlan CapturedRoom JSON.",
    )
    p_conv.add_argument(
        "image",
        type=Path,
        help="Floor-plan PNG to convert (paired .json, if present, supplies doors).",
    )
    p_conv.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (stdout if omitted).",
    )
    p_conv.add_argument(
        "--ceiling",
        type=float,
        default=DEFAULT_CEILING_M,
        help=f"Ceiling height in meters (default: {DEFAULT_CEILING_M}).",
    )
    p_conv.add_argument(
        "--pixels-per-meter",
        type=float,
        default=DEFAULT_PIXELS_PER_METER,
        help=(
            f"Pixel-to-meter scale (default: {DEFAULT_PIXELS_PER_METER} px/m "
            f"-> 1024-px image spans ~{1024 / DEFAULT_PIXELS_PER_METER:.1f} m)."
        ),
    )
    p_conv.add_argument(
        "--wall-thickness",
        type=int,
        default=None,
        help="Override wall stroke thickness (px). Auto-estimated if omitted.",
    )
    p_conv.add_argument(
        "--door-width",
        type=int,
        default=None,
        help="Override door opening width (px). Defaults to 10 x wall thickness.",
    )

    p_view = sub.add_parser(
        "view",
        help="Render a RoomPlan CapturedRoom JSON as 3D boxes (matplotlib).",
    )
    p_view.add_argument(
        "room_json",
        type=Path,
        help="CapturedRoom JSON produced by `convert`.",
    )
    p_view.add_argument(
        "-s", "--save",
        type=Path,
        default=None,
        help="Save to PNG at this path instead of opening an interactive window.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "evaluate":
        evaluate(args.dataset)
    elif args.command == "convert":
        convert(
            image_path=args.image,
            out_path=args.output,
            ceiling_m=args.ceiling,
            pixels_per_meter=args.pixels_per_meter,
            wall_thickness_px=args.wall_thickness,
            door_width_px=args.door_width,
        )
    elif args.command == "view":
        from visualize import view_room
        view_room(args.room_json, save_path=args.save)


if __name__ == "__main__":
    main()
