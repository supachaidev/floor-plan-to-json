# Floor Plan to JSON

Convert 2D raster floor plans into 3D **Apple RoomPlan** `CapturedRoom` JSON.

The pipeline runs end-to-end on synthetic data: a procedural generator
emits paired image/JSON samples, a Hough-based vectorizer recovers both
wall endpoints and door openings from the raster alone, a 3D mapper
lifts those primitives into RoomPlan-compatible surfaces, and a CLI
ties it all together with evaluation and visualization tools.

---

## Pipeline

```
  generator.py        vectorizer.py         roomplan_mapper.py      visualize.py
      |                    |                        |                    |
      v                    v                        v                    v
  PNG + GT JSON  --->  walls + doors  --->  3D CapturedRoom JSON  --->  matplotlib 3D
   (dataset/)          (px, raster-only)    (meters, 4x4 xforms)         preview
                            ^
                            |
                        main.py evaluate    (MSE vs. ground truth)
```

The vectorizer reads only the image — no ground-truth JSON is consulted
during `convert`. Ground truth is used exclusively by `evaluate` as the
scoring reference.

---

## Setup

```bash
pip install -r requirements.txt   # opencv-python, shapely, numpy
pip install matplotlib            # only needed for `view`
```

Tested on Python 3.12.

---

## CLI

All commands are exposed via `main.py`.

### `evaluate` — measure vectorizer accuracy

```bash
python3 main.py evaluate dataset
```

Pairs each predicted wall with the closest ground-truth wall of the same
orientation and prints per-sample / aggregate MSE on endpoint positions
(px²) and wall lengths (px²), plus a plain-English verdict. On the
reference dataset:

```
Overall position MSE: 11.258 px^2   (RMSE = 3.355 px over 120 endpoints)
Overall length   MSE: 32.883 px^2   (RMSE = 5.734 px over 60 walls)
Overall rating: excellent (sub-stroke).
```

### `convert` — lift a floor plan into RoomPlan JSON

```bash
python3 main.py convert dataset/floorplan_000.png -o room_000.json
```

Vectorizes walls *and* doors from the raster alone — no ground-truth
JSON is consulted. Doors are recovered by scanning each merged wall for
gaps of door-width size, then probing perpendicular to each gap endpoint
for the leaf stroke to pinpoint the hinge and swing direction.

Output strictly follows the `CapturedRoom` shape: `identifier`,
`version`, `walls[]`, `doors[]`, `windows[]`, `openings[]`, `objects[]`,
`story`. Each surface carries `dimensions` (width/height/depth in
meters), `transform` (16-float column-major `simd_float4x4`),
`confidence`, `curve`, and `completedEdges`. Each door has a
`parentIdentifier` linking to its wall.

Flags: `--ceiling 2.5`, `--pixels-per-meter 100`, `-o/--output`.

### `view` — 3D preview of a CapturedRoom JSON

```bash
python3 main.py view room_000.json           # interactive
python3 main.py view room_000.json -s out.png
```

Draws each wall / door as an oriented box via matplotlib 3D.

### `generator.py` — regenerate the reference dataset

```bash
python3 generator.py -n 10 -o dataset --seed 1000
```

BSP-based procedural generator that produces rectangular-room floor
plans with axis-aligned walls, interior / exterior doors, and clean
door-leaf / swing-arc annotations. Each sample is emitted as a PNG
and a ground-truth JSON.

### `vectorizer.py` — standalone wall + door extractor

```bash
python3 vectorizer.py dataset/floorplan_000.png
```

Prints detected wall endpoints and door openings (with hinge and
swing_to). Useful for debugging the Hough + gap-scan pipeline without
invoking the rest of the engine.

---

## Testing

Five increasingly-thorough ways to confirm the engine is working:

**1. Single-sample sanity check** — run the standalone vectorizer:

```bash
python3 vectorizer.py dataset/floorplan_000.png
```

Expect one `[H]` / `[V]` line per wall and one `[D]` line per door.

**2. Dataset-wide door counts** — confirm every sample's detected door
count matches ground truth:

```bash
python3 -c "
import json
from pathlib import Path
from vectorizer import vectorize_plan
for i in range(10):
    png = Path(f'dataset/floorplan_{i:03d}.png')
    gt = json.loads(Path(f'dataset/floorplan_{i:03d}.json').read_text())
    _, d, _ = vectorize_plan(png)
    ok = 'OK' if len(d) == len(gt['doors']) else 'MISMATCH'
    print(f'{png.name}  gt_doors={len(gt[\"doors\"])}  pred_doors={len(d)}  {ok}')
"
```

On the reference dataset: 10/10 OK.

**3. Wall geometry accuracy** — the `evaluate` subcommand reports MSE
of wall endpoint positions and lengths:

```bash
python3 main.py evaluate dataset
```

Reference dataset: position RMSE 3.4 px, length RMSE 5.7 px (sub-stroke).

**4. Visual confirmation** — render each sample and flip through:

```bash
for i in $(seq -w 0 9); do
  python3 main.py convert dataset/floorplan_00${i}.png -o /tmp/r${i}.json
  python3 main.py view /tmp/r${i}.json -s /tmp/r${i}.png
done
```

Expect door-sized red boxes sitting in wall gaps.

**5. Raster-only proof** — verify `convert` does not depend on any
paired ground-truth JSON:

```bash
cp dataset/floorplan_000.png /tmp/blind.png       # just the image
python3 main.py convert /tmp/blind.png -o /tmp/blind_room.json
python3 -c "import json; r=json.load(open('/tmp/blind_room.json')); \
  print(f'walls={len(r[\"walls\"])}  doors={len(r[\"doors\"])}')"
```

Expect `walls=6  doors=3` — the doors came entirely from the raster.

---

## Modules

| File                  | Role                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `generator.py`        | Procedural floor-plan generator (BSP + rasterizer)                |
| `vectorizer.py`       | `HoughLinesP` → axis filter → collinear clustering → merged walls + per-wall door gap scan |
| `roomplan_mapper.py`  | 2D walls + doors → RoomPlan `CapturedRoom` JSON                   |
| `visualize.py`        | 3D box render of a `CapturedRoom`                                 |
| `main.py`             | CLI (`evaluate`, `convert`, `view`)                               |

---

## Dataset layout

```
dataset/
  floorplan_000.png    floorplan_000.json
  floorplan_001.png    floorplan_001.json
  ...
```

Each JSON holds `image_size`, `wall_thickness`, and arrays of `rooms`,
`walls`, and `doors` with integer pixel coordinates. These are the
"ground truth" against which the vectorizer is measured.

---

## Coordinate conventions

- **2D (image):** `+x` right, `+y` down, integer pixels.
- **3D (RoomPlan):** `+x` right, `+y` up, `+z` forward (right-handed, y-up).
- Image center → world origin. Pixel `+y` → world `+z`.
- Walls: local `+x` = length, `+y` = height, `+z` = thickness. Horizontal
  2D walls use identity rotation; vertical walls rotate +90° around
  world-y.
