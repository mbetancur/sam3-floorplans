# Hospital Floorplan Room Detection Dataset

This dataset is structured for training SAM3 to detect room shapes in hospital floorplan images.

## Directory Structure

```
floorplans/
├── train/                          # Training images and annotations
│   ├── _annotations.coco.json     # COCO-format annotations
│   └── *.png                       # Floorplan images
├── val/                            # Validation images and annotations
│   ├── _annotations.coco.json
│   └── *.png
├── test/                           # Test images and annotations
│   ├── _annotations.coco.json
│   └── *.png
├── dataset_utils.py               # Utility script for validation/conversion
└── README.md                       # This file
```

## Annotation Format (COCO JSON)

### Complete JSON Structure

```json
{
  "info": { ... },
  "licenses": [ ... ],
  "images": [ ... ],
  "annotations": [ ... ],
  "categories": [ ... ]
}
```

### Images Array

Each image entry requires:

| Field       | Type   | Required    | Description                                  |
|-------------|--------|-------------|----------------------------------------------|
| `id`        | int    | **YES**     | Unique image identifier                      |
| `file_name` | string | **YES**     | Image filename (relative to directory)       |
| `width`     | int    | **YES**     | Image width in pixels                        |
| `height`    | int    | **YES**     | Image height in pixels                       |

### Annotations Array

Each annotation requires:

| Field          | Type           | Required                | Description                                     |
|----------------|----------------|-------------------------|-------------------------------------------------|
| `id`           | int            | **YES**                 | Unique annotation ID (globally unique)          |
| `image_id`     | int            | **YES**                 | References images[].id                          |
| `category_id`  | int            | **YES**                 | References categories[].id (use 1 for "room")   |
| `bbox`         | [x,y,w,h]      | **YES**                 | Bounding box in PIXELS (XYWH format)            |
| `iscrowd`      | 0 or 1         | **YES**                 | 0=normal, 1=ambiguous region                    |
| `segmentation` | [[x1,y1,...]]  | **YES for mask training** | Polygon points in PIXELS                      |
| `area`         | float          | Optional                | Area in square pixels (auto-computed if missing)|

### Categories Array

For room detection, use a single category:

```json
{
  "categories": [
    {"id": 1, "name": "room", "supercategory": "structure"}
  ]
}
```

**Important**: The `name` field becomes the text prompt during training.

## Coordinate System

```
(0,0) ────────────────────────► X (width)
  │
  │    ┌─────────────┐
  │    │             │  bbox: [x, y, w, h]
  │    │    ROOM     │  x = left edge (pixels)
  │    │             │  y = top edge (pixels)
  │    └─────────────┘  w = width (pixels)
  │                     h = height (pixels)
  ▼
  Y (height)
```

### Polygon Format

Polygons are flattened coordinate arrays:
```json
"segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]]
```

For a rectangular room:
```json
"segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]]
//               TL────────TR────────BR────────BL
```

## Utility Scripts

### Validate Annotations

```bash
python dataset_utils.py validate --annotations train/_annotations.coco.json
```

### Print Statistics

```bash
python dataset_utils.py stats --annotations train/_annotations.coco.json
```

### Convert from LabelMe

```bash
python dataset_utils.py create-from-labelme \
    --input /path/to/labelme/output \
    --output train/
```

## Annotation Tips for Sharp Corners

1. **Use polygon mode** in your annotation tool (not auto-segment)
2. **Place vertices exactly on corners** - at least 4 points per rectangular room
3. **Be consistent with wall thickness** - annotate either inside or outside edges
4. **Mark partial rooms** with `iscrowd: 1` if they're cut by image boundaries

## Recommended Dataset Sizes

| Phase              | Images  | Rooms/Image | Total Annotations | Purpose             |
|--------------------|---------|-------------|-------------------|---------------------|
| PoC (Minimum)      | 30-50   | 30-40       | ~1,200-2,000      | Validate feasibility|
| Initial Training   | 100-150 | 30-40       | ~3,500-6,000      | Baseline model      |
| Production         | 300-500 | 30-40       | ~12,000-20,000    | Robust performance  |

## Annotation Tools

Recommended tools for polygon annotation:

1. **CVAT** (free, self-hosted) - Best for polygon annotation with magnetism
2. **Roboflow** - AI-assisted, direct COCO export
3. **LabelMe** - Simple polygon annotation tool

All export COCO format directly or can be converted using `dataset_utils.py`.

