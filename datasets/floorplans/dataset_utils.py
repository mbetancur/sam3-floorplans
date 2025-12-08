#!/usr/bin/env python3
"""
Floorplan Dataset Utilities

This module provides utilities for creating, validating, and managing
COCO-format annotations for hospital floorplan room detection.

Usage:
    python dataset_utils.py validate --annotations train/_annotations.coco.json
    python dataset_utils.py stats --annotations train/_annotations.coco.json
    python dataset_utils.py create-from-labelme --input labelme_output/ --output train/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys


def calculate_polygon_area(polygon: List[float]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Args:
        polygon: Flat list of coordinates [x1, y1, x2, y2, ...]
    
    Returns:
        Area in square pixels
    """
    n = len(polygon) // 2
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = polygon[2*i], polygon[2*i + 1]
        x_j, y_j = polygon[2*j], polygon[2*j + 1]
        area += x_i * y_j
        area -= x_j * y_i
    
    return abs(area) / 2.0


def polygon_to_bbox(polygon: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert polygon to bounding box in XYWH format.
    
    Args:
        polygon: Flat list of coordinates [x1, y1, x2, y2, ...]
    
    Returns:
        Tuple of (x, y, width, height)
    """
    xs = polygon[0::2]
    ys = polygon[1::2]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def create_annotation(
    annotation_id: int,
    image_id: int,
    polygon: List[float],
    category_id: int = 1,
    iscrowd: int = 0
) -> Dict[str, Any]:
    """
    Create a COCO-format annotation dictionary.
    
    Args:
        annotation_id: Unique ID for this annotation
        image_id: ID of the image this annotation belongs to
        polygon: Flat list of polygon coordinates [x1, y1, x2, y2, ...]
        category_id: Category ID (default 1 for "room")
        iscrowd: 0 for normal instances, 1 for crowd/ambiguous
    
    Returns:
        COCO annotation dictionary
    """
    bbox = polygon_to_bbox(polygon)
    area = calculate_polygon_area(polygon)
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "segmentation": [polygon],
        "area": area,
        "iscrowd": iscrowd
    }


def create_image_entry(
    image_id: int,
    file_name: str,
    width: int,
    height: int
) -> Dict[str, Any]:
    """
    Create a COCO-format image entry.
    
    Args:
        image_id: Unique ID for this image
        file_name: Relative path to the image file
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        COCO image dictionary
    """
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": 1
    }


def validate_annotations(annotations_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a COCO annotations file for SAM3 training.
    
    Args:
        annotations_path: Path to the annotations JSON file
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    warnings = []
    
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {annotations_path}"]
    
    # Check required top-level keys
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")
    
    if errors:
        return False, errors
    
    # Validate categories
    if not data["categories"]:
        errors.append("Categories array is empty - need at least one category")
    else:
        category_ids = set()
        for cat in data["categories"]:
            if "id" not in cat:
                errors.append(f"Category missing 'id': {cat}")
            if "name" not in cat:
                errors.append(f"Category missing 'name': {cat}")
            else:
                category_ids.add(cat["id"])
    
    # Validate images
    image_ids = set()
    image_dimensions = {}
    for img in data["images"]:
        required_img_fields = ["id", "file_name", "width", "height"]
        for field in required_img_fields:
            if field not in img:
                errors.append(f"Image missing '{field}': {img}")
        
        if "id" in img:
            if img["id"] in image_ids:
                errors.append(f"Duplicate image ID: {img['id']}")
            image_ids.add(img["id"])
            if "width" in img and "height" in img:
                image_dimensions[img["id"]] = (img["width"], img["height"])
    
    # Validate annotations
    annotation_ids = set()
    for ann in data["annotations"]:
        # Check required fields
        required_ann_fields = ["id", "image_id", "category_id", "bbox", "iscrowd"]
        for field in required_ann_fields:
            if field not in ann:
                errors.append(f"Annotation {ann.get('id', '?')} missing '{field}'")
        
        # Check annotation ID uniqueness
        if "id" in ann:
            if ann["id"] in annotation_ids:
                errors.append(f"Duplicate annotation ID: {ann['id']}")
            annotation_ids.add(ann["id"])
        
        # Check image_id reference
        if "image_id" in ann and ann["image_id"] not in image_ids:
            errors.append(f"Annotation {ann.get('id', '?')} references non-existent image_id: {ann['image_id']}")
        
        # Check category_id reference
        if "category_id" in ann and ann["category_id"] not in category_ids:
            errors.append(f"Annotation {ann.get('id', '?')} references non-existent category_id: {ann['category_id']}")
        
        # Validate bbox format (should be [x, y, w, h])
        if "bbox" in ann:
            bbox = ann["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"Annotation {ann.get('id', '?')} has invalid bbox format (expected [x,y,w,h]): {bbox}")
            elif any(v < 0 for v in bbox[:2]) or any(v <= 0 for v in bbox[2:]):
                warnings.append(f"Annotation {ann.get('id', '?')} has suspicious bbox values: {bbox}")
        
        # Validate segmentation (required for mask training)
        if "segmentation" not in ann or not ann["segmentation"]:
            warnings.append(f"Annotation {ann.get('id', '?')} missing segmentation - REQUIRED for mask training")
        elif isinstance(ann["segmentation"], list):
            for poly in ann["segmentation"]:
                if isinstance(poly, list) and len(poly) < 6:
                    errors.append(f"Annotation {ann.get('id', '?')} has polygon with < 3 points")
        
        # Check iscrowd value
        if "iscrowd" in ann and ann["iscrowd"] not in [0, 1]:
            errors.append(f"Annotation {ann.get('id', '?')} has invalid iscrowd value: {ann['iscrowd']}")
    
    # Print warnings
    for warning in warnings:
        print(f"WARNING: {warning}")
    
    return len(errors) == 0, errors


def print_dataset_stats(annotations_path: str) -> None:
    """
    Print statistics about the dataset.
    
    Args:
        annotations_path: Path to the annotations JSON file
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    num_images = len(data["images"])
    num_annotations = len(data["annotations"])
    num_categories = len(data["categories"])
    
    # Count annotations per image
    anns_per_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        anns_per_image[img_id] = anns_per_image.get(img_id, 0) + 1
    
    # Count annotations with segmentation
    with_segmentation = sum(1 for ann in data["annotations"] if ann.get("segmentation"))
    
    # Calculate area statistics
    areas = [ann.get("area", 0) for ann in data["annotations"]]
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total images:       {num_images}")
    print(f"Total annotations:  {num_annotations}")
    print(f"Categories:         {num_categories}")
    print(f"With segmentation:  {with_segmentation} ({100*with_segmentation/max(1,num_annotations):.1f}%)")
    
    if num_images > 0 and anns_per_image:
        avg_anns = num_annotations / num_images
        min_anns = min(anns_per_image.values())
        max_anns = max(anns_per_image.values())
        print(f"\nAnnotations per image:")
        print(f"  Average: {avg_anns:.1f}")
        print(f"  Min:     {min_anns}")
        print(f"  Max:     {max_anns}")
    
    if areas:
        avg_area = sum(areas) / len(areas)
        print(f"\nArea statistics (pixels²):")
        print(f"  Average: {avg_area:,.0f}")
        print(f"  Min:     {min(areas):,.0f}")
        print(f"  Max:     {max(areas):,.0f}")
    
    print("\nCategories:")
    for cat in data["categories"]:
        cat_count = sum(1 for ann in data["annotations"] if ann["category_id"] == cat["id"])
        print(f"  [{cat['id']}] {cat['name']}: {cat_count} annotations")
    
    print("="*50 + "\n")


def convert_labelme_to_coco(
    labelme_dir: str,
    output_dir: str,
    category_name: str = "room"
) -> str:
    """
    Convert LabelMe JSON annotations to COCO format.
    
    Args:
        labelme_dir: Directory containing LabelMe JSON files
        output_dir: Output directory for images and annotations
        category_name: Name for the category (default "room")
    
    Returns:
        Path to the generated annotations file
    """
    import glob
    from PIL import Image
    import shutil
    
    labelme_files = glob.glob(os.path.join(labelme_dir, "*.json"))
    
    coco_data = {
        "info": {
            "description": "Hospital Floorplan Room Detection Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Converted from LabelMe",
            "date_created": "2025-01-01"
        },
        "licenses": [{"id": 1, "name": "Custom License", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": category_name, "supercategory": "structure"}
        ]
    }
    
    annotation_id = 1
    
    for image_id, labelme_file in enumerate(labelme_files, start=1):
        with open(labelme_file, 'r') as f:
            labelme_data = json.load(f)
        
        # Get image info
        image_name = labelme_data.get("imagePath", os.path.basename(labelme_file).replace(".json", ".png"))
        image_width = labelme_data.get("imageWidth")
        image_height = labelme_data.get("imageHeight")
        
        # Copy image if it exists
        source_image = os.path.join(labelme_dir, image_name)
        if os.path.exists(source_image):
            shutil.copy(source_image, os.path.join(output_dir, image_name))
        
        # Add image entry
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": image_width,
            "height": image_height,
            "license": 1
        })
        
        # Convert shapes to annotations
        for shape in labelme_data.get("shapes", []):
            if shape["shape_type"] == "polygon":
                # Flatten polygon points
                points = shape["points"]
                polygon = []
                for point in points:
                    polygon.extend(point)
                
                # Create annotation
                ann = create_annotation(
                    annotation_id=annotation_id,
                    image_id=image_id,
                    polygon=polygon,
                    category_id=1,
                    iscrowd=0
                )
                coco_data["annotations"].append(ann)
                annotation_id += 1
    
    # Save annotations
    output_path = os.path.join(output_dir, "_annotations.coco.json")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Converted {len(labelme_files)} LabelMe files to COCO format")
    print(f"Output: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Floorplan Dataset Utilities for SAM3 Training"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate COCO annotations")
    validate_parser.add_argument(
        "--annotations", "-a",
        required=True,
        help="Path to COCO annotations JSON file"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Print dataset statistics")
    stats_parser.add_argument(
        "--annotations", "-a",
        required=True,
        help="Path to COCO annotations JSON file"
    )
    
    # Convert from LabelMe command
    convert_parser = subparsers.add_parser(
        "create-from-labelme",
        help="Convert LabelMe annotations to COCO format"
    )
    convert_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing LabelMe JSON files"
    )
    convert_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for COCO dataset"
    )
    convert_parser.add_argument(
        "--category",
        default="room",
        help="Category name (default: room)"
    )
    
    args = parser.parse_args()
    
    if args.command == "validate":
        is_valid, errors = validate_annotations(args.annotations)
        if is_valid:
            print("✓ Annotations are valid!")
            print_dataset_stats(args.annotations)
            sys.exit(0)
        else:
            print("✗ Validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
    
    elif args.command == "stats":
        print_dataset_stats(args.annotations)
    
    elif args.command == "create-from-labelme":
        os.makedirs(args.output, exist_ok=True)
        convert_labelme_to_coco(args.input, args.output, args.category)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

