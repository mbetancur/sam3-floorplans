#!/usr/bin/env python3
"""
Floorplan Room Detection CLI

Detect rooms in floorplan images using a fine-tuned SAM3 model
and output results as JSON.

Usage:
    python scripts/detect_rooms.py --image floorplan.png --output rooms.json
    python scripts/detect_rooms.py --image-dir ./floorplans/ --output results.json
    python scripts/detect_rooms.py --image floorplan.png --checkpoint model.pt --visualize

Examples:
    # Single image detection
    python scripts/detect_rooms.py \\
        --image hospital_floor1.png \\
        --checkpoint checkpoints/sam3_floorplan.pt \\
        --output floor1_rooms.json

    # Batch detection with visualization
    python scripts/detect_rooms.py \\
        --image-dir ./floorplans/ \\
        --checkpoint checkpoints/sam3_floorplan.pt \\
        --output all_rooms.json \\
        --visualize \\
        --vis-dir ./visualizations/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect rooms in floorplan images using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to a single floorplan image"
    )
    input_group.add_argument(
        "--image-dir", "-d",
        type=str,
        help="Directory containing floorplan images"
    )
    
    # Model options
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to fine-tuned SAM3 checkpoint (uses base model if not specified)"
    )
    parser.add_argument(
        "--bpe-path",
        type=str,
        default="assets/bpe_simple_vocab_16e6.txt.gz",
        help="Path to BPE vocabulary file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    
    # Detection options
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to include a room (default: 0.5)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=500.0,
        help="Minimum room area in pixels (default: 500)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="room",
        help="Text prompt for detection (default: 'room')"
    )
    parser.add_argument(
        "--no-orthogonal",
        action="store_true",
        help="Disable orthogonal corner enforcement"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable rectangle simplification"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--coco-format",
        action="store_true",
        help="Output in COCO annotation format"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="Directory for visualization outputs (default: same as output)"
    )
    
    return parser.parse_args()


def get_image_files(image_dir: str) -> list:
    """Get all image files from a directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_dir = Path(image_dir)
    
    files = []
    for ext in extensions:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    return sorted(files)


def main():
    args = parse_args()
    
    # Import here to avoid slow startup for --help
    from sam3.floorplan_utils.inference import (
        FloorplanRoomDetector,
        RoomDetectionConfig,
    )
    
    # Create config
    config = RoomDetectionConfig(
        score_threshold=args.score_threshold,
        min_area=args.min_area,
        text_prompt=args.prompt,
        enforce_orthogonal=not args.no_orthogonal,
        simplify_rectangles=not args.no_simplify,
    )
    
    # Initialize detector
    detector = FloorplanRoomDetector(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        device=args.device,
        config=config,
    )
    
    # Get input images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = get_image_files(args.image_dir)
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            sys.exit(1)
        print(f"Found {len(image_paths)} images")
    
    # Run detection
    results = detector.detect_batch([str(p) for p in image_paths], config)
    
    # Print summary
    total_rooms = sum(r.total_rooms for r in results)
    print(f"\n{'='*50}")
    print(f"Detection Complete")
    print(f"{'='*50}")
    print(f"Images processed: {len(results)}")
    print(f"Total rooms detected: {total_rooms}")
    print(f"Average rooms per image: {total_rooms / len(results):.1f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.coco_format:
        coco_data = detector.to_coco_format(results)
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"\nCOCO annotations saved to: {output_path}")
    else:
        if len(results) == 1:
            detector.save_json(results[0], output_path)
        else:
            detector.save_json(results, output_path)
    
    # Generate visualizations
    if args.visualize:
        vis_dir = Path(args.vis_dir) if args.vis_dir else output_path.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations in: {vis_dir}")
        
        for result, image_path in zip(results, image_paths):
            vis = detector.visualize(image_path, result)
            
            # Save visualization
            vis_path = vis_dir / f"{image_path.stem}_rooms.png"
            Image.fromarray(vis).save(vis_path)
            print(f"  Saved: {vis_path.name}")
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()

