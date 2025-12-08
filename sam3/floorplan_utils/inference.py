"""
Floorplan room detection inference module.

This module provides a high-level API for detecting rooms in floorplan images
using a fine-tuned SAM3 model and exporting results as JSON.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from PIL import Image

from .postprocessing import (
    masks_to_room_polygons,
    visualize_room_polygons,
    polygon_to_coco_segmentation,
)


@dataclass
class RoomDetectionConfig:
    """Configuration for room detection."""
    
    # Detection thresholds
    score_threshold: float = 0.5
    min_area: float = 500.0
    
    # Polygon processing
    epsilon_ratio: float = 0.015
    enforce_orthogonal: bool = True
    simplify_rectangles: bool = True
    
    # Text prompt for detection
    text_prompt: str = "room"


@dataclass
class RoomResult:
    """Result for a single detected room."""
    
    id: int
    polygon: List[List[int]]
    bbox: List[int]  # [x_min, y_min, x_max, y_max]
    area_pixels: float
    confidence: float
    num_vertices: int


@dataclass 
class FloorplanResult:
    """Complete detection result for a floorplan image."""
    
    image_path: str
    image_width: int
    image_height: int
    rooms: List[RoomResult]
    total_rooms: int
    detection_config: Dict[str, Any]
    timestamp: str
    model_checkpoint: Optional[str] = None


class FloorplanRoomDetector:
    """
    High-level API for detecting rooms in floorplan images.
    
    Usage:
        detector = FloorplanRoomDetector(checkpoint_path="path/to/checkpoint.pt")
        result = detector.detect("floorplan.png")
        detector.save_json(result, "output.json")
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: str = "assets/bpe_simple_vocab_16e6.txt.gz",
        device: str = "cuda",
        config: Optional[RoomDetectionConfig] = None,
    ):
        """
        Initialize the room detector.
        
        Args:
            checkpoint_path: Path to fine-tuned SAM3 checkpoint
            bpe_path: Path to BPE vocabulary file
            device: Device to run inference on ("cuda" or "cpu")
            config: Detection configuration
        """
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.device = device
        self.config = config or RoomDetectionConfig()
        
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        print(f"Loading SAM3 model...")
        if self.checkpoint_path:
            print(f"  Checkpoint: {self.checkpoint_path}")
        
        self._model = build_sam3_image_model(
            checkpoint_path=self.checkpoint_path,
            bpe_path=self.bpe_path,
            device=self.device,
            enable_segmentation=True,
        )
        self._processor = Sam3Processor(self._model)
        print("Model loaded successfully!")
    
    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        config: Optional[RoomDetectionConfig] = None,
    ) -> FloorplanResult:
        """
        Detect rooms in a floorplan image.
        
        Args:
            image: Path to image, PIL Image, or numpy array
            config: Override detection config for this call
        
        Returns:
            FloorplanResult with detected rooms
        """
        self._load_model()
        
        config = config or self.config
        
        # Load image
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        width, height = pil_image.size
        
        # Run SAM3 inference
        inference_state = self._processor.set_image(pil_image)
        output = self._processor.set_text_prompt(
            state=inference_state,
            prompt=config.text_prompt
        )
        
        masks = output["masks"]
        scores = output["scores"]
        
        # Convert masks to room polygons
        room_dicts = masks_to_room_polygons(
            masks=masks,
            scores=scores,
            score_threshold=config.score_threshold,
            epsilon_ratio=config.epsilon_ratio,
            min_area=config.min_area,
            enforce_orthogonal=config.enforce_orthogonal,
            simplify_rectangles=config.simplify_rectangles,
        )
        
        # Convert to RoomResult objects
        rooms = [RoomResult(**room) for room in room_dicts]
        
        # Create result
        result = FloorplanResult(
            image_path=image_path or "unknown",
            image_width=width,
            image_height=height,
            rooms=rooms,
            total_rooms=len(rooms),
            detection_config=asdict(config),
            timestamp=datetime.now().isoformat(),
            model_checkpoint=self.checkpoint_path,
        )
        
        return result
    
    def detect_batch(
        self,
        images: List[Union[str, Path]],
        config: Optional[RoomDetectionConfig] = None,
    ) -> List[FloorplanResult]:
        """
        Detect rooms in multiple floorplan images.
        
        Args:
            images: List of image paths
            config: Override detection config for this call
        
        Returns:
            List of FloorplanResult for each image
        """
        results = []
        for i, image_path in enumerate(images):
            print(f"Processing {i+1}/{len(images)}: {image_path}")
            result = self.detect(image_path, config)
            results.append(result)
        
        return results
    
    def visualize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        result: FloorplanResult,
        show_labels: bool = True,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Visualize detection results on the image.
        
        Args:
            image: Input image
            result: Detection result
            show_labels: Whether to show room IDs
            alpha: Overlay transparency
        
        Returns:
            Visualization as numpy array (RGB)
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        image_array = np.array(pil_image)
        
        # Convert RoomResult to dict format
        rooms = [asdict(room) for room in result.rooms]
        
        # Create visualization
        vis = visualize_room_polygons(
            image_array,
            rooms,
            show_labels=show_labels,
            alpha=alpha,
        )
        
        return vis
    
    @staticmethod
    def save_json(
        result: Union[FloorplanResult, List[FloorplanResult]],
        output_path: Union[str, Path],
        indent: int = 2,
    ) -> None:
        """
        Save detection results to JSON file.
        
        Args:
            result: Single result or list of results
            output_path: Path to output JSON file
            indent: JSON indentation
        """
        if isinstance(result, list):
            data = {
                "results": [_result_to_dict(r) for r in result],
                "total_images": len(result),
                "total_rooms": sum(r.total_rooms for r in result),
            }
        else:
            data = _result_to_dict(result)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        print(f"Results saved to: {output_path}")
    
    @staticmethod
    def load_json(json_path: Union[str, Path]) -> Union[FloorplanResult, List[FloorplanResult]]:
        """
        Load detection results from JSON file.
        
        Args:
            json_path: Path to JSON file
        
        Returns:
            FloorplanResult or list of results
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if "results" in data:
            return [_dict_to_result(r) for r in data["results"]]
        else:
            return _dict_to_result(data)
    
    @staticmethod
    def to_coco_format(
        results: List[FloorplanResult],
        category_name: str = "room",
    ) -> Dict[str, Any]:
        """
        Convert results to COCO annotation format.
        
        Args:
            results: List of detection results
            category_name: Name for the room category
        
        Returns:
            COCO-format dictionary
        """
        coco = {
            "info": {
                "description": "Floorplan Room Detection Results",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": category_name, "supercategory": "structure"}
            ],
        }
        
        annotation_id = 1
        
        for image_id, result in enumerate(results, start=1):
            # Add image entry
            coco["images"].append({
                "id": image_id,
                "file_name": os.path.basename(result.image_path),
                "width": result.image_width,
                "height": result.image_height,
            })
            
            # Add annotations
            for room in result.rooms:
                polygon = room.polygon
                bbox = room.bbox
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    "segmentation": polygon_to_coco_segmentation(polygon),
                    "area": room.area_pixels,
                    "iscrowd": 0,
                    "score": room.confidence,
                })
                annotation_id += 1
        
        return coco


def _result_to_dict(result: FloorplanResult) -> Dict[str, Any]:
    """Convert FloorplanResult to dictionary."""
    return {
        "image": result.image_path,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "rooms": [asdict(room) for room in result.rooms],
        "metadata": {
            "total_rooms": result.total_rooms,
            "timestamp": result.timestamp,
            "model_checkpoint": result.model_checkpoint,
            "detection_config": result.detection_config,
        }
    }


def _dict_to_result(data: Dict[str, Any]) -> FloorplanResult:
    """Convert dictionary to FloorplanResult."""
    rooms = [RoomResult(**room) for room in data["rooms"]]
    metadata = data.get("metadata", {})
    
    return FloorplanResult(
        image_path=data["image"],
        image_width=data["image_width"],
        image_height=data["image_height"],
        rooms=rooms,
        total_rooms=metadata.get("total_rooms", len(rooms)),
        detection_config=metadata.get("detection_config", {}),
        timestamp=metadata.get("timestamp", ""),
        model_checkpoint=metadata.get("model_checkpoint"),
    )

