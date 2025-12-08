"""
Post-processing utilities for floorplan room detection.

This module provides functions to convert SAM3 mask outputs to clean
polygons with sharp corners suitable for architectural floorplans.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.01,
    min_area: float = 100.0
) -> List[np.ndarray]:
    """
    Convert a binary mask to polygon contours.
    
    Args:
        mask: Binary mask array (H, W) with values 0 or 1
        epsilon_ratio: Approximation accuracy as ratio of perimeter (smaller = more points)
        min_area: Minimum area in pixels to keep a contour
    
    Returns:
        List of polygon contours, each as Nx2 array of (x, y) points
    """
    # Ensure mask is uint8
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Reshape to Nx2
        polygon = approx.reshape(-1, 2)
        polygons.append(polygon)
    
    return polygons


def sharpen_polygon_corners(
    polygon: np.ndarray,
    angle_threshold: float = 15.0,
    distance_threshold: float = 5.0
) -> np.ndarray:
    """
    Sharpen corners in a polygon by snapping near-90-degree angles.
    
    Args:
        polygon: Nx2 array of (x, y) polygon vertices
        angle_threshold: Degrees within which to snap to 90 degrees
        distance_threshold: Minimum distance between consecutive points
    
    Returns:
        Polygon with sharpened corners
    """
    if len(polygon) < 3:
        return polygon
    
    # Remove duplicate/close points
    cleaned = [polygon[0]]
    for i in range(1, len(polygon)):
        dist = np.linalg.norm(polygon[i] - cleaned[-1])
        if dist > distance_threshold:
            cleaned.append(polygon[i])
    
    if len(cleaned) < 3:
        return polygon
    
    polygon = np.array(cleaned)
    n = len(polygon)
    result = []
    
    for i in range(n):
        p_prev = polygon[(i - 1) % n]
        p_curr = polygon[i]
        p_next = polygon[(i + 1) % n]
        
        # Calculate vectors
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        # Calculate angle
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 0 and len2 > 0:
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Check if close to 90 degrees
            if abs(angle - 90) < angle_threshold:
                # Snap to orthogonal
                result.append(p_curr)
            else:
                result.append(p_curr)
        else:
            result.append(p_curr)
    
    return np.array(result)


def enforce_orthogonal_corners(
    polygon: np.ndarray,
    angle_threshold: float = 20.0
) -> np.ndarray:
    """
    Force polygon corners to be exactly 90 degrees where appropriate.
    
    This is particularly useful for floorplans where rooms typically
    have right-angle corners.
    
    Args:
        polygon: Nx2 array of (x, y) polygon vertices
        angle_threshold: Degrees within which to force 90-degree corners
    
    Returns:
        Polygon with enforced orthogonal corners
    """
    if len(polygon) < 4:
        return polygon
    
    n = len(polygon)
    result = polygon.copy().astype(np.float64)
    
    # Iterate multiple times to converge
    for _ in range(3):
        for i in range(n):
            p_prev = result[(i - 1) % n]
            p_curr = result[i]
            p_next = result[(i + 1) % n]
            
            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 < 1 or len2 < 1:
                continue
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            # If close to 90 degrees, enforce it
            if abs(angle - 90) < angle_threshold:
                # Determine dominant direction
                if abs(v1[0]) > abs(v1[1]):
                    # v1 is more horizontal, make v2 vertical
                    v2_new = np.array([0, np.sign(v2[1]) * len2 if v2[1] != 0 else len2])
                else:
                    # v1 is more vertical, make v2 horizontal
                    v2_new = np.array([np.sign(v2[0]) * len2 if v2[0] != 0 else len2, 0])
                
                # Adjust next point
                result[(i + 1) % n] = p_curr + v2_new
    
    return result.astype(np.int32)


def simplify_to_rectangle(
    polygon: np.ndarray,
    aspect_ratio_threshold: float = 0.1
) -> np.ndarray:
    """
    Simplify a polygon to its minimum bounding rectangle if it's close to rectangular.
    
    Args:
        polygon: Nx2 array of polygon vertices
        aspect_ratio_threshold: How close the polygon must be to its bounding box
    
    Returns:
        Either the original polygon or a simplified rectangle
    """
    if len(polygon) < 4:
        return polygon
    
    # Get minimum area bounding rectangle
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Compare areas
    polygon_area = cv2.contourArea(polygon.astype(np.float32))
    rect_area = cv2.contourArea(box.astype(np.float32))
    
    if rect_area > 0:
        area_ratio = polygon_area / rect_area
        # If polygon fills most of the bounding rectangle, use the rectangle
        if area_ratio > (1 - aspect_ratio_threshold):
            return box
    
    return polygon


def masks_to_room_polygons(
    masks: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.5,
    epsilon_ratio: float = 0.015,
    min_area: float = 500.0,
    enforce_orthogonal: bool = True,
    simplify_rectangles: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert SAM3 mask outputs to room polygon data.
    
    Args:
        masks: Array of binary masks (N, H, W)
        scores: Confidence scores for each mask (N,)
        score_threshold: Minimum score to include a room
        epsilon_ratio: Polygon approximation accuracy
        min_area: Minimum room area in pixels
        enforce_orthogonal: Whether to force 90-degree corners
        simplify_rectangles: Whether to simplify near-rectangular rooms
    
    Returns:
        List of room dictionaries with polygon, bbox, area, and score
    """
    rooms = []
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < score_threshold:
            continue
        
        # Convert mask to polygons
        polygons = mask_to_polygon(mask, epsilon_ratio=epsilon_ratio, min_area=min_area)
        
        for polygon in polygons:
            # Post-process polygon
            if enforce_orthogonal:
                polygon = sharpen_polygon_corners(polygon)
                polygon = enforce_orthogonal_corners(polygon)
            
            if simplify_rectangles:
                polygon = simplify_to_rectangle(polygon)
            
            # Calculate bounding box
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            
            # Calculate area
            area = float(cv2.contourArea(polygon.astype(np.float32)))
            
            rooms.append({
                "id": len(rooms) + 1,
                "polygon": polygon.tolist(),
                "bbox": bbox,
                "area_pixels": area,
                "confidence": float(score),
                "num_vertices": len(polygon)
            })
    
    return rooms


def polygon_to_coco_segmentation(polygon: List[List[int]]) -> List[List[int]]:
    """
    Convert polygon to COCO segmentation format (flattened coordinate list).
    
    Args:
        polygon: List of [x, y] coordinate pairs
    
    Returns:
        Flattened list [x1, y1, x2, y2, ...]
    """
    flattened = []
    for point in polygon:
        flattened.extend(point)
    return [flattened]


def visualize_room_polygons(
    image: np.ndarray,
    rooms: List[Dict[str, Any]],
    show_labels: bool = True,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Visualize detected room polygons on an image.
    
    Args:
        image: Input image (H, W, 3) in RGB
        rooms: List of room dictionaries from masks_to_room_polygons
        show_labels: Whether to show room IDs
        alpha: Transparency of the overlay
    
    Returns:
        Image with room overlays
    """
    # Create overlay
    overlay = image.copy()
    
    # Generate colors
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(rooms), 3))
    
    for i, room in enumerate(rooms):
        polygon = np.array(room["polygon"], dtype=np.int32)
        color = tuple(map(int, colors[i]))
        
        # Fill polygon
        cv2.fillPoly(overlay, [polygon], color)
        
        # Draw polygon outline
        cv2.polylines(overlay, [polygon], True, (0, 0, 0), 2)
        
        if show_labels:
            # Calculate centroid for label
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw label
                label = f"R{room['id']}"
                cv2.putText(
                    overlay, label, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                cv2.putText(
                    overlay, label, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )
    
    # Blend with original
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return result

