"""Restricted area trespass detection"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from .yolo_detector import YOLODetector
from ..utils.logger import setup_logger


class RestrictedAreaDetector:
    """Restricted area trespass detection"""
    
    def __init__(self, config: Dict[str, Any], yolo_detector: YOLODetector = None):
        """
        Initialize restricted area detector
        
        Args:
            config: Configuration dictionary
            yolo_detector: Optional YOLO detector instance
        """
        self.logger = setup_logger("RestrictedAreaDetector")
        self.config = config
        self.yolo = yolo_detector
        self.restricted_zones = []  # List of polygons
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.yolo is not None and self.yolo.is_available()
    
    def add_restricted_zone(
        self,
        zone_id: str,
        polygon: List[Tuple[int, int]],
        name: Optional[str] = None
    ) -> None:
        """
        Add a restricted zone
        
        Args:
            zone_id: Unique zone identifier
            polygon: List of (x, y) points defining the zone
            name: Optional zone name
        """
        zone = {
            'id': zone_id,
            'name': name or zone_id,
            'polygon': np.array(polygon, dtype=np.int32)
        }
        
        self.restricted_zones.append(zone)
        self.logger.info(f"Added restricted zone: {zone_id}")
    
    def remove_restricted_zone(self, zone_id: str) -> bool:
        """
        Remove a restricted zone
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            True if zone was removed
        """
        initial_count = len(self.restricted_zones)
        self.restricted_zones = [
            z for z in self.restricted_zones
            if z['id'] != zone_id
        ]
        
        removed = len(self.restricted_zones) < initial_count
        if removed:
            self.logger.info(f"Removed restricted zone: {zone_id}")
        
        return removed
    
    def is_point_in_zone(
        self,
        point: Tuple[float, float],
        zone_polygon: np.ndarray
    ) -> bool:
        """
        Check if point is inside a zone polygon
        
        Args:
            point: (x, y) coordinate
            zone_polygon: Zone polygon as numpy array
            
        Returns:
            True if point is inside zone
        """
        result = cv2.pointPolygonTest(
            zone_polygon,
            point,
            measureDist=False
        )
        
        return result >= 0
    
    def detect_trespass(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect trespass in restricted zones
        
        Args:
            image: Input image
            
        Returns:
            List of trespass detections
        """
        if not self.is_available():
            return []
        
        if not self.restricted_zones:
            return []
        
        # Detect persons
        persons = self.yolo.detect_persons(image)
        
        trespassers = []
        
        for person in persons:
            bbox = person['bbox']
            
            # Use bottom center of bounding box as person's location
            person_point = (
                (bbox[0] + bbox[2]) / 2,
                bbox[3]  # Bottom of bbox
            )
            
            # Check each restricted zone
            for zone in self.restricted_zones:
                if self.is_point_in_zone(person_point, zone['polygon']):
                    trespassers.append({
                        'person_bbox': bbox,
                        'zone_id': zone['id'],
                        'zone_name': zone['name'],
                        'location': person_point,
                        'confidence': person['confidence']
                    })
        
        return trespassers
    
    def draw_zones(
        self,
        image: np.ndarray,
        fill: bool = False,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw restricted zones on image
        
        Args:
            image: Input image
            fill: Whether to fill zones
            alpha: Transparency for filled zones
            
        Returns:
            Image with drawn zones
        """
        img_draw = image.copy()
        
        for zone in self.restricted_zones:
            polygon = zone['polygon']
            
            if fill:
                # Create overlay for transparency
                overlay = img_draw.copy()
                cv2.fillPoly(overlay, [polygon], (0, 0, 255))
                img_draw = cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0)
            
            # Draw border
            cv2.polylines(img_draw, [polygon], True, (0, 0, 255), 2)
            
            # Draw label
            centroid = polygon.mean(axis=0).astype(int)
            cv2.putText(
                img_draw,
                zone['name'],
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return img_draw
    
    def draw_trespass(
        self,
        image: np.ndarray,
        trespassers: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw trespass detections on image
        
        Args:
            image: Input image
            trespassers: List of trespass detections
            
        Returns:
            Image with drawn trespassers
        """
        # First draw zones
        img_draw = self.draw_zones(image, fill=True)
        
        # Draw trespassers
        for trespasser in trespassers:
            bbox = trespasser['person_bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw box in red
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label
            label = f"TRESPASS: {trespasser['zone_name']}"
            cv2.putText(
                img_draw,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        return img_draw
