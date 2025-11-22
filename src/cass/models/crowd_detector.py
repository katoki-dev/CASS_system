"""Crowd detection and counting"""

from typing import List, Dict, Any
import numpy as np
from .yolo_detector import YOLODetector
from ..utils.logger import setup_logger


class CrowdDetector:
    """Crowd detection and people counting"""
    
    def __init__(self, config: Dict[str, Any], yolo_detector: YOLODetector = None):
        """
        Initialize crowd detector
        
        Args:
            config: Crowd detection configuration
            yolo_detector: Optional YOLO detector instance
        """
        self.logger = setup_logger("CrowdDetector")
        self.config = config
        self.method = config.get('method', 'yolo_counting')
        self.density_threshold = config.get('density_threshold', 50)
        
        # Use provided YOLO detector or create new one
        self.yolo = yolo_detector
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.yolo is not None and self.yolo.is_available()
    
    def count_people(self, image: np.ndarray) -> int:
        """
        Count people in image
        
        Args:
            image: Input image
            
        Returns:
            Number of people detected
        """
        if not self.is_available():
            return 0
        
        detections = self.yolo.detect_persons(image)
        return len(detections)
    
    def detect_crowd(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect crowd and analyze density
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with crowd information
        """
        count = self.count_people(image)
        
        # Calculate density (people per area)
        h, w = image.shape[:2]
        area = (h * w) / (1000 * 1000)  # Area in megapixels
        density = count / area if area > 0 else 0
        
        # Determine crowd level
        if count < 5:
            level = "low"
        elif count < 20:
            level = "medium"
        elif count < 50:
            level = "high"
        else:
            level = "critical"
        
        is_crowded = count >= self.density_threshold
        
        return {
            'count': count,
            'density': density,
            'level': level,
            'is_crowded': is_crowded,
            'threshold': self.density_threshold
        }
    
    def get_person_locations(self, image: np.ndarray) -> List[tuple]:
        """
        Get locations of detected people
        
        Args:
            image: Input image
            
        Returns:
            List of (x, y) center coordinates
        """
        if not self.is_available():
            return []
        
        detections = self.yolo.detect_persons(image)
        
        locations = []
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            locations.append((center_x, center_y))
        
        return locations
