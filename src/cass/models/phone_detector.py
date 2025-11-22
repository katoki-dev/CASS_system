"""Mobile phone detection"""

from typing import List, Dict, Any
import numpy as np
from .yolo_detector import YOLODetector
from ..utils.logger import setup_logger


class PhoneDetector:
    """Mobile phone detection"""
    
    def __init__(self, config: Dict[str, Any], yolo_detector: YOLODetector = None):
        """
        Initialize phone detector
        
        Args:
            config: Configuration dictionary
            yolo_detector: Optional YOLO detector instance
        """
        self.logger = setup_logger("PhoneDetector")
        self.config = config
        self.yolo = yolo_detector
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.yolo is not None and self.yolo.is_available()
    
    def detect_phones(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect mobile phones in image
        
        Args:
            image: Input image
            
        Returns:
            List of phone detections
        """
        if not self.is_available():
            return []
        
        return self.yolo.detect_phones(image)
    
    def detect_phone_usage(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect phone usage with context
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with phone usage information
        """
        phones = self.detect_phones(image)
        persons = self.yolo.detect_persons(image)
        
        # Check if phones are near persons (indicating usage)
        phone_users = []
        
        for phone in phones:
            phone_bbox = phone['bbox']
            phone_center = (
                (phone_bbox[0] + phone_bbox[2]) / 2,
                (phone_bbox[1] + phone_bbox[3]) / 2
            )
            
            # Find nearest person
            for person in persons:
                person_bbox = person['bbox']
                
                # Check if phone is within person's bounding box
                if (person_bbox[0] <= phone_center[0] <= person_bbox[2] and
                    person_bbox[1] <= phone_center[1] <= person_bbox[3]):
                    
                    phone_users.append({
                        'person_bbox': person_bbox,
                        'phone_bbox': phone_bbox,
                        'confidence': phone['confidence']
                    })
                    break
        
        return {
            'total_phones': len(phones),
            'phone_users': phone_users,
            'usage_detected': len(phone_users) > 0
        }
