"""YOLO-based object detector"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
from ..utils.logger import setup_logger


class YOLODetector:
    """YOLO-based object detection"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO detector
        
        Args:
            config: YOLO configuration dictionary
        """
        self.logger = setup_logger("YOLODetector")
        self.config = config
        self.model = None
        self.device = config.get('device', 'cpu')
        self.conf_threshold = config.get('conf_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            weights = self.config.get('weights', 'yolov8n.pt')
            
            # If weights is just a filename, it will download from Ultralytics
            # If it's a path, it will load from disk
            self.logger.info(f"Loading YOLO model: {weights}")
            self.model = YOLO(weights)
            
            # Set device
            if self.device and hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            self.logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model is not None
    
    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array
            classes: Optional list of class IDs to detect
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_available():
            return []
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=classes,
                verbose=False
            )
            
            # Parse results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names[cls]
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def detect_persons(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in image
        
        Args:
            image: Input image
            
        Returns:
            List of person detections
        """
        # Person class ID is 0 in COCO
        return self.detect(image, classes=[0])
    
    def detect_phones(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect mobile phones in image
        
        Args:
            image: Input image
            
        Returns:
            List of phone detections
        """
        # Cell phone class ID is 67 in COCO
        return self.detect(image, classes=[67])
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_conf: bool = True
    ) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: List of detections
            show_conf: Whether to show confidence scores
            
        Returns:
            Image with drawn detections
        """
        img_draw = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = det['class_name']
            if show_conf:
                label += f" {det['confidence']:.2f}"
            
            # Background for text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Text
            cv2.putText(
                img_draw,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
        
        return img_draw
    
    def count_objects(
        self,
        image: np.ndarray,
        class_filter: Optional[List[int]] = None
    ) -> Dict[str, int]:
        """
        Count objects by class
        
        Args:
            image: Input image
            class_filter: Optional list of classes to count
            
        Returns:
            Dictionary mapping class names to counts
        """
        detections = self.detect(image, classes=class_filter)
        
        counts = {}
        for det in detections:
            class_name = det['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        
        return counts
