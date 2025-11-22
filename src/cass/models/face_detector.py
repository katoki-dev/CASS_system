"""Face detection and recognition"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from ..utils.logger import setup_logger


class FaceDetector:
    """Face detection and recognition"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face detector
        
        Args:
            config: Face detection configuration
        """
        self.logger = setup_logger("FaceDetector")
        self.config = config
        self.face_cascade = None
        self.recognition_model = None
        self.recognition_threshold = config.get('recognition_threshold', 0.6)
        
        self._load_detector()
    
    def _load_detector(self) -> None:
        """Load face detection model"""
        try:
            # Use OpenCV's Haar Cascade for basic face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            self.logger.info("Face detector loaded successfully")
            
            # Optionally load face recognition model
            # This would require facenet-pytorch or similar
            # For now, just detection is implemented
            
        except Exception as e:
            self.logger.error(f"Failed to load face detector: {e}")
            self.face_cascade = None
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.face_cascade is not None
    
    def detect_faces(
        self,
        image: np.ndarray,
        min_size: tuple = (30, 30)
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in image
        
        Args:
            image: Input image
            min_size: Minimum face size
            
        Returns:
            List of face detections
        """
        if not self.is_available():
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_size
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 1.0,  # Haar cascade doesn't provide confidence
                    'face_id': None  # Would be populated by recognition
                })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face(
        self,
        image: np.ndarray,
        bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Extract face region from image
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Face image or None
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            face = image[y1:y2, x1:x2]
            return face
        except Exception as e:
            self.logger.error(f"Face extraction failed: {e}")
            return None
    
    def blur_faces(
        self,
        image: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Blur detected faces in image (privacy protection)
        
        Args:
            image: Input image
            detections: Optional pre-computed detections
            
        Returns:
            Image with blurred faces
        """
        if detections is None:
            detections = self.detect_faces(image)
        
        img_blur = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract face region
            face = img_blur[y1:y2, x1:x2]
            
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            
            # Replace face region with blurred version
            img_blur[y1:y2, x1:x2] = blurred_face
        
        return img_blur
    
    def draw_faces(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw face detections on image
        
        Args:
            image: Input image
            detections: List of face detections
            
        Returns:
            Image with drawn faces
        """
        img_draw = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = "Face"
            if det.get('face_id'):
                label = f"ID: {det['face_id']}"
            
            cv2.putText(
                img_draw,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
        
        return img_draw
