"""Fall detection using pose estimation"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from ..utils.logger import setup_logger


class FallDetector:
    """Fall detection using pose estimation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fall detector
        
        Args:
            config: Fall detection configuration
        """
        self.logger = setup_logger("FallDetector")
        self.config = config
        self.pose_model = None
        self.pose_type = config.get('pose_model', 'mediapipe')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        self._load_pose_model()
    
    def _load_pose_model(self) -> None:
        """Load pose estimation model"""
        try:
            if self.pose_type == 'mediapipe':
                import mediapipe as mp
                
                self.mp_pose = mp.solutions.pose
                self.pose_model = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.logger.info("MediaPipe Pose loaded successfully")
                
            else:
                self.logger.warning(f"Unsupported pose model: {self.pose_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load pose model: {e}")
            self.pose_model = None
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.pose_model is not None
    
    def detect_poses(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect poses in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of pose detections
        """
        if not self.is_available():
            return []
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.pose_model.process(image_rgb)
            
            poses = []
            
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })
                
                poses.append({
                    'landmarks': landmarks,
                    'is_fall': self._check_fall(landmarks)
                })
            
            return poses
            
        except Exception as e:
            self.logger.error(f"Pose detection failed: {e}")
            return []
    
    def _check_fall(self, landmarks: List[Dict[str, float]]) -> bool:
        """
        Check if pose indicates a fall
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            True if fall detected
        """
        if not landmarks or len(landmarks) < 33:
            return False
        
        try:
            # Get key landmarks (MediaPipe indices)
            nose = landmarks[0]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Calculate average hip position
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate average knee position
            knee_y = (left_knee['y'] + right_knee['y']) / 2
            
            # Calculate average ankle position
            ankle_y = (left_ankle['y'] + left_ankle['y']) / 2
            
            # Check if person is horizontal (fall indicator)
            # In normal standing, y values increase from head to feet
            # In fall, body is more horizontal
            
            # Method 1: Check if hip is higher than it should be relative to nose
            # (inverted because y increases downward in image coordinates)
            vertical_ratio = abs(hip_y - nose['y'])
            
            # Method 2: Check body angle
            # If hips and shoulders are at similar height, person might be fallen
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # If shoulder and hip are at similar height (horizontal body)
            body_horizontal = abs(shoulder_y - hip_y) < 0.15
            
            # Method 3: Check if torso is low (close to ground)
            # Hip should be relatively low in the frame
            is_low = hip_y > 0.6  # More than 60% down the image
            
            # Combine heuristics
            is_fall = body_horizontal and (vertical_ratio < 0.3 or is_low)
            
            return is_fall
            
        except Exception as e:
            self.logger.error(f"Fall check failed: {e}")
            return False
    
    def detect_falls(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect falls in image
        
        Args:
            image: Input image
            
        Returns:
            List of fall detections
        """
        poses = self.detect_poses(image)
        falls = [pose for pose in poses if pose.get('is_fall', False)]
        return falls
    
    def draw_poses(
        self,
        image: np.ndarray,
        poses: List[Dict[str, Any]],
        highlight_falls: bool = True
    ) -> np.ndarray:
        """
        Draw poses on image
        
        Args:
            image: Input image
            poses: List of pose detections
            highlight_falls: Whether to highlight fall detections
            
        Returns:
            Image with drawn poses
        """
        img_draw = image.copy()
        h, w = image.shape[:2]
        
        for pose in poses:
            landmarks = pose.get('landmarks', [])
            is_fall = pose.get('is_fall', False)
            
            # Choose color based on fall detection
            color = (0, 0, 255) if (is_fall and highlight_falls) else (0, 255, 0)
            
            # Draw landmarks
            for lm in landmarks:
                if lm['visibility'] > 0.5:
                    x = int(lm['x'] * w)
                    y = int(lm['y'] * h)
                    cv2.circle(img_draw, (x, y), 3, color, -1)
            
            # Draw fall label if detected
            if is_fall and highlight_falls:
                cv2.putText(
                    img_draw,
                    "FALL DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
        
        return img_draw
