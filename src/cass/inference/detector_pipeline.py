"""Main detection pipeline combining all detectors"""

from typing import Dict, Any, List, Optional
import numpy as np
from ..models import *
from ..vlm_llm import IncidentAnalyzer
from ..utils.logger import setup_logger


class DetectorPipeline:
    """Main detection pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.logger = setup_logger("DetectorPipeline")
        self.config = config
        
        # Initialize detectors
        self._init_detectors()
        
        # Initialize VLM/LLM analyzer if enabled
        vlm_llm_config = config.get('vlm_llm', {})
        if vlm_llm_config.get('enabled', False):
            self.analyzer = IncidentAnalyzer(vlm_llm_config)
        else:
            self.analyzer = None
    
    def _init_detectors(self) -> None:
        """Initialize all detection models"""
        try:
            # YOLO detector (base)
            yolo_config = self.config.get('models', {}).get('yolo', {})
            self.yolo_detector = YOLODetector(yolo_config)
            
            # Fall detector
            fall_config = self.config.get('models', {}).get('fall_detection', {})
            self.fall_detector = FallDetector(fall_config)
            
            # Crowd detector
            crowd_config = self.config.get('models', {}).get('crowd_detection', {})
            self.crowd_detector = CrowdDetector(crowd_config, self.yolo_detector)
            
            # Face detector
            face_config = self.config.get('models', {}).get('face_recognition', {})
            self.face_detector = FaceDetector(face_config)
            
            # Emotion detector
            emotion_config = self.config.get('models', {}).get('emotion_detection', {})
            self.emotion_detector = EmotionDetector(emotion_config, self.face_detector)
            
            # Phone detector
            self.phone_detector = PhoneDetector({}, self.yolo_detector)
            
            # Restricted area detector
            self.area_detector = RestrictedAreaDetector({}, self.yolo_detector)
            
            self.logger.info("All detectors initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detectors: {e}")
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a single frame with all enabled detectors
        
        Args:
            frame: Input frame
            detections_to_run: Optional list of detection types to run
                              (fall, crowd, face, emotion, phone, area)
                              If None, runs all detectors
            
        Returns:
            Dictionary with all detection results
        """
        results = {
            'timestamp': None,  # Would be set by video processor
            'detections': {}
        }
        
        # Default to all detections
        if detections_to_run is None:
            detections_to_run = ['fall', 'crowd', 'face', 'emotion', 'phone', 'area']
        
        try:
            # Fall detection
            if 'fall' in detections_to_run and self.fall_detector.is_available():
                falls = self.fall_detector.detect_falls(frame)
                results['detections']['falls'] = falls
                
                if falls:
                    results['alerts'] = results.get('alerts', [])
                    results['alerts'].append({
                        'type': 'fall_detected',
                        'severity': 'critical',
                        'count': len(falls)
                    })
            
            # Crowd detection
            if 'crowd' in detections_to_run and self.crowd_detector.is_available():
                crowd_info = self.crowd_detector.detect_crowd(frame)
                results['detections']['crowd'] = crowd_info
                
                if crowd_info.get('is_crowded'):
                    results['alerts'] = results.get('alerts', [])
                    results['alerts'].append({
                        'type': 'crowd_detected',
                        'severity': 'high',
                        'count': crowd_info['count'],
                        'level': crowd_info['level']
                    })
            
            # Face detection
            if 'face' in detections_to_run and self.face_detector.is_available():
                faces = self.face_detector.detect_faces(frame)
                results['detections']['faces'] = faces
            
            # Emotion detection
            if 'emotion' in detections_to_run and self.emotion_detector.is_available():
                emotions = self.emotion_detector.detect_emotions(frame)
                results['detections']['emotions'] = emotions
                
                # Check for distress
                distress = [e for e in emotions if e['emotion'] in ['fear', 'sad', 'angry']]
                if distress:
                    results['alerts'] = results.get('alerts', [])
                    results['alerts'].append({
                        'type': 'emotional_distress',
                        'severity': 'medium',
                        'count': len(distress)
                    })
            
            # Phone detection
            if 'phone' in detections_to_run and self.phone_detector.is_available():
                phone_usage = self.phone_detector.detect_phone_usage(frame)
                results['detections']['phones'] = phone_usage
                
                if phone_usage.get('usage_detected'):
                    results['alerts'] = results.get('alerts', [])
                    results['alerts'].append({
                        'type': 'phone_detected',
                        'severity': 'medium',
                        'count': phone_usage['total_phones']
                    })
            
            # Restricted area detection
            if 'area' in detections_to_run and self.area_detector.is_available():
                trespassers = self.area_detector.detect_trespass(frame)
                results['detections']['trespass'] = trespassers
                
                if trespassers:
                    results['alerts'] = results.get('alerts', [])
                    results['alerts'].append({
                        'type': 'restricted_area_breach',
                        'severity': 'critical',
                        'count': len(trespassers)
                    })
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_incident(
        self,
        frame: np.ndarray,
        detection_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze incident using VLM/LLM if available
        
        Args:
            frame: Frame image
            detection_results: Detection results
            
        Returns:
            Analysis results or None
        """
        if not self.analyzer or not self.analyzer.is_available():
            return None
        
        # Extract incident data from detection results
        alerts = detection_results.get('alerts', [])
        if not alerts:
            return None
        
        # Analyze most severe alert
        alert = alerts[0]
        
        incident_data = {
            'event_type': alert['type'],
            'severity': alert['severity'],
            'timestamp': detection_results.get('timestamp', 'unknown'),
            'location': 'unknown'  # Would come from camera metadata
        }
        
        analysis = self.analyzer.analyze_incident(incident_data, frame)
        
        return analysis
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detection_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw all detections on frame
        
        Args:
            frame: Input frame
            detection_results: Detection results
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        detections = detection_results.get('detections', {})
        
        try:
            # Draw falls
            if 'falls' in detections and detections['falls']:
                vis_frame = self.fall_detector.draw_poses(
                    vis_frame,
                    detections['falls'],
                    highlight_falls=True
                )
            
            # Draw faces
            if 'faces' in detections and detections['faces']:
                vis_frame = self.face_detector.draw_faces(
                    vis_frame,
                    detections['faces']
                )
            
            # Draw emotions
            if 'emotions' in detections and detections['emotions']:
                vis_frame = self.emotion_detector.draw_emotions(
                    vis_frame,
                    detections['emotions']
                )
            
            # Draw trespass
            if 'trespass' in detections and detections['trespass']:
                vis_frame = self.area_detector.draw_trespass(
                    vis_frame,
                    detections['trespass']
                )
            
            # Draw alerts summary
            alerts = detection_results.get('alerts', [])
            if alerts:
                y_pos = 30
                for alert in alerts:
                    text = f"{alert['type']}: {alert.get('count', 1)}"
                    import cv2
                    cv2.putText(
                        vis_frame,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    y_pos += 30
        
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
        
        return vis_frame
