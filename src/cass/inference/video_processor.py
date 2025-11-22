"""Video processing with detection pipeline"""

from typing import Optional, Callable, Dict, Any
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from .detector_pipeline import DetectorPipeline
from ..utils.logger import setup_logger


class VideoProcessor:
    """Process video streams with detection pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize video processor
        
        Args:
            config: Configuration dictionary
        """
        self.logger = setup_logger("VideoProcessor")
        self.config = config
        
        # Initialize detection pipeline
        self.pipeline = DetectorPipeline(config)
        
        # Circular buffer for auto-clipping
        buffer_size = config.get('inference', {}).get('buffer_size', 600)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Processing parameters
        self.frame_skip = config.get('inference', {}).get('frame_skip', 2)
        self.frame_count = 0
        
        # Callbacks
        self.on_alert_callback = None
    
    def process_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        display: bool = False,
        detections_to_run: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process video from file or stream
        
        Args:
            source: Video source (file path, RTSP URL, or webcam index)
            output_path: Optional path to save output video
            display: Whether to display video while processing
            detections_to_run: Which detections to run
            
        Returns:
            Processing statistics
        """
        self.logger.info(f"Processing video from: {source}")
        
        # Open video source
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return {'success': False, 'error': 'Failed to open video source'}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Video writer for output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing stats
        stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_alerts': 0,
            'alert_types': {}
        }
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to buffer
                self.frame_buffer.append({
                    'frame': frame.copy(),
                    'timestamp': datetime.now()
                })
                
                # Skip frames for performance
                if self.frame_count % (self.frame_skip + 1) != 0:
                    self.frame_count += 1
                    stats['frames_skipped'] += 1
                    continue
                
                # Process frame
                results = self.pipeline.process_frame(frame, detections_to_run)
                results['timestamp'] = datetime.now().isoformat()
                
                # Update stats
                stats['frames_processed'] += 1
                
                if 'alerts' in results:
                    stats['total_alerts'] += len(results['alerts'])
                    
                    for alert in results['alerts']:
                        alert_type = alert['type']
                        stats['alert_types'][alert_type] = \
                            stats['alert_types'].get(alert_type, 0) + 1
                        
                        # Trigger callback if set
                        if self.on_alert_callback:
                            self.on_alert_callback(alert, results, frame)
                
                # Visualize detections
                vis_frame = self.pipeline.visualize_detections(frame, results)
                
                # Write output
                if writer:
                    writer.write(vis_frame)
                
                # Display
                if display:
                    cv2.imshow('CASS Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        self.logger.info(f"Processing complete: {stats['frames_processed']} frames")
        
        stats['success'] = True
        return stats
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        detections_to_run: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image_path: Path to image file
            output_path: Optional path to save output
            detections_to_run: Which detections to run
            
        Returns:
            Detection results
        """
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            return {'success': False, 'error': 'Failed to load image'}
        
        # Process
        results = self.pipeline.process_frame(frame, detections_to_run)
        results['timestamp'] = datetime.now().isoformat()
        
        # Visualize
        vis_frame = self.pipeline.visualize_detections(frame, results)
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, vis_frame)
            results['output_path'] = output_path
        
        results['success'] = True
        return results
    
    def create_clip(
        self,
        pre_event_seconds: int = 10,
        post_event_seconds: int = 20,
        fps: int = 30
    ) -> Optional[np.ndarray]:
        """
        Create video clip from buffer
        
        Args:
            pre_event_seconds: Seconds before event
            post_event_seconds: Seconds after event
            fps: Frames per second
            
        Returns:
            Array of frames or None
        """
        if not self.frame_buffer:
            return None
        
        # Calculate frame counts
        pre_frames = pre_event_seconds * fps
        post_frames = post_event_seconds * fps
        
        # Get frames from buffer
        total_frames = len(self.frame_buffer)
        start_idx = max(0, total_frames - pre_frames)
        
        clip_frames = []
        for i in range(start_idx, total_frames):
            if i < len(self.frame_buffer):
                clip_frames.append(self.frame_buffer[i]['frame'])
        
        return clip_frames if clip_frames else None
    
    def save_clip(
        self,
        frames: list,
        output_path: str,
        fps: int = 30
    ) -> bool:
        """
        Save clip to file
        
        Args:
            frames: List of frames
            output_path: Output file path
            fps: Frames per second
            
        Returns:
            True if successful
        """
        if not frames:
            return False
        
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            
            self.logger.info(f"Clip saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save clip: {e}")
            return False
    
    def set_alert_callback(self, callback: Callable) -> None:
        """
        Set callback function for alerts
        
        Args:
            callback: Function to call on alert
                     Signature: callback(alert: Dict, results: Dict, frame: np.ndarray)
        """
        self.on_alert_callback = callback
