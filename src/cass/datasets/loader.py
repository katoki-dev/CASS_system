"""Dataset loader for different formats"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


class DatasetLoader:
    """Load datasets in various formats"""
    
    def __init__(self, dataset_path: str, format: str):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to dataset
            format: Dataset format (coco, yolo, video, images, custom)
        """
        self.dataset_path = Path(dataset_path)
        self.format = format.lower()
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load dataset annotations
        
        Returns:
            List of annotations
        """
        if self.format == 'coco':
            return self._load_coco_annotations()
        elif self.format == 'yolo':
            return self._load_yolo_annotations()
        elif self.format == 'video':
            return self._load_video_list()
        elif self.format == 'images':
            return self._load_image_list()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _load_coco_annotations(self) -> List[Dict[str, Any]]:
        """Load COCO format annotations"""
        annotation_file = self.dataset_path / "annotations.json"
        
        if not annotation_file.exists():
            # Try common COCO paths
            for ann_path in [
                self.dataset_path / "instances_train.json",
                self.dataset_path / "annotations" / "instances_train.json"
            ]:
                if ann_path.exists():
                    annotation_file = ann_path
                    break
        
        if not annotation_file.exists():
            return []
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        return coco_data.get('annotations', [])
    
    def _load_yolo_annotations(self) -> List[Dict[str, Any]]:
        """Load YOLO format annotations"""
        annotations = []
        
        # YOLO format: each image has a corresponding .txt file
        label_dir = self.dataset_path / "labels"
        if not label_dir.exists():
            label_dir = self.dataset_path
        
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            image_name = label_file.stem + ".jpg"  # Assume jpg
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        'image': image_name,
                        'class_id': int(parts[0]),
                        'bbox': [float(x) for x in parts[1:5]]
                    })
        
        return annotations
    
    def _load_video_list(self) -> List[Dict[str, Any]]:
        """Load list of video files"""
        video_files = []
        
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(self.dataset_path.glob(ext))
        
        return [{'path': str(v), 'type': 'video'} for v in video_files]
    
    def _load_image_list(self) -> List[Dict[str, Any]]:
        """Load list of image files"""
        image_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(self.dataset_path.glob(ext))
        
        return [{'path': str(img), 'type': 'image'} for img in image_files]
    
    def get_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image
        
        Args:
            image_path: Path to image
            
        Returns:
            Image as numpy array or None
        """
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = self.dataset_path / image_path
        
        if not img_path.exists():
            return None
        
        return cv2.imread(str(img_path))
    
    def get_video_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        vid_path = Path(video_path)
        if not vid_path.is_absolute():
            vid_path = self.dataset_path / video_path
        
        if not vid_path.exists():
            return []
        
        cap = cv2.VideoCapture(str(vid_path))
        frames = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        return frames
    
    def get_sample(self, index: int = 0) -> Optional[Tuple[np.ndarray, Any]]:
        """
        Get a sample from dataset
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (image/frame, annotation) or None
        """
        annotations = self.load_annotations()
        
        if index >= len(annotations):
            return None
        
        annotation = annotations[index]
        
        if self.format in ['coco', 'yolo', 'images']:
            image = self.get_image(annotation.get('image', annotation.get('path', '')))
            return (image, annotation) if image is not None else None
        elif self.format == 'video':
            frames = self.get_video_frames(annotation['path'], max_frames=1)
            return (frames[0], annotation) if frames else None
        
        return None
