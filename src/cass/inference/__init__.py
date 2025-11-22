"""Inference pipeline for real-time detection"""

from .detector_pipeline import DetectorPipeline
from .video_processor import VideoProcessor
from .pipeline import main

__all__ = ["DetectorPipeline", "VideoProcessor", "main"]
