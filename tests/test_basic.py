"""Basic tests for CASS system"""

import pytest
from pathlib import Path


def test_imports():
    """Test that all modules can be imported"""
    from cass import load_config, setup_logger
    from cass.datasets import DatasetManager, DatasetRegistry, DatasetLoader
    from cass.models import YOLODetector, FallDetector, CrowdDetector
    from cass.vlm_llm import LLMHandler, VLMHandler, IncidentAnalyzer
    from cass.training import ModelTrainer
    from cass.testing import ModelTester
    from cass.inference import DetectorPipeline, VideoProcessor
    from cass.api import app
    
    assert True  # If we get here, imports worked


def test_config_loading():
    """Test configuration loading"""
    from cass.utils.config import load_config
    
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    config = load_config(str(config_path))
    
    assert config is not None
    assert config.get('models') is not None
    assert config.get('datasets') is not None


def test_logger():
    """Test logger setup"""
    from cass.utils.logger import setup_logger
    
    logger = setup_logger("test")
    assert logger is not None
    
    logger.info("Test log message")


def test_dataset_registry():
    """Test dataset registry"""
    from cass.datasets import DatasetRegistry
    
    registry = DatasetRegistry()
    
    # Add a test dataset
    registry.register_dataset(
        category='test',
        name='test_dataset',
        path='/tmp/test',
        format='coco',
        enabled=True
    )
    
    # Check it was added
    datasets = registry.get_datasets('test', enabled_only=False)
    assert len(datasets) == 1
    assert datasets[0]['name'] == 'test_dataset'


def test_yolo_detector_init():
    """Test YOLO detector initialization"""
    from cass.models import YOLODetector
    
    config = {
        'weights': 'yolov8n.pt',
        'device': 'cpu',
        'conf_threshold': 0.25
    }
    
    detector = YOLODetector(config)
    # Should initialize even if weights not present
    assert detector is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
