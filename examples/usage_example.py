#!/usr/bin/env python3
"""
Example usage of CASS system
"""

from cass.datasets import DatasetManager
from cass.inference import VideoProcessor
from cass.utils.config import load_config
from cass.utils.logger import setup_logger
import cv2

# Set up logger
logger = setup_logger("Example")

def example_dataset_management():
    """Example: Managing datasets"""
    logger.info("=== Dataset Management Example ===")
    
    # Initialize dataset manager
    dm = DatasetManager('configs/config.yaml')
    
    # List all datasets
    datasets = dm.list_datasets()
    logger.info(f"Available datasets: {datasets}")
    
    # Add a new dataset (example - adjust paths as needed)
    # dm.add_dataset(
    #     category='fall_detection',
    #     name='custom_dataset',
    #     path='/path/to/dataset',
    #     format='video',
    #     enabled=True
    # )


def example_image_detection():
    """Example: Detect in a single image"""
    logger.info("=== Image Detection Example ===")
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Initialize video processor
    processor = VideoProcessor(config.get_all())
    
    # Process image (replace with your image path)
    image_path = 'test.jpg'
    
    if cv2.imread(image_path) is not None:
        results = processor.process_image(
            image_path=image_path,
            output_path='output.jpg',
            detections_to_run=['fall', 'crowd', 'face']
        )
        
        if results.get('success'):
            logger.info("Detection completed!")
            
            # Check for alerts
            alerts = results.get('alerts', [])
            if alerts:
                logger.info(f"Found {len(alerts)} alerts:")
                for alert in alerts:
                    logger.info(f"  - {alert['type']}: {alert['severity']}")
            else:
                logger.info("No alerts detected")
        else:
            logger.error(f"Detection failed: {results.get('error')}")
    else:
        logger.warning(f"Image not found: {image_path}")


if __name__ == "__main__":
    logger.info("CASS System Examples")
    logger.info("=" * 50)
    
    # Run examples
    try:
        example_dataset_management()
        print()
        
        example_image_detection()
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
    
    logger.info("Examples completed!")
