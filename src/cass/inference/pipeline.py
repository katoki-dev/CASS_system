"""Inference CLI interface"""

import argparse
from pathlib import Path
from .video_processor import VideoProcessor
from ..utils.config import load_config
from ..utils.logger import setup_logger


def main():
    """Main inference entry point"""
    parser = argparse.ArgumentParser(description="CASS Inference Pipeline")
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Video source (file path, RTSP URL, or webcam index)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output video path'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display video while processing'
    )
    
    parser.add_argument(
        '--detections',
        type=str,
        help='Comma-separated list of detections to run (fall,crowd,face,emotion,phone,area)'
    )
    
    parser.add_argument(
        '--image-mode',
        action='store_true',
        help='Process as single image instead of video'
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger("InferenceCLI")
    
    # Load configuration
    try:
        config_obj = load_config(args.config)
        config = config_obj.get_all()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    # Parse detections
    detections_to_run = None
    if args.detections:
        detections_to_run = [d.strip() for d in args.detections.split(',')]
    
    # Initialize processor
    processor = VideoProcessor(config)
    
    # Process
    if args.image_mode:
        logger.info(f"Processing image: {args.source}")
        
        results = processor.process_image(
            image_path=args.source,
            output_path=args.output,
            detections_to_run=detections_to_run
        )
        
        if results.get('success'):
            logger.info("Image processing complete!")
            
            alerts = results.get('alerts', [])
            if alerts:
                logger.info(f"Alerts detected: {len(alerts)}")
                for alert in alerts:
                    logger.info(f"  - {alert['type']}: severity={alert['severity']}")
            else:
                logger.info("No alerts detected")
            
            if args.output:
                logger.info(f"Output saved to {args.output}")
        else:
            logger.error(f"Processing failed: {results.get('error')}")
    
    else:
        logger.info(f"Processing video: {args.source}")
        
        stats = processor.process_video(
            source=args.source,
            output_path=args.output,
            display=args.display,
            detections_to_run=detections_to_run
        )
        
        if stats.get('success'):
            logger.info("Video processing complete!")
            logger.info(f"Frames processed: {stats['frames_processed']}")
            logger.info(f"Total alerts: {stats['total_alerts']}")
            
            if stats['alert_types']:
                logger.info("Alert breakdown:")
                for alert_type, count in stats['alert_types'].items():
                    logger.info(f"  - {alert_type}: {count}")
        else:
            logger.error(f"Processing failed: {stats.get('error')}")


if __name__ == "__main__":
    main()
