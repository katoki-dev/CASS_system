"""Training CLI interface"""

import argparse
import yaml
from pathlib import Path
from .trainer import ModelTrainer
from ..utils.config import load_config
from ..utils.logger import setup_logger


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="CASS Model Training")
    
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['yolo', 'fall', 'emotion', 'face'],
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--dataset-category',
        type=str,
        required=True,
        help='Dataset category (e.g., fall_detection, crowd_detection)'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Dataset name'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='runs/train',
        help='Output directory for training results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help='Training device (cuda, cpu, mps)'
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger("TrainingCLI")
    
    # Load configuration
    try:
        config = load_config(args.config)
        training_config = config.get('training', {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(training_config)
    
    # Prepare training kwargs
    kwargs = {}
    if args.epochs:
        kwargs['epochs'] = args.epochs
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.device:
        kwargs['device'] = args.device
    
    # Train based on model type
    logger.info(f"Starting {args.model_type} training")
    
    if args.model_type == 'yolo':
        results = trainer.train_yolo(
            dataset_category=args.dataset_category,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            **kwargs
        )
    elif args.model_type == 'fall':
        results = trainer.train_fall_detection(
            dataset_category=args.dataset_category,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            **kwargs
        )
    elif args.model_type == 'emotion':
        results = trainer.train_emotion_detection(
            dataset_category=args.dataset_category,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            **kwargs
        )
    else:
        logger.error(f"Unsupported model type: {args.model_type}")
        return
    
    # Print results
    if results.get('success'):
        logger.info("Training completed successfully!")
        if results.get('model_path'):
            logger.info(f"Model saved to: {results['model_path']}")
    else:
        logger.error(f"Training failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
