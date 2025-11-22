"""Testing CLI interface"""

import argparse
from pathlib import Path
from .model_tester import ModelTester
from ..utils.config import load_config
from ..utils.logger import setup_logger


def main():
    """Main testing entry point"""
    parser = argparse.ArgumentParser(description="CASS Model Testing")
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights or comma-separated list for benchmarking'
    )
    
    parser.add_argument(
        '--dataset-category',
        type=str,
        required=True,
        help='Dataset category'
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
        '--benchmark',
        action='store_true',
        help='Run benchmark mode (compare multiple models)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of samples to test'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        default='test_report.txt',
        help='Output path for test report'
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger("TestingCLI")
    
    # Load configuration
    try:
        config = load_config(args.config)
        testing_config = config.get('testing', {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    # Initialize tester
    tester = ModelTester(testing_config)
    
    # Parse models
    models = [m.strip() for m in args.model.split(',')]
    
    # Run tests
    if args.benchmark and len(models) > 1:
        logger.info(f"Benchmarking {len(models)} models")
        
        results = tester.benchmark_models(
            models=models,
            dataset_category=args.dataset_category,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples
        )
        
        if results.get('success'):
            logger.info("Benchmark completed!")
            
            # Print comparison
            comparison = results.get('comparison', {})
            if comparison:
                logger.info(f"Fastest model: {comparison.get('fastest_model')}")
                
                for i, model in enumerate(comparison.get('models', [])):
                    fps = comparison.get('fps', [])[i]
                    logger.info(f"  {Path(model).name}: {fps:.2f} FPS")
    
    else:
        # Test single model
        logger.info(f"Testing model: {models[0]}")
        
        results = tester.test_yolo_model(
            model_path=models[0],
            dataset_category=args.dataset_category,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples
        )
        
        if results.get('success'):
            logger.info("Testing completed!")
            logger.info(f"Inference Time: {results['avg_inference_time']:.4f}s")
            logger.info(f"FPS: {results['fps']:.2f}")
    
    # Generate report
    if tester.results:
        if tester.generate_report(args.report):
            logger.info(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
