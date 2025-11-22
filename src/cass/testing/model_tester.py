"""Model testing and benchmarking"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
from pathlib import Path
from ..utils.logger import setup_logger
from ..datasets.manager import DatasetManager
from ..models import *


class ModelTester:
    """Model testing and benchmarking interface"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model tester
        
        Args:
            config: Testing configuration
        """
        self.logger = setup_logger("ModelTester")
        self.config = config
        self.dataset_manager = DatasetManager()
        self.results = []
    
    def test_yolo_model(
        self,
        model_path: str,
        dataset_category: str,
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test YOLO model on a dataset
        
        Args:
            model_path: Path to model weights
            dataset_category: Dataset category
            dataset_name: Dataset name
            **kwargs: Additional parameters
            
        Returns:
            Test results dictionary
        """
        self.logger.info(f"Testing YOLO model: {model_path}")
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Get dataset
            loader = self.dataset_manager.get_loader(dataset_category, dataset_name)
            if not loader:
                raise ValueError(f"Dataset not found: {dataset_category}/{dataset_name}")
            
            # Get test samples
            annotations = loader.load_annotations()
            
            if not annotations:
                return {
                    'success': False,
                    'error': 'No test data available'
                }
            
            # Test metrics
            total_samples = len(annotations)
            inference_times = []
            detections_count = []
            
            # Test on subset (or all if small dataset)
            test_samples = min(total_samples, kwargs.get('max_samples', 100))
            
            for i in range(test_samples):
                sample = loader.get_sample(i)
                if sample is None:
                    continue
                
                image, annotation = sample
                
                # Measure inference time
                start_time = time.time()
                results = model(image, verbose=False)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
                
                # Count detections
                num_detections = len(results[0].boxes) if results else 0
                detections_count.append(num_detections)
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times) if inference_times else 0
            avg_detections = np.mean(detections_count) if detections_count else 0
            
            results = {
                'success': True,
                'model_path': model_path,
                'dataset': f"{dataset_category}/{dataset_name}",
                'total_samples': total_samples,
                'tested_samples': test_samples,
                'avg_inference_time': avg_inference_time,
                'avg_detections': avg_detections,
                'fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            }
            
            self.results.append(results)
            
            self.logger.info(f"Testing completed: {avg_inference_time:.4f}s per image, {results['fps']:.2f} FPS")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Testing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def benchmark_models(
        self,
        models: List[str],
        dataset_category: str,
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark multiple models
        
        Args:
            models: List of model paths
            dataset_category: Dataset category
            dataset_name: Dataset name
            **kwargs: Additional parameters
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Benchmarking {len(models)} models")
        
        benchmark_results = []
        
        for model_path in models:
            result = self.test_yolo_model(
                model_path=model_path,
                dataset_category=dataset_category,
                dataset_name=dataset_name,
                **kwargs
            )
            
            if result.get('success'):
                benchmark_results.append(result)
        
        # Create comparison
        comparison = self._create_comparison(benchmark_results)
        
        return {
            'success': True,
            'results': benchmark_results,
            'comparison': comparison
        }
    
    def _create_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison of benchmark results"""
        if not results:
            return {}
        
        comparison = {
            'models': [r['model_path'] for r in results],
            'inference_times': [r['avg_inference_time'] for r in results],
            'fps': [r['fps'] for r in results],
            'detections': [r['avg_detections'] for r in results]
        }
        
        # Find best performing model
        fastest_idx = np.argmin(comparison['inference_times'])
        comparison['fastest_model'] = comparison['models'][fastest_idx]
        
        return comparison
    
    def test_detection_accuracy(
        self,
        model_path: str,
        dataset_category: str,
        dataset_name: str,
        ground_truth: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Test detection accuracy against ground truth
        
        Args:
            model_path: Path to model
            dataset_category: Dataset category
            dataset_name: Dataset name
            ground_truth: Optional ground truth annotations
            
        Returns:
            Accuracy metrics
        """
        # This would implement detailed accuracy testing
        # Including precision, recall, mAP, etc.
        # For now, return placeholder
        
        self.logger.info("Accuracy testing not fully implemented")
        
        return {
            'success': True,
            'message': 'Detailed accuracy metrics not yet implemented',
            'metrics': {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mAP': 0.0
            }
        }
    
    def generate_report(
        self,
        output_path: str = "test_report.txt"
    ) -> bool:
        """
        Generate testing report
        
        Args:
            output_path: Path to save report
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w') as f:
                f.write("CASS Model Testing Report\n")
                f.write("=" * 50 + "\n\n")
                
                for i, result in enumerate(self.results, 1):
                    f.write(f"Test {i}:\n")
                    f.write(f"  Model: {result.get('model_path', 'N/A')}\n")
                    f.write(f"  Dataset: {result.get('dataset', 'N/A')}\n")
                    f.write(f"  Samples: {result.get('tested_samples', 0)}\n")
                    f.write(f"  Avg Inference Time: {result.get('avg_inference_time', 0):.4f}s\n")
                    f.write(f"  FPS: {result.get('fps', 0):.2f}\n")
                    f.write(f"  Avg Detections: {result.get('avg_detections', 0):.2f}\n")
                    f.write("\n")
            
            self.logger.info(f"Report saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return False
