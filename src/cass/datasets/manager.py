"""Dataset manager - high-level interface for dataset operations"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from .registry import DatasetRegistry
from .loader import DatasetLoader
from ..utils.logger import setup_logger


class DatasetManager:
    """High-level manager for dataset operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize dataset manager
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = setup_logger("DatasetManager")
        self.registry = DatasetRegistry(config_path)
        self.loaders: Dict[str, DatasetLoader] = {}
    
    def add_dataset(
        self,
        category: str,
        name: str,
        path: str,
        format: str,
        enabled: bool = True,
        **metadata
    ) -> bool:
        """
        Add a new dataset to the registry
        
        Args:
            category: Dataset category
            name: Dataset name
            path: Path to dataset
            format: Dataset format
            enabled: Whether to enable the dataset
            **metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Validate path exists
            if not Path(path).exists():
                self.logger.error(f"Dataset path does not exist: {path}")
                return False
            
            # Register dataset
            self.registry.register_dataset(
                category=category,
                name=name,
                path=path,
                format=format,
                enabled=enabled,
                **metadata
            )
            
            self.logger.info(f"Added dataset '{name}' to category '{category}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add dataset: {e}")
            return False
    
    def list_datasets(
        self,
        category: Optional[str] = None,
        enabled_only: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available datasets
        
        Args:
            category: Specific category to list (None for all)
            enabled_only: Only return enabled datasets
            
        Returns:
            Dictionary of datasets by category
        """
        if category:
            return {category: self.registry.get_datasets(category, enabled_only)}
        
        # List all categories
        all_datasets = {}
        for cat in self.registry.list_categories():
            datasets = self.registry.get_datasets(cat, enabled_only)
            if datasets:
                all_datasets[cat] = datasets
        
        return all_datasets
    
    def get_loader(self, category: str, name: str) -> Optional[DatasetLoader]:
        """
        Get dataset loader for a specific dataset
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            DatasetLoader instance or None
        """
        # Check if loader already exists
        loader_key = f"{category}:{name}"
        if loader_key in self.loaders:
            return self.loaders[loader_key]
        
        # Get dataset info
        dataset_info = self.registry.get_dataset(category, name)
        if not dataset_info:
            self.logger.error(f"Dataset not found: {category}/{name}")
            return None
        
        if not dataset_info.get('enabled', False):
            self.logger.warning(f"Dataset is disabled: {category}/{name}")
            return None
        
        # Create loader
        try:
            loader = DatasetLoader(
                dataset_path=dataset_info['path'],
                format=dataset_info['format']
            )
            self.loaders[loader_key] = loader
            return loader
            
        except Exception as e:
            self.logger.error(f"Failed to create loader: {e}")
            return None
    
    def enable_dataset(self, category: str, name: str) -> bool:
        """
        Enable a dataset
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            True if successful
        """
        success = self.registry.enable_dataset(category, name)
        if success:
            self.logger.info(f"Enabled dataset: {category}/{name}")
        return success
    
    def disable_dataset(self, category: str, name: str) -> bool:
        """
        Disable a dataset
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            True if successful
        """
        success = self.registry.disable_dataset(category, name)
        if success:
            self.logger.info(f"Disabled dataset: {category}/{name}")
        return success
    
    def get_statistics(self, category: str, name: str) -> Dict[str, Any]:
        """
        Get statistics for a dataset
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            Dictionary with dataset statistics
        """
        loader = self.get_loader(category, name)
        if not loader:
            return {}
        
        try:
            annotations = loader.load_annotations()
            
            stats = {
                'total_samples': len(annotations),
                'format': loader.format,
                'path': str(loader.dataset_path)
            }
            
            # Format-specific statistics
            if loader.format == 'coco':
                class_counts = {}
                for ann in annotations:
                    cat_id = ann.get('category_id', 'unknown')
                    class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
                stats['class_distribution'] = class_counts
                
            elif loader.format == 'yolo':
                class_counts = {}
                for ann in annotations:
                    class_id = ann.get('class_id', 'unknown')
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                stats['class_distribution'] = class_counts
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def save_config(self, config_path: str) -> bool:
        """
        Save dataset registry to config file
        
        Args:
            config_path: Path to save config
            
        Returns:
            True if successful
        """
        try:
            self.registry.save_to_config(config_path)
            self.logger.info(f"Saved dataset config to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
