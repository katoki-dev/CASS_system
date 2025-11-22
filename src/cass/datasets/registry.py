"""Dataset registry for managing available datasets"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml


class DatasetRegistry:
    """Registry for managing dataset configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize dataset registry
        
        Args:
            config_path: Path to config file containing dataset definitions
        """
        self.datasets: Dict[str, List[Dict[str, Any]]] = {}
        if config_path:
            self.load_from_config(config_path)
    
    def load_from_config(self, config_path: str) -> None:
        """Load datasets from configuration file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'datasets' in config:
            self.datasets = config['datasets']
    
    def register_dataset(
        self,
        category: str,
        name: str,
        path: str,
        format: str,
        enabled: bool = True,
        **metadata
    ) -> None:
        """
        Register a new dataset
        
        Args:
            category: Dataset category (e.g., 'fall_detection', 'crowd_detection')
            name: Dataset name
            path: Path to dataset
            format: Dataset format (e.g., 'coco', 'yolo', 'video')
            enabled: Whether dataset is enabled
            **metadata: Additional metadata
        """
        if category not in self.datasets:
            self.datasets[category] = []
        
        dataset_info = {
            'name': name,
            'path': path,
            'format': format,
            'enabled': enabled,
            **metadata
        }
        
        # Check if dataset already exists
        existing = next(
            (d for d in self.datasets[category] if d['name'] == name),
            None
        )
        
        if existing:
            # Update existing dataset
            idx = self.datasets[category].index(existing)
            self.datasets[category][idx] = dataset_info
        else:
            # Add new dataset
            self.datasets[category].append(dataset_info)
    
    def get_datasets(self, category: str, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get datasets for a category
        
        Args:
            category: Dataset category
            enabled_only: Return only enabled datasets
            
        Returns:
            List of dataset configurations
        """
        if category not in self.datasets:
            return []
        
        datasets = self.datasets[category]
        
        if enabled_only:
            datasets = [d for d in datasets if d.get('enabled', False)]
        
        return datasets
    
    def get_dataset(self, category: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Get specific dataset by name
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            Dataset configuration or None if not found
        """
        datasets = self.datasets.get(category, [])
        return next((d for d in datasets if d['name'] == name), None)
    
    def list_categories(self) -> List[str]:
        """Get list of all dataset categories"""
        return list(self.datasets.keys())
    
    def enable_dataset(self, category: str, name: str) -> bool:
        """
        Enable a dataset
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(category, name)
        if dataset:
            dataset['enabled'] = True
            return True
        return False
    
    def disable_dataset(self, category: str, name: str) -> bool:
        """
        Disable a dataset
        
        Args:
            category: Dataset category
            name: Dataset name
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(category, name)
        if dataset:
            dataset['enabled'] = False
            return True
        return False
    
    def save_to_config(self, config_path: str) -> None:
        """
        Save datasets to configuration file
        
        Args:
            config_path: Path to save config file
        """
        # Load existing config to preserve other sections
        config = {}
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # Update datasets section
        config['datasets'] = self.datasets
        
        # Save
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
