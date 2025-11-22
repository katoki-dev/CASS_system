"""Configuration management utilities"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for CASS system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml
        """
        if config_path is None:
            # Default to configs/config.yaml in project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots)
        
        Args:
            key: Configuration key (e.g., 'models.yolo.device')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key
        
        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            path: Path to save config. If None, uses original config_path
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary
        
        Args:
            updates: Dictionary with configuration updates
        """
        self._config.update(updates)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object
    """
    return Config(config_path)
