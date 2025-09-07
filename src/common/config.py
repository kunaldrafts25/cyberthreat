"""Configuration management for the cyber threat AI system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class Config:
    """Centralized configuration management with validation and environment overrides."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from file path or default location.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config: DictConfig = self._load_config()
        self._apply_env_overrides()
        self._validate_config()
        
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return OmegaConf.create(config_dict)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Device override
        if 'CUDA_VISIBLE_DEVICES' in os.environ and self._config.system.device == "auto":
            self._config.system.device = "cuda" if os.environ['CUDA_VISIBLE_DEVICES'] else "cpu"
            
        # Batch size override for memory constraints
        if 'BATCH_SIZE' in os.environ:
            self._config.data.batch_size = int(os.environ['BATCH_SIZE'])
            
        # Dataset path override
        if 'DATASET_PATH' in os.environ:
            self._config.data.dataset_path = os.environ['DATASET_PATH']
    
    def _validate_config(self) -> None:
        """Validate configuration values and create directories."""
        # Validate splits sum to 1.0
        total_split = (self._config.data.train_split + 
                      self._config.data.val_split + 
                      self._config.data.test_split)
        if not 0.99 <= total_split <= 1.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
            
        # Create required directories
        for dir_key in ['checkpoints_dir', 'logs_dir', 'artifacts_dir']:
            dir_path = Path(self._config.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Validate dataset exists
        dataset_path = Path(self._config.data.dataset_path)
        if not dataset_path.exists():
            logger.warning(f"Dataset not found at {dataset_path}. Training will fail.")
            
        logger.info(f"Configuration loaded and validated from {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        OmegaConf.set(self._config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the raw configuration object."""
        return self._config
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file.
        
        Args:
            path: Save path. If None, overwrites original config file.
        """
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.safe_dump(OmegaConf.to_container(self._config, resolve=True), f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to config file for initial load
        
    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Force reload of global configuration.
    
    Args:
        config_path: Path to new config file
        
    Returns:
        Reloaded Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config
