"""Centralized logging configuration for the cyber threat AI system."""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from .config import get_config


def setup_logging(
    config_path: Optional[Union[str, Path]] = None,
    log_level: Optional[str] = None
) -> None:
    """Configure loguru logger with settings from configuration.
    
    Args:
        config_path: Path to config file (optional)
        log_level: Override log level (optional)
    """
    config = get_config(config_path)
    
    # Remove default handler
    logger.remove()
    
    # Determine log level
    level = log_level or config.get("logging.level", "INFO")
    
    # Console handler
    logger.add(
        sys.stderr,
        format=config.get("logging.format", 
                         "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"),
        level=level,
        colorize=True
    )
    
    # File handler for all logs
    logs_dir = Path(config.get("paths.logs_dir", "reports/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        logs_dir / "cyber_threat_ai.log",
        format=config.get("logging.format", 
                         "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
        level=level,
        rotation=config.get("logging.rotation", "100 MB"),
        retention=config.get("logging.retention", "30 days"),
        compression="zip"
    )
    
    # Separate file for errors
    logger.add(
        logs_dir / "errors.log",
        format=config.get("logging.format",
                         "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
        level="ERROR",
        rotation=config.get("logging.rotation", "100 MB"),
        retention=config.get("logging.retention", "30 days"),
        compression="zip"
    )
    
    # Training-specific log
    logger.add(
        logs_dir / "training.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        filter=lambda record: "training" in record["extra"],
        level="INFO",
        rotation="10 MB"
    )
    
    logger.info("Logging configured successfully")


def get_logger(name: str) -> "Logger":
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> "Logger":
        """Get logger instance for this class."""
        return logger.bind(name=self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function entry and exit with parameters."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def log_training_metrics(**metrics):
    """Log training metrics to the training-specific log file.
    
    Args:
        **metrics: Metric name-value pairs to log
    """
    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                            for k, v in metrics.items()])
    logger.bind(training=True).info(f"METRICS | {metric_str}")


def log_model_info(model, model_name: str = "Model"):
    """Log model architecture information.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"{model_name} initialized:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
