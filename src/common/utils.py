"""Utility functions for the cyber threat AI system."""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device(device_str: str = "auto") -> torch.device:
    """Get PyTorch device with intelligent auto-detection.
    
    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        
    Returns:
        PyTorch device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
        logger.info(f"Using specified device: {device}")
    
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.debug(f"JSON saved to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file as dictionary.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"JSON loaded from {filepath}")
    return data


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB
    """
    memory_info = {}
    
    # CPU memory (if psutil available)
    try:
        import psutil
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        memory_info['cpu_memory_mb'] = 0.0
    
    # GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    else:
        memory_info['gpu_memory_allocated_mb'] = 0.0
        memory_info['gpu_memory_reserved_mb'] = 0.0
    
    return memory_info


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = 0.0
        self.end_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.description} completed in {format_time(duration)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time


def batch_generator(data: List[Any], batch_size: int, shuffle: bool = False) -> List[List[Any]]:
    """Generate batches from a list of data.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before batching
        
    Yields:
        Batches of data
    """
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def moving_average(values: List[float], window_size: int = 10) -> List[float]:
    """Calculate moving average of a list of values.
    
    Args:
        values: List of numeric values
        window_size: Size of the moving window
        
    Returns:
        List of moving averages
    """
    if len(values) < window_size:
        return values
    
    averages = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        end = i + 1
        avg = sum(values[start:end]) / (end - start)
        averages.append(avg)
    
    return averages


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and converting to lowercase.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    return text.strip().lower()


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value for division by zero
        
    Returns:
        Division result or default
    """
    try:
        return a / b if b != 0 else default
    except (ZeroDivisionError, TypeError):
        return default


def clip_gradient_norm(model: nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients by norm and return the total norm.
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm value
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def exponential_decay(initial_value: float, decay_rate: float, step: int) -> float:
    """Calculate exponential decay value.
    
    Args:
        initial_value: Starting value
        decay_rate: Decay rate (0 < decay_rate < 1)
        step: Current step
        
    Returns:
        Decayed value
    """
    return initial_value * (decay_rate ** step)


def cosine_annealing(initial_value: float, min_value: float, step: int, max_steps: int) -> float:
    """Calculate cosine annealing schedule value.
    
    Args:
        initial_value: Starting value
        min_value: Minimum value
        step: Current step
        max_steps: Total number of steps
        
    Returns:
        Annealed value
    """
    if step >= max_steps:
        return min_value
    
    cos_inner = np.pi * step / max_steps
    cos_out = np.cos(cos_inner) + 1
    return min_value + (initial_value - min_value) / 2 * cos_out
