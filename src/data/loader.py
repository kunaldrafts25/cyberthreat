"""Robust data loader for the zero-day attack detection dataset."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from ..common.config import get_config
from ..common.utils import normalize_text


class ThreatDataset(Dataset):
    """PyTorch Dataset for threat detection data with multi-modal features."""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 features: Dict[str, np.ndarray],
                 labels: np.ndarray,
                 indices: Optional[List[int]] = None):
        """Initialize threat dataset.
        
        Args:
            data: Raw pandas DataFrame
            features: Dictionary of processed feature arrays by modality
            labels: Encoded labels
            indices: Subset of indices to use (for train/val/test splits)
        """
        self.data = data
        self.features = features
        self.labels = labels
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(data)))
            
        logger.info(f"Dataset initialized with {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing features and label tensors
        """
        actual_idx = self.indices[idx]
        
        sample = {
            'label': torch.tensor(self.labels[actual_idx], dtype=torch.long),
            'index': torch.tensor(actual_idx, dtype=torch.long)
        }
        
        # Add all feature modalities
        for modality, feature_array in self.features.items():
            sample[modality] = torch.tensor(feature_array[actual_idx], dtype=torch.float32)
        
        return sample


class ThreatDataLoader:
    """Main data loader for threat detection with label processing and feature extraction."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize data loader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.label_encoder = LabelEncoder()
        self.scalers: Dict[str, StandardScaler] = {}
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_features: Optional[Dict[str, np.ndarray]] = None
        self._labels: Optional[np.ndarray] = None
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the raw dataset from configured path.
        
        Returns:
            Raw pandas DataFrame
        """
        dataset_path = Path(self.config.get("data.dataset_path"))
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        # Auto-detect file format
        if dataset_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(dataset_path)
        elif dataset_path.suffix.lower() == '.csv':
            df = pd.read_csv(dataset_path)
        else:
            # Try both formats
            try:
                df = pd.read_parquet(dataset_path)
            except:
                try:
                    df = pd.read_csv(dataset_path)
                except Exception as e:
                    raise ValueError(f"Could not read dataset file: {e}")
        
        logger.info(f"Dataset loaded with shape {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Apply sample limit if configured
        sample_limit = self.config.get("data.sample_limit")
        if sample_limit is not None and sample_limit < len(df):
            df = df.sample(n=sample_limit, random_state=self.config.get("system.seed", 42))
            logger.info(f"Dataset sampled to {len(df)} rows")
        
        self._raw_data = df
        return df
    
    def extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode labels using the configured priority system.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Encoded labels array
        """
        target_columns = self.config.get("data.target_column_priority", [])
        label_mapping = self.config.get("data.label_mapping")
        anomaly_thresholds = self.config.get("data.anomaly_thresholds")
        drop_unknown = self.config.get("data.drop_unknown_labels", True)
        
        labels = []
        
        for idx, row in df.iterrows():
            label = -1  # Unknown label
            
            # Try each target column in priority order
            for col in target_columns:
                if col in df.columns and pd.notna(row[col]):
                    value = row[col]
                    
                    # Handle string labels
                    if isinstance(value, str):
                        value_norm = normalize_text(value)
                        
                        # Check mapping categories
                        for class_idx, keywords in enumerate(['benign', 'suspicious', 'malicious']):
                            if value_norm in label_mapping.get(keywords, []):
                                label = class_idx
                                break
                        
                        if label != -1:
                            break
                    
                    # Handle numeric values (for Anomaly Score fallback)
                    elif col == "Anomaly Score" and isinstance(value, (int, float)):
                        if value >= anomaly_thresholds.get('malicious', 0.8):
                            label = 2  # malicious
                        elif value >= anomaly_thresholds.get('suspicious', 0.5):
                            label = 1  # suspicious
                        else:
                            label = 0  # benign
                        break
            
            labels.append(label)
        
        labels = np.array(labels)
        
        # Handle unknown labels
        if drop_unknown:
            valid_mask = labels != -1
            logger.info(f"Dropping {(~valid_mask).sum()} samples with unknown labels")
            return labels[valid_mask], valid_mask
        else:
            # Replace -1 with a default class (benign)
            labels[labels == -1] = 0
            return labels, np.ones(len(labels), dtype=bool)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self._raw_data is None:
            self.load_dataset()
        
        df = self._raw_data
        
        # Extract labels
        labels, valid_mask = self.extract_labels(df)
        df = df[valid_mask].reset_index(drop=True)
        
        # Process features (will be implemented in preprocessing.py)
        from .preprocessing import ThreatFeatureProcessor
        processor = ThreatFeatureProcessor(self.config)
        features = processor.process_features(df)
        
        # Create train/val/test splits
        train_ratio = self.config.get("data.train_split", 0.7)
        val_ratio = self.config.get("data.val_split", 0.15)
        test_ratio = self.config.get("data.test_split", 0.15)
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            range(len(df)),
            test_size=test_ratio,
            random_state=self.config.get("system.seed", 42),
            stratify=labels
        )
        
        # Second split: separate train and validation
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=relative_val_size,
            random_state=self.config.get("system.seed", 42),
            stratify=labels[train_val_indices]
        )
        
        logger.info(f"Data splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        # Create datasets
        train_dataset = ThreatDataset(df, features, labels, train_indices)
        val_dataset = ThreatDataset(df, features, labels, val_indices)
        test_dataset = ThreatDataset(df, features, labels, test_indices)
        
        # Create data loaders
        batch_size = self.config.get("data.batch_size", 32)
        num_workers = self.config.get("data.num_workers", 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset.
        
        Returns:
            Tensor of class weights
        """
        if self._labels is None:
            if self._raw_data is None:
                self.load_dataset()
            self._labels, _ = self.extract_labels(self._raw_data)
        
        unique, counts = np.unique(self._labels, return_counts=True)
        total = len(self._labels)
        weights = total / (len(unique) * counts)
        
        # Create weight tensor for all classes (0, 1, 2)
        class_weights = torch.ones(3)
        for class_idx, weight in zip(unique, weights):
            if 0 <= class_idx <= 2:
                class_weights[class_idx] = weight
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights
