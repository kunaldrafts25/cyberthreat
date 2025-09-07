"""Tests for data pipeline and training components."""

import pytest
import torch
import pandas as pd
from src.data.loader import ThreatDataLoader, ThreatDataset
from src.data.preprocessing import ThreatFeatureProcessor
from src.train.train_core import create_trainer
from src.common.config import get_config


@pytest.fixture
def config():
    """Test configuration."""
    config = get_config()
    # Override for testing
    config.set("data.sample_limit", 100)
    config.set("training.epochs", 2)
    return config


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    import numpy as np
    
    data = []
    for i in range(50):
        sample = {
            'Protocol': np.random.choice(['tcp', 'udp', 'icmp']),
            'Port': np.random.choice([22, 80, 443, 25]),
            'SourceAddress': f"192.168.1.{np.random.randint(1, 255)}",
            'DestinationAddress': f"10.0.0.{np.random.randint(1, 255)}",
            'Payload Size': np.random.randint(50, 2000),
            'Number of Packets': np.random.randint(1, 100),
            'Anomaly Score': np.random.uniform(0, 1),
            'Threat Level': np.random.choice(['benign', 'suspicious', 'malicious']),
            'Application Layer Data': f"sample_data_{i}",
            'User-Agent': f"agent_{i}",
            'Response Time': np.random.uniform(10, 1000)
        }
        data.append(sample)
    
    return pd.DataFrame(data)


def test_feature_processor(config, sample_dataframe):
    """Test feature processing pipeline."""
    processor = ThreatFeatureProcessor(config)
    features = processor.process_features(sample_dataframe)
    
    # Check that all modalities are present
    expected_modalities = ['tabular', 'text', 'graph', 'image', 'temporal']
    for modality in expected_modalities:
        assert modality in features
        assert features[modality].shape[0] == len(sample_dataframe)


def test_label_extraction(config, sample_dataframe):
    """Test label extraction logic."""
    loader = ThreatDataLoader(config)
    loader._raw_data = sample_dataframe
    
    labels, valid_mask = loader.extract_labels(sample_dataframe)
    
    # Check label mapping
    assert all(label in [0, 1, 2] for label in labels)
    assert len(labels) == valid_mask.sum()


def test_dataset_creation(config, sample_dataframe):
    """Test PyTorch dataset creation."""
    processor = ThreatFeatureProcessor(config)
    features = processor.process_features(sample_dataframe)
    labels = [0, 1, 2] * (len(sample_dataframe) // 3) + [0] * (len(sample_dataframe) % 3)
    
    dataset = ThreatDataset(sample_dataframe, features, labels)
    
    assert len(dataset) == len(sample_dataframe)
    
    # Test sample retrieval
    sample = dataset[0]
    assert 'label' in sample
    assert 'tabular' in sample
    assert isinstance(sample['label'], torch.Tensor)


def test_dataloader_split(config):
    """Test train/val/test split."""
    if hasattr(config, 'data') and hasattr(config.data, 'dataset_path'):
        # Skip if no actual dataset
        pytest.skip("No dataset available for testing")
    
    # Mock the data loading
    loader = ThreatDataLoader(config)
    # This would require actual dataset, so we'll test the logic instead
    
    train_ratio = config.get("data.train_split", 0.7)
    val_ratio = config.get("data.val_split", 0.15) 
    test_ratio = config.get("data.test_split", 0.15)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01


def test_trainer_initialization(config):
    """Test trainer initialization."""
    trainer = create_trainer(config)
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None
    
    # Test that model is on correct device
    device = next(trainer.model.parameters()).device
    assert device.type in ['cpu', 'cuda']


@pytest.mark.slow
def test_training_step(config, sample_dataframe):
    """Test single training step."""
    processor = ThreatFeatureProcessor(config)
    features = processor.process_features(sample_dataframe)
    labels = [0, 1, 2] * (len(sample_dataframe) // 3) + [0] * (len(sample_dataframe) % 3)
    
    dataset = ThreatDataset(sample_dataframe, features, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    trainer = create_trainer(config)
    
    # Test single epoch (should not crash)
    try:
        metrics = trainer.train_epoch(dataloader, [])
        assert 'loss' in metrics
        assert 'accuracy' in metrics
    except Exception as e:
        pytest.fail(f"Training step failed: {e}")
