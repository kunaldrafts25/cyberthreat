"""Tests for model components."""

import pytest
import torch
from src.models.vit_small import create_vit_small
from src.models.gcn import create_network_gcn
from src.models.transformer_text import create_threat_text_processor
from src.models.fusion import create_multimodal_fusion
from src.models.triage_head import create_threat_triage_head
from src.models.threat_system import create_threat_detection_system
from src.common.config import get_config


@pytest.fixture
def config():
    """Test configuration."""
    return get_config()


@pytest.fixture
def sample_batch():
    """Sample input batch."""
    return {
        'tabular': torch.randn(4, 256),
        'text': torch.randint(0, 1000, (4, 128)),
        'graph': torch.randn(4, 25),
        'image': torch.randn(4, 4096),  # 64x64 flattened
        'temporal': torch.randn(4, 8),
        'label': torch.randint(0, 3, (4,))
    }


def test_vit_small_forward(config, sample_batch):
    """Test ViT small forward pass."""
    model = create_vit_small(config)
    output = model(sample_batch['image'])
    
    assert 'cls_features' in output
    assert 'global_features' in output
    assert output['cls_features'].shape[0] == 4


def test_gcn_forward(config, sample_batch):
    """Test GCN forward pass."""
    model = create_network_gcn(config)
    output = model(sample_batch['graph'])
    
    assert 'embeddings' in output
    assert output['embeddings'].shape[0] == 4


def test_text_processor_forward(config, sample_batch):
    """Test text processor forward pass."""
    model = create_threat_text_processor(config)
    output = model(sample_batch['text'])
    
    assert 'embeddings' in output
    assert output['embeddings'].shape[0] == 4


def test_fusion_network(config):
    """Test multi-modal fusion."""
    model = create_multimodal_fusion(config)
    
    modal_features = {
        'tabular': torch.randn(4, 256),
        'text': torch.randn(4, 384),
        'graph': torch.randn(4, 256),
        'image': torch.randn(4, 768),
        'temporal': torch.randn(4, 8)
    }
    
    output = model(modal_features)
    assert 'fused_embedding' in output
    assert output['fused_embedding'].shape[0] == 4


def test_triage_head(config):
    """Test triage head with uncertainty."""
    model = create_threat_triage_head(config)
    features = torch.randn(4, 512)
    
    output = model(features, return_uncertainty=True)
    
    assert 'probabilities' in output
    assert 'uncertainty' in output
    assert output['probabilities'].shape == (4, 3)


def test_threat_system_integration(config, sample_batch):
    """Test complete threat detection system."""
    model = create_threat_detection_system()
    
    # Test inference mode
    output = model(sample_batch, mode="inference")
    
    assert 'logits' in output
    assert 'probabilities' in output
    assert output['logits'].shape == (4, 3)


def test_model_shapes_consistency(config, sample_batch):
    """Test that all models maintain batch dimension consistency."""
    models = {
        'vit': create_vit_small(config),
        'gcn': create_network_gcn(config),
        'text': create_threat_text_processor(config),
        'triage': create_threat_triage_head(config)
    }
    
    batch_size = 4
    
    # Test ViT
    vit_out = models['vit'](sample_batch['image'])
    assert vit_out['embeddings'].shape[0] == batch_size
    
    # Test GCN
    gcn_out = models['gcn'](sample_batch['graph'])
    assert gcn_out['embeddings'].shape[0] == batch_size
    
    # Test Text
    text_out = models['text'](sample_batch['text'])
    assert text_out['embeddings'].shape[0] == batch_size
    
    # Test Triage
    triage_out = models['triage'](torch.randn(batch_size, 512))
    assert triage_out['probabilities'].shape[0] == batch_size
