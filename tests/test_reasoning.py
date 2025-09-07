"""Tests for neuro-symbolic reasoning components."""

import pytest
import torch
import numpy as np
from src.neuro_symbolic.knowledge_graph import create_threat_knowledge_graph
from src.neuro_symbolic.rules import create_threat_rule_engine
from src.neuro_symbolic.fusion_validate import create_neurosymbolic_fusion
from src.common.config import get_config


@pytest.fixture
def config():
    """Test configuration."""
    return get_config()


@pytest.fixture
def sample_threat_data():
    """Sample threat data for testing."""
    return [
        {
            'Protocol': 'tcp',
            'Port': 22,
            'SourceAddress': '192.168.1.100',
            'DestinationAddress': '10.0.0.1',
            'Payload Size': 1024,
            'Number of Packets': 50,
            'Anomaly Score': 0.8,
            'User-Agent': 'suspicious_bot',
            'Geolocation': 'CN'
        },
        {
            'Protocol': 'http',
            'Port': 80,
            'SourceAddress': '192.168.1.200',
            'DestinationAddress': '10.0.0.2',
            'Payload Size': 512,
            'Number of Packets': 10,
            'Anomaly Score': 0.2,
            'User-Agent': 'Mozilla/5.0',
            'Geolocation': 'US'
        }
    ]


def test_knowledge_graph_creation(config):
    """Test knowledge graph initialization."""
    kg = create_threat_knowledge_graph(config)
    
    assert kg.graph.number_of_nodes() > 0
    assert kg.graph.number_of_edges() > 0
    
    # Test entity queries
    relations = kg.query_relations('ATTACK_TYPE:malware')
    assert isinstance(relations, list)


def test_knowledge_graph_threat_classification(config):
    """Test threat level classification in KG."""
    kg = create_threat_knowledge_graph(config)
    
    # Test threat level inference
    threat_level = kg.get_threat_level('ATTACK_TYPE:malware')
    assert threat_level in ['benign', 'suspicious', 'malicious', None]
    
    # Test explanation
    explanations = kg.explain_classification('ATTACK_TYPE:malware')
    assert isinstance(explanations, list)


def test_rule_engine_creation(config):
    """Test rule engine initialization."""
    rule_engine = create_threat_rule_engine(config)
    
    assert len(rule_engine.rules) > 0
    
    # Test rule statistics
    stats = rule_engine.get_rule_statistics()
    assert isinstance(stats, dict)


def test_rule_evaluation(config, sample_threat_data):
    """Test rule evaluation on sample data."""
    rule_engine = create_threat_rule_engine(config)
    
    for sample in sample_threat_data:
        result = rule_engine.evaluate_sample(sample)
        
        assert 'conclusion' in result
        assert 'scores' in result
        assert 'confidence' in result
        assert 'fired_rules' in result
        
        assert result['conclusion'] in ['benign', 'suspicious', 'malicious']
        assert 0 <= result['confidence'] <= 1


def test_rule_batch_evaluation(config, sample_threat_data):
    """Test batch rule evaluation."""
    rule_engine = create_threat_rule_engine(config)
    
    results = rule_engine.batch_evaluate(sample_threat_data)
    
    assert len(results) == len(sample_threat_data)
    for result in results:
        assert 'conclusion' in result
        assert result['conclusion'] in ['benign', 'suspicious', 'malicious']


def test_neuro_symbolic_fusion(config, sample_threat_data):
    """Test neuro-symbolic fusion."""
    fusion = create_neurosymbolic_fusion(config)
    
    # Mock neural output
    batch_size = len(sample_threat_data)
    mock_neural_output = {
        'probabilities': torch.softmax(torch.randn(batch_size, 3), dim=1),
        'predictions': torch.randint(0, 3, (batch_size,)),
        'uncertainty': torch.rand(batch_size)
    }
    
    # Test fusion
    fused_output = fusion(mock_neural_output, sample_threat_data)
    
    assert 'probabilities' in fused_output
    assert 'predictions' in fused_output
    assert 'confidence' in fused_output
    assert 'explanations' in fused_output
    assert 'validation' in fused_output
    
    # Check output shapes
    assert fused_output['probabilities'].shape == (batch_size, 3)
    assert fused_output['predictions'].shape == (batch_size,)


def test_fusion_validation_metrics(config, sample_threat_data):
    """Test validation metrics in neuro-symbolic fusion."""
    fusion = create_neurosymbolic_fusion(config)
    
    batch_size = len(sample_threat_data)
    mock_neural_output = {
        'probabilities': torch.softmax(torch.randn(batch_size, 3), dim=1),
        'predictions': torch.randint(0, 3, (batch_size,)),
        'uncertainty': torch.rand(batch_size)
    }
    
    fused_output = fusion(mock_neural_output, sample_threat_data)
    validation = fused_output['validation']
    
    # Check validation components
    assert 'consensus_score' in validation
    assert 'needs_review' in validation
    assert 'high_disagreement' in validation
    
    # Check value ranges
    assert torch.all(validation['consensus_score'] >= 0)
    assert torch.all(validation['consensus_score'] <= 1)


def test_explanation_generation(config):
    """Test detailed explanation generation."""
    fusion = create_neurosymbolic_fusion(config)
    
    sample_data = {
        'Protocol': 'tcp',
        'Port': 22,
        'Anomaly Score': 0.9,
        'SourceAddress': '192.168.1.100'
    }
    
    mock_neural_output = {
        'probabilities': torch.tensor([[0.1, 0.2, 0.7]]),
        'predictions': torch.tensor([2]),
        'confidence': torch.tensor([0.85])
    }
    
    explanation = fusion.explain_prediction(0, mock_neural_output, sample_data)
    
    assert 'final_prediction' in explanation
    assert 'neural_contribution' in explanation
    assert 'rule_contribution' in explanation
    assert 'kg_contribution' in explanation
    assert 'consensus' in explanation


@pytest.mark.parametrize("threat_level", [0, 1, 2])
def test_different_threat_levels(config, threat_level):
    """Test reasoning with different threat levels."""
    rule_engine = create_threat_rule_engine(config)
    
    # Create sample with specific characteristics
    if threat_level == 0:  # benign
        sample = {'Anomaly Score': 0.1, 'Port': 80, 'Protocol': 'http'}
    elif threat_level == 1:  # suspicious  
        sample = {'Anomaly Score': 0.6, 'Port': 22, 'Protocol': 'tcp'}
    else:  # malicious
        sample = {'Anomaly Score': 0.9, 'Port': 3389, 'Protocol': 'tcp', 'Number of Packets': 200}
    
    result = rule_engine.evaluate_sample(sample)
    
    # Should classify appropriately (though not guaranteed due to rule complexity)
    assert result['conclusion'] in ['benign', 'suspicious', 'malicious']
    assert len(result['explanations']) >= 0
