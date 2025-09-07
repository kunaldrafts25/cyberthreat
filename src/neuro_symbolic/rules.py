"""Rule-based reasoning system for cybersecurity threat detection."""

import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from loguru import logger


class ThreatRule:
    """Individual threat detection rule."""
    
    def __init__(self, 
                 rule_id: str,
                 description: str,
                 conditions: List[Dict],
                 conclusion: str,
                 confidence: float = 1.0,
                 priority: int = 1):
        """Initialize threat rule.
        
        Args:
            rule_id: Unique rule identifier
            description: Human-readable rule description
            conditions: List of condition dictionaries
            conclusion: Rule conclusion (benign/suspicious/malicious)
            confidence: Rule confidence score
            priority: Rule priority (higher = more important)
        """
        self.rule_id = rule_id
        self.description = description
        self.conditions = conditions
        self.conclusion = conclusion
        self.confidence = confidence
        self.priority = priority
        self.activation_count = 0
    
    def evaluate(self, sample: Dict[str, Union[str, float, int]]) -> Tuple[bool, float]:
        """Evaluate rule against a sample.
        
        Args:
            sample: Sample data dictionary
            
        Returns:
            Tuple of (rule_fired, confidence_score)
        """
        for condition in self.conditions:
            if not self._evaluate_condition(condition, sample):
                return False, 0.0
        
        self.activation_count += 1
        return True, self.confidence
    
    def _evaluate_condition(self, condition: Dict, sample: Dict) -> bool:
        """Evaluate individual condition.
        
        Args:
            condition: Condition dictionary
            sample: Sample data
            
        Returns:
            True if condition is met
        """
        field = condition.get('field')
        operator = condition.get('operator')
        value = condition.get('value')
        
        if field not in sample:
            return False
        
        sample_value = sample[field]
        
        try:
            if operator == 'equals':
                return sample_value == value
            elif operator == 'not_equals':
                return sample_value != value
            elif operator == 'greater_than':
                return float(sample_value) > float(value)
            elif operator == 'less_than':
                return float(sample_value) < float(value)
            elif operator == 'greater_equal':
                return float(sample_value) >= float(value)
            elif operator == 'less_equal':
                return float(sample_value) <= float(value)
            elif operator == 'contains':
                return str(value).lower() in str(sample_value).lower()
            elif operator == 'not_contains':
                return str(value).lower() not in str(sample_value).lower()
            elif operator == 'regex':
                return bool(re.search(str(value), str(sample_value), re.IGNORECASE))
            elif operator == 'in_list':
                return sample_value in value
            elif operator == 'not_in_list':
                return sample_value not in value
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            logger.warning(f"Error evaluating condition {condition}: {e}")
            return False


class ThreatRuleEngine:
    """Rule-based engine for cybersecurity threat detection."""
    
    def __init__(self, config):
        """Initialize threat rule engine.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.rules: List[ThreatRule] = []
        self.rule_weights = {}
        
        # Initialize with cybersecurity rules
        self._initialize_cyber_rules()
        
        logger.info(f"ThreatRuleEngine initialized with {len(self.rules)} rules")
    
    def _initialize_cyber_rules(self):
        """Initialize core cybersecurity detection rules."""
        
        # High-risk port rules
        self.add_rule(ThreatRule(
            rule_id="high_risk_port_22",
            description="SSH brute force detection on port 22",
            conditions=[
                {'field': 'Port', 'operator': 'equals', 'value': 22},
                {'field': 'Number of Packets', 'operator': 'greater_than', 'value': 100},
                {'field': 'Payload Size', 'operator': 'less_than', 'value': 100}
            ],
            conclusion="suspicious",
            confidence=0.8,
            priority=3
        ))
        
        self.add_rule(ThreatRule(
            rule_id="rdp_brute_force",
            description="RDP brute force attack detection",
            conditions=[
                {'field': 'Port', 'operator': 'equals', 'value': 3389},
                {'field': 'Number of Packets', 'operator': 'greater_than', 'value': 50},
                {'field': 'Response Time', 'operator': 'greater_than', 'value': 1000}
            ],
            conclusion="malicious",
            confidence=0.9,
            priority=4
        ))
        
        # Anomaly score rules
        self.add_rule(ThreatRule(
            rule_id="high_anomaly_score",
            description="High anomaly score indicates threat",
            conditions=[
                {'field': 'Anomaly Score', 'operator': 'greater_equal', 'value': 0.8}
            ],
            conclusion="malicious",
            confidence=0.85,
            priority=3
        ))
        
        self.add_rule(ThreatRule(
            rule_id="medium_anomaly_score",
            description="Medium anomaly score indicates suspicion",
            conditions=[
                {'field': 'Anomaly Score', 'operator': 'greater_equal', 'value': 0.5},
                {'field': 'Anomaly Score', 'operator': 'less_than', 'value': 0.8}
            ],
            conclusion="suspicious",
            confidence=0.7,
            priority=2
        ))
        
        # Geolocation rules
        self.add_rule(ThreatRule(
            rule_id="high_risk_geolocation",
            description="Traffic from high-risk geolocation",
            conditions=[
                {'field': 'Geolocation', 'operator': 'in_list', 
                 'value': ['CN', 'RU', 'KP', 'IR', 'unknown']}
            ],
            conclusion="suspicious",
            confidence=0.6,
            priority=1
        ))
        
        # Protocol-based rules
        self.add_rule(ThreatRule(
            rule_id="unencrypted_high_volume",
            description="High volume unencrypted traffic",
            conditions=[
                {'field': 'Protocol', 'operator': 'in_list', 'value': ['http', 'ftp', 'telnet']},
                {'field': 'Payload Size', 'operator': 'greater_than', 'value': 10000}
            ],
            conclusion="suspicious",
            confidence=0.5,
            priority=1
        ))
        
        # User agent rules
        self.add_rule(ThreatRule(
            rule_id="bot_user_agent",
            description="Known bot user agent detected",
            conditions=[
                {'field': 'User-Agent', 'operator': 'regex', 
                 'value': r'(bot|crawler|spider|scan|attack|exploit)'}
            ],
            conclusion="suspicious",
            confidence=0.7,
            priority=2
        ))
        
        # Traffic pattern rules
        self.add_rule(ThreatRule(
            rule_id="unusual_packet_count",
            description="Unusual packet count pattern",
            conditions=[
                {'field': 'Number of Packets', 'operator': 'greater_than', 'value': 1000},
                {'field': 'Payload Size', 'operator': 'less_than', 'value': 50}
            ],
            conclusion="suspicious",
            confidence=0.6,
            priority=2
        ))
        
        # Data exfiltration rules
        self.add_rule(ThreatRule(
            rule_id="data_exfiltration",
            description="Potential data exfiltration pattern",
            conditions=[
                {'field': 'Data Transfer Rate', 'operator': 'greater_than', 'value': 1000000},
                {'field': 'Port', 'operator': 'not_in_list', 'value': [80, 443, 22, 21]}
            ],
            conclusion="malicious",
            confidence=0.8,
            priority=4
        ))
        
        # Time-based rules
        self.add_rule(ThreatRule(
            rule_id="off_hours_activity",
            description="Suspicious activity during off hours",
            conditions=[
                {'field': 'Response Time', 'operator': 'greater_than', 'value': 5000},
                {'field': 'Payload Size', 'operator': 'greater_than', 'value': 5000}
            ],
            conclusion="suspicious",
            confidence=0.4,
            priority=1
        ))
    
    def add_rule(self, rule: ThreatRule):
        """Add rule to the engine.
        
        Args:
            rule: ThreatRule to add
        """
        self.rules.append(rule)
        self.rule_weights[rule.rule_id] = rule.priority / sum(r.priority for r in self.rules)
    
    def evaluate_sample(self, sample: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, List]]:
        """Evaluate a sample against all rules.
        
        Args:
            sample: Sample data dictionary
            
        Returns:
            Dictionary containing rule evaluation results
        """
        fired_rules = []
        rule_scores = {'benign': 0.0, 'suspicious': 0.0, 'malicious': 0.0}
        explanations = []
        
        # Sort rules by priority (descending)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            fired, confidence = rule.evaluate(sample)
            
            if fired:
                fired_rules.append(rule.rule_id)
                rule_scores[rule.conclusion] += confidence * rule.priority
                explanations.append(f"Rule {rule.rule_id}: {rule.description}")
        
        # Normalize scores
        total_score = sum(rule_scores.values())
        if total_score > 0:
            for key in rule_scores:
                rule_scores[key] /= total_score
        
        # Determine final conclusion
        max_score = max(rule_scores.values())
        conclusion = 'benign'
        for category, score in rule_scores.items():
            if score == max_score:
                conclusion = category
                break
        
        return {
            'conclusion': conclusion,
            'scores': rule_scores,
            'confidence': max_score,
            'fired_rules': fired_rules,
            'explanations': explanations,
            'num_fired_rules': len(fired_rules)
        }
    
    def batch_evaluate(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate multiple samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            List of evaluation results
        """
        return [self.evaluate_sample(sample) for sample in samples]
    
    def get_rule_statistics(self) -> Dict[str, Dict]:
        """Get statistics about rule activations.
        
        Returns:
            Dictionary of rule statistics
        """
        stats = {}
        total_activations = sum(rule.activation_count for rule in self.rules)
        
        for rule in self.rules:
            stats[rule.rule_id] = {
                'description': rule.description,
                'activations': rule.activation_count,
                'activation_rate': rule.activation_count / max(total_activations, 1),
                'confidence': rule.confidence,
                'priority': rule.priority,
                'conclusion': rule.conclusion
            }
        
        return stats
    
    def update_rule_weights(self, performance_metrics: Dict[str, float]):
        """Update rule weights based on performance.
        
        Args:
            performance_metrics: Dictionary of rule_id -> performance_score
        """
        for rule_id, score in performance_metrics.items():
            if rule_id in self.rule_weights:
                # Adjust weight based on performance
                self.rule_weights[rule_id] *= (1.0 + score * 0.1)
        
        # Renormalize weights
        total_weight = sum(self.rule_weights.values())
        if total_weight > 0:
            for rule_id in self.rule_weights:
                self.rule_weights[rule_id] /= total_weight


def create_threat_rule_engine(config) -> ThreatRuleEngine:
    """Create threat rule engine from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ThreatRuleEngine
    """
    return ThreatRuleEngine(config)
