"""Knowledge graph for cyber threat intelligence and reasoning."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import torch
import torch.nn as nn
from loguru import logger


class ThreatKnowledgeGraph:
    """Knowledge graph for cybersecurity threat intelligence."""
    
    def __init__(self, config):
        """Initialize threat knowledge graph.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.relation_types = set()
        
        # Initialize with cybersecurity entities and relations
        self._initialize_cyber_entities()
        self._initialize_relations()
        
        logger.info(f"ThreatKnowledgeGraph initialized with {self.graph.number_of_nodes()} nodes")
    
    def _initialize_cyber_entities(self):
        """Initialize core cybersecurity entities."""
        # Entity types with examples
        entities = {
            'IP': ['suspicious_ip', 'malicious_ip', 'benign_ip', 'internal_ip', 'external_ip'],
            'PORT': ['port_80', 'port_443', 'port_22', 'port_3389', 'port_25'],
            'PROTOCOL': ['tcp', 'udp', 'icmp', 'http', 'https', 'ftp', 'ssh'],
            'GEOLOCATION': ['high_risk_country', 'safe_country', 'unknown_location'],
            'USER_AGENT': ['suspicious_agent', 'known_bot', 'legitimate_browser'],
            'ATTACK_TYPE': ['ddos', 'brute_force', 'malware', 'phishing', 'injection'],
            'THREAT_LEVEL': ['benign', 'suspicious', 'malicious'],
            'FAMILY': ['botnet', 'ransomware', 'trojan', 'worm', 'spyware']
        }
        
        for entity_type, instances in entities.items():
            # Add entity type node
            self.graph.add_node(entity_type, node_type='entity_type')
            
            # Add instance nodes
            for instance in instances:
                node_id = f"{entity_type}:{instance}"
                self.graph.add_node(node_id, node_type='entity', entity_type=entity_type)
                self.graph.add_edge(entity_type, node_id, relation='instance_of')
    
    def _initialize_relations(self):
        """Initialize threat-related relations."""
        relations = [
            # Network relations
            ('IP:suspicious_ip', 'PORT:port_22', 'communicates_via'),
            ('IP:malicious_ip', 'GEOLOCATION:high_risk_country', 'located_in'),
            ('PROTOCOL:tcp', 'PORT:port_80', 'uses_port'),
            ('PROTOCOL:https', 'PORT:port_443', 'uses_port'),
            
            # Attack patterns
            ('ATTACK_TYPE:brute_force', 'PORT:port_22', 'targets'),
            ('ATTACK_TYPE:brute_force', 'PROTOCOL:ssh', 'uses_protocol'),
            ('ATTACK_TYPE:ddos', 'PROTOCOL:tcp', 'exploits'),
            ('ATTACK_TYPE:malware', 'USER_AGENT:known_bot', 'uses_agent'),
            
            # Threat classifications
            ('ATTACK_TYPE:ddos', 'THREAT_LEVEL:malicious', 'classified_as'),
            ('ATTACK_TYPE:brute_force', 'THREAT_LEVEL:malicious', 'classified_as'),
            ('FAMILY:botnet', 'THREAT_LEVEL:malicious', 'classified_as'),
            ('FAMILY:ransomware', 'THREAT_LEVEL:malicious', 'classified_as'),
            
            # Risk associations
            ('GEOLOCATION:high_risk_country', 'THREAT_LEVEL:suspicious', 'increases_risk'),
            ('USER_AGENT:suspicious_agent', 'THREAT_LEVEL:suspicious', 'indicates'),
            
            # Protocol security
            ('PROTOCOL:http', 'THREAT_LEVEL:suspicious', 'less_secure_than'),
            ('PROTOCOL:https', 'THREAT_LEVEL:benign', 'preferred_over'),
        ]
        
        for src, dst, relation in relations:
            if src in self.graph.nodes and dst in self.graph.nodes:
                self.graph.add_edge(src, dst, relation=relation)
                self.relation_types.add(relation)
        
        logger.info(f"Initialized {len(relations)} knowledge relations")
    
    def add_entity(self, entity_id: str, entity_type: str, attributes: Optional[Dict] = None):
        """Add new entity to knowledge graph.
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type of entity
            attributes: Optional entity attributes
        """
        attrs = {'node_type': 'entity', 'entity_type': entity_type}
        if attributes:
            attrs.update(attributes)
        
        self.graph.add_node(entity_id, **attrs)
        
        # Connect to entity type if it exists
        if entity_type in self.graph.nodes:
            self.graph.add_edge(entity_type, entity_id, relation='instance_of')
    
    def add_relation(self, src: str, dst: str, relation: str, confidence: float = 1.0):
        """Add relation between entities.
        
        Args:
            src: Source entity ID
            dst: Destination entity ID
            relation: Relation type
            confidence: Relation confidence score
        """
        if src in self.graph.nodes and dst in self.graph.nodes:
            self.graph.add_edge(src, dst, relation=relation, confidence=confidence)
            self.relation_types.add(relation)
        else:
            logger.warning(f"Cannot add relation: {src} or {dst} not in graph")
    
    def query_relations(self, entity: str, relation: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Query relations for an entity.
        
        Args:
            entity: Entity ID to query
            relation: Optional relation type filter
            
        Returns:
            List of (source, relation, target) tuples
        """
        relations = []
        
        if entity not in self.graph.nodes:
            return relations
        
        # Outgoing relations
        for target in self.graph.successors(entity):
            edge_data = self.graph[entity][target]
            rel = edge_data.get('relation', 'unknown')
            if relation is None or rel == relation:
                relations.append((entity, rel, target))
        
        # Incoming relations
        for source in self.graph.predecessors(entity):
            edge_data = self.graph[source][entity]
            rel = edge_data.get('relation', 'unknown')
            if relation is None or rel == relation:
                relations.append((source, rel, entity))
        
        return relations
    
    def get_threat_level(self, entity: str) -> Optional[str]:
        """Get threat level for an entity based on knowledge graph.
        
        Args:
            entity: Entity ID
            
        Returns:
            Threat level string or None
        """
        # Direct classification
        for target in self.graph.successors(entity):
            if target.startswith('THREAT_LEVEL:'):
                return target.split(':')[1]
        
        # Indirect inference through relations
        suspicious_score = 0
        malicious_score = 0
        
        for rel_source, relation, rel_target in self.query_relations(entity):
            if relation in ['classified_as', 'indicates', 'increases_risk']:
                if 'suspicious' in rel_target.lower():
                    suspicious_score += 1
                elif 'malicious' in rel_target.lower():
                    malicious_score += 2
        
        if malicious_score > 0:
            return 'malicious'
        elif suspicious_score > 0:
            return 'suspicious'
        else:
            return 'benign'
    
    def explain_classification(self, entity: str) -> List[str]:
        """Explain why an entity has a certain threat classification.
        
        Args:
            entity: Entity ID
            
        Returns:
            List of explanation strings
        """
        explanations = []
        relations = self.query_relations(entity)
        
        for src, rel, dst in relations:
            if rel in ['classified_as', 'indicates', 'increases_risk']:
                if src == entity:
                    explanations.append(f"Entity {entity} is {rel.replace('_', ' ')} {dst}")
                else:
                    explanations.append(f"Entity {src} {rel.replace('_', ' ')} {entity}")
        
        return explanations
    
    def propagate_threat(self, source_entity: str, max_hops: int = 2) -> Dict[str, float]:
        """Propagate threat scores through knowledge graph.
        
        Args:
            source_entity: Source entity for propagation
            max_hops: Maximum number of hops for propagation
            
        Returns:
            Dictionary of entity -> threat_score
        """
        threat_scores = {source_entity: 1.0}
        
        for hop in range(max_hops):
            new_scores = {}
            
            for entity, score in threat_scores.items():
                if entity not in self.graph.nodes:
                    continue
                
                # Propagate to connected entities
                for neighbor in nx.all_neighbors(self.graph, entity):
                    edge_data = self.graph.get_edge_data(entity, neighbor) or self.graph.get_edge_data(neighbor, entity)
                    if edge_data:
                        relation = edge_data.get('relation', 'unknown')
                        confidence = edge_data.get('confidence', 0.5)
                        
                        # Decay threat score based on relation type and confidence
                        decay_factor = 0.7 if relation in ['classified_as', 'indicates'] else 0.3
                        propagated_score = score * decay_factor * confidence
                        
                        if neighbor not in threat_scores:
                            new_scores[neighbor] = max(new_scores.get(neighbor, 0), propagated_score)
            
            threat_scores.update(new_scores)
        
        return threat_scores
    
    def save_to_file(self, filepath: str):
        """Save knowledge graph to file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'relation_types': self.relation_types,
                'entity_embeddings': self.entity_embeddings
            }, f)
        
        logger.info(f"Knowledge graph saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load knowledge graph from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.relation_types = data['relation_types']
            self.entity_embeddings = data.get('entity_embeddings', {})
        
        logger.info(f"Knowledge graph loaded from {filepath}")


def create_threat_knowledge_graph(config) -> ThreatKnowledgeGraph:
    """Create threat knowledge graph from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ThreatKnowledgeGraph
    """
    kg = ThreatKnowledgeGraph(config)
    
    # Try to load existing KG from cache
    kg_cache_path = config.get("paths.kg_cache")
    if kg_cache_path and Path(kg_cache_path).exists():
        try:
            kg.load_from_file(kg_cache_path)
            logger.info("Loaded existing knowledge graph from cache")
        except Exception as e:
            logger.warning(f"Failed to load KG from cache: {e}")
    
    return kg
