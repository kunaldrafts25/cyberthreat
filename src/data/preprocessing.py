"""Feature engineering and preprocessing for multi-modal threat detection."""

import hashlib
import ipaddress
import re
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer
from loguru import logger

from ..common.config import get_config
from ..common.utils import normalize_text


class ThreatFeatureProcessor:
    """Multi-modal feature processor for threat detection data."""
    
    def __init__(self, config):
        """Initialize feature processor with configuration.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        
        # Initialize tokenizer for text processing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        except:
            logger.warning("Could not load HuggingFace tokenizer, using simple tokenization")
            self.tokenizer = None
        
        self.max_seq_length = config.get("data.max_seq_length", 512)
        self.embedding_dim = config.get("data.embedding_dim", 128)
        self.image_size = config.get("data.image_size", 64)
        self.graph_max_nodes = config.get("data.graph_max_nodes", 100)
    
    def process_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process all features into multi-modal representations.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Dictionary of feature arrays by modality
        """
        logger.info("Processing multi-modal features...")
        
        features = {}
        
        # Tabular/numerical features
        features['tabular'] = self._process_tabular_features(df)
        
        # Text features (Application Layer Data, User-Agent, etc.)
        features['text'] = self._process_text_features(df)
        
        # Network graph features
        features['graph'] = self._process_graph_features(df)
        
        # Byte-level features (convert to image-like representation)
        features['image'] = self._process_byte_features(df)
        
        # Temporal features
        features['temporal'] = self._process_temporal_features(df)
        
        logger.info(f"Feature processing complete. Modalities: {list(features.keys())}")
        for modality, feat_array in features.items():
            logger.info(f"  {modality}: shape {feat_array.shape}")
        
        return features
    
    def _process_tabular_features(self, df: pd.DataFrame) -> np.ndarray:
        """Process numerical and categorical tabular features.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Processed tabular features array
        """
        numerical_cols = [
            'BTC', 'USD', 'Netflow Bytes', 'Port', 'Payload Size',
            'Number of Packets', 'Anomaly Score', 'Response Time',
            'Data Transfer Rate'
        ]
        
        categorical_cols = [
            'Protocol', 'Flag', 'Family', 'Clusters'
        ]
        
        features = []
        
        # Process numerical features
        numerical_data = []
        for col in numerical_cols:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                numerical_data.append(values.reshape(-1, 1))
        
        if numerical_data:
            numerical_array = np.hstack(numerical_data)
            if 'tabular_scaler' not in self.scalers:
                self.scalers['tabular_scaler'] = StandardScaler()
                numerical_array = self.scalers['tabular_scaler'].fit_transform(numerical_array)
            else:
                numerical_array = self.scalers['tabular_scaler'].transform(numerical_array)
            features.append(numerical_array)
        
        # Process categorical features
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    encoded = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_vals = set(df[col].astype(str))
                    known_vals = set(self.encoders[col].classes_)
                    unseen_vals = unique_vals - known_vals
                    
                    if unseen_vals:
                        # Add unseen values to encoder
                        all_vals = list(known_vals) + list(unseen_vals)
                        self.encoders[col].classes_ = np.array(all_vals)
                    
                    encoded = self.encoders[col].transform(df[col].astype(str))
                
                # One-hot encode
                n_classes = len(self.encoders[col].classes_)
                one_hot = np.eye(n_classes)[encoded]
                features.append(one_hot)
        
        # IP address features
        if 'SourceAddress' in df.columns:
            features.append(self._encode_ip_addresses(df['SourceAddress']))
        
        if 'DestinationAddress' in df.columns:
            features.append(self._encode_ip_addresses(df['DestinationAddress']))
        
        # Geolocation features
        if 'Geolocation' in df.columns:
            features.append(self._encode_geolocation(df['Geolocation']))
        
        if features:
            return np.hstack(features)
        else:
            # Return dummy features if no tabular data
            return np.zeros((len(df), 10))
    
    def _process_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Process text-based features using transformer tokenization.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Tokenized text features array
        """
        text_cols = ['Application Layer Data', 'User-Agent', 'Event Description']
        
        # Combine all text fields
        text_data = []
        for idx, row in df.iterrows():
            combined_text = ""
            for col in text_cols:
                if col in df.columns and pd.notna(row[col]):
                    combined_text += f"{col}: {row[col]} "
            
            if not combined_text.strip():
                combined_text = "[EMPTY]"
            
            text_data.append(combined_text.strip())
        
        if self.tokenizer:
            # Use HuggingFace tokenizer
            encodings = self.tokenizer(
                text_data,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors='np'
            )
            return encodings['input_ids'].astype(np.float32)
        else:
            # Fallback to TF-IDF
            if 'text_vectorizer' not in self.vectorizers:
                self.vectorizers['text_vectorizer'] = TfidfVectorizer(
                    max_features=self.max_seq_length,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                features = self.vectorizers['text_vectorizer'].fit_transform(text_data).toarray()
            else:
                features = self.vectorizers['text_vectorizer'].transform(text_data).toarray()
            
            return features.astype(np.float32)
    
    def _process_graph_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create network graph features from connection data.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Graph feature representations
        """
        # Create network graph from source/destination pairs
        G = nx.Graph()
        
        for idx, row in df.iterrows():
            src = row.get('SourceAddress', f'unknown_src_{idx}')
            dst = row.get('DestinationAddress', f'unknown_dst_{idx}')
            port = row.get('Port', 80)
            protocol = row.get('Protocol', 'tcp')
            
            # Add edge with attributes
            if pd.notna(src) and pd.notna(dst):
                G.add_edge(src, dst, port=port, protocol=protocol, flow_id=idx)
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Extract graph features for each sample
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        features = []
        for idx, row in df.iterrows():
            src = row.get('SourceAddress', f'unknown_src_{idx}')
            dst = row.get('DestinationAddress', f'unknown_dst_{idx}')
            
            # Node-level features
            src_features = self._get_node_features(G, src, node_to_idx) if src in G.nodes else np.zeros(10)
            dst_features = self._get_node_features(G, dst, node_to_idx) if dst in G.nodes else np.zeros(10)
            
            # Edge features
            if G.has_edge(src, dst):
                edge_features = self._get_edge_features(G, src, dst)
            else:
                edge_features = np.zeros(5)
            
            # Combine features
            graph_feat = np.concatenate([src_features, dst_features, edge_features])
            features.append(graph_feat)
        
        return np.array(features)
    
    def _process_byte_features(self, df: pd.DataFrame) -> np.ndarray:
        """Convert payload/packet data to image-like byte matrices.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Image-like byte feature arrays
        """
        images = []
        
        for idx, row in df.iterrows():
            # Create byte representation from available data
            byte_data = []
            
            # Use payload size and packet count to create byte pattern
            payload_size = row.get('Payload Size', 0)
            num_packets = row.get('Number of Packets', 1)
            
            # Create synthetic byte pattern (in real implementation, this would use actual packet bytes)
            if payload_size > 0:
                # Generate deterministic byte pattern based on other features
                seed_str = f"{row.get('SourceAddress', '')}{row.get('DestinationAddress', '')}{payload_size}"
                seed_bytes = hashlib.md5(seed_str.encode()).digest()
                
                # Extend to create larger byte sequence
                extended_bytes = []
                for i in range(min(payload_size, 1024)):
                    extended_bytes.append(seed_bytes[i % len(seed_bytes)])
                
                byte_data = extended_bytes
            
            # Convert to image-like matrix
            if byte_data:
                # Pad or truncate to fixed size
                target_size = self.image_size * self.image_size
                if len(byte_data) > target_size:
                    byte_data = byte_data[:target_size]
                else:
                    byte_data.extend([0] * (target_size - len(byte_data)))
                
                # Reshape to image
                image = np.array(byte_data).reshape(self.image_size, self.image_size)
                # Normalize to [0, 1]
                image = image / 255.0
            else:
                # Empty image
                image = np.zeros((self.image_size, self.image_size))
            
            images.append(image.flatten())  # Flatten for storage
        
        return np.array(images)
    
    def _process_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract temporal features from timestamps.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Temporal feature array
        """
        features = []
        
        for idx, row in df.iterrows():
            temp_features = []
            
            # Parse time if available
            if 'Time' in df.columns and pd.notna(row['Time']):
                try:
                    timestamp = pd.to_datetime(row['Time'])
                    temp_features.extend([
                        timestamp.hour / 24.0,
                        timestamp.day / 31.0,
                        timestamp.month / 12.0,
                        timestamp.weekday() / 7.0,
                        np.sin(2 * np.pi * timestamp.hour / 24),
                        np.cos(2 * np.pi * timestamp.hour / 24)
                    ])
                except:
                    temp_features.extend([0.0] * 6)
            else:
                temp_features.extend([0.0] * 6)
            
            # Response time features
            if 'Response Time' in df.columns:
                resp_time = pd.to_numeric(row.get('Response Time', 0), errors='coerce')
                temp_features.extend([
                    resp_time / 1000.0 if not np.isnan(resp_time) else 0.0,  # Normalize to seconds
                    1.0 if resp_time > 1000 else 0.0  # High latency flag
                ])
            else:
                temp_features.extend([0.0, 0.0])
            
            features.append(temp_features)
        
        return np.array(features)
    
    def _encode_ip_addresses(self, ip_series: pd.Series) -> np.ndarray:
        """Encode IP addresses as numerical features.
        
        Args:
            ip_series: Series of IP addresses
            
        Returns:
            Encoded IP features
        """
        features = []
        
        for ip in ip_series:
            ip_features = []
            
            try:
                ip_obj = ipaddress.ip_address(str(ip))
                
                if isinstance(ip_obj, ipaddress.IPv4Address):
                    # IPv4: split into octets
                    octets = str(ip).split('.')
                    ip_features = [int(octet) / 255.0 for octet in octets]
                    ip_features.append(0.0)  # IPv4 flag
                elif isinstance(ip_obj, ipaddress.IPv6Address):
                    # IPv6: use hash-based representation
                    hash_val = hash(str(ip)) % (2**32)
                    ip_features = [
                        ((hash_val >> 24) & 0xFF) / 255.0,
                        ((hash_val >> 16) & 0xFF) / 255.0,
                        ((hash_val >> 8) & 0xFF) / 255.0,
                        (hash_val & 0xFF) / 255.0,
                        1.0  # IPv6 flag
                    ]
                else:
                    ip_features = [0.0] * 5
                    
            except:
                ip_features = [0.0] * 5
            
            features.append(ip_features)
        
        return np.array(features)
    
    def _encode_geolocation(self, geo_series: pd.Series) -> np.ndarray:
        """Encode geolocation data as numerical features.
        
        Args:
            geo_series: Series of geolocation strings
            
        Returns:
            Encoded geolocation features
        """
        features = []
        
        for geo in geo_series:
            geo_features = [0.0] * 4  # lat, lon, country_code, city_code
            
            if pd.notna(geo):
                geo_str = str(geo).lower()
                
                # Extract coordinates if present (format: "lat,lon" or "city,country")
                if ',' in geo_str:
                    parts = geo_str.split(',')
                    try:
                        # Try to parse as coordinates
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        geo_features[0] = (lat + 90) / 180.0  # Normalize latitude
                        geo_features[1] = (lon + 180) / 360.0  # Normalize longitude
                    except:
                        # Parse as location names
                        city_hash = hash(parts[0].strip()) % 1000
                        country_hash = hash(parts[1].strip()) % 1000
                        geo_features[2] = city_hash / 1000.0
                        geo_features[3] = country_hash / 1000.0
                else:
                    # Single location name
                    location_hash = hash(geo_str) % 1000
                    geo_features[2] = location_hash / 1000.0
            
            features.append(geo_features)
        
        return np.array(features)
    
    def _get_node_features(self, G: nx.Graph, node: str, node_to_idx: Dict) -> np.ndarray:
        """Extract node-level graph features.
        
        Args:
            G: NetworkX graph
            node: Node identifier
            node_to_idx: Mapping from node to index
            
        Returns:
            Node feature array
        """
        if node not in G.nodes:
            return np.zeros(10)
        
        features = []
        
        # Basic centrality measures
        degree = G.degree(node)
        degree_dict = dict(G.degree())
        max_degree = max(list(degree_dict.values())) if degree_dict else 1
        features.append(degree / max_degree)
        
        try:
            betweenness = nx.betweenness_centrality(G).get(node, 0)
            closeness = nx.closeness_centrality(G).get(node, 0)
            eigenvector = nx.eigenvector_centrality(G, max_iter=100).get(node, 0)
        except:
            betweenness = closeness = eigenvector = 0
        
        features.extend([betweenness, closeness, eigenvector])
        
        # Clustering coefficient
        clustering = nx.clustering(G, node)
        features.append(clustering)
        
        # Node identifier features (hashed)
        node_hash = hash(str(node)) % 10000
        features.extend([
            (node_hash % 1000) / 1000.0,
            ((node_hash // 1000) % 10) / 10.0,
            node_to_idx.get(node, 0) / len(node_to_idx)
        ])
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10])
    
    def _get_edge_features(self, G: nx.Graph, src: str, dst: str) -> np.ndarray:
        """Extract edge-level graph features.
        
        Args:
            G: NetworkX graph
            src: Source node
            dst: Destination node
            
        Returns:
            Edge feature array
        """
        if not G.has_edge(src, dst):
            return np.zeros(5)
        
        edge_data = G[src][dst]
        features = []
        
        # Port feature
        port = edge_data.get('port', 80)
        features.append(port / 65535.0)  # Normalize port number
        
        # Protocol feature
        protocol = edge_data.get('protocol', 'tcp')
        protocol_map = {'tcp': 0.25, 'udp': 0.5, 'icmp': 0.75, 'http': 1.0}
        features.append(protocol_map.get(protocol.lower(), 0.0))
        
        # Flow ID (normalized)
        flow_id = edge_data.get('flow_id', 0)
        features.append(flow_id / 10000.0)  # Assume max ~10k flows
        
        # Edge weight (number of connections)
        weight = edge_data.get('weight', 1)
        features.append(min(weight / 10.0, 1.0))  # Cap at 10 connections
        
        # Bidirectional connection flag
        reverse_exists = G.has_edge(dst, src)
        features.append(1.0 if reverse_exists else 0.0)
        
        return np.array(features)
