"""Federated learning components for distributed threat detection."""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from loguru import logger

from ..common.config import get_config
from ..common.utils import get_device
from ..common.metrics import MetricsTracker


class FederatedClient:
    """Federated learning client for distributed threat detection training."""
    
    def __init__(self, 
                 client_id: int,
                 model: nn.Module,
                 train_loader: DataLoader,
                 config):
        """Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model: Local model copy
            train_loader: Client's training data
            config: System configuration
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = get_device(config.get("system.device", "auto"))
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("training.learning_rate", 1e-4)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_tracker = MetricsTracker()
        
        logger.info(f"Federated client {client_id} initialized with {len(train_loader.dataset)} samples")
    
    def local_train(self, global_round: int) -> Dict[str, float]:
        """Perform local training for specified number of epochs.
        
        Args:
            global_round: Current global round number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics_tracker.reset()
        
        local_epochs = self.config.get("federated.local_epochs", 3)
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            
            for batch in self.train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.loss_fn(outputs['logits'], batch['label'])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                predictions = torch.argmax(outputs['logits'], dim=1)
                self.metrics_tracker.update(
                    predictions, 
                    batch['label'],
                    torch.softmax(outputs['logits'], dim=1),
                    outputs.get('uncertainty', None)
                )
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
            logger.debug(f"Client {self.client_id} - Round {global_round}, Epoch {epoch+1}/{local_epochs}, Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        # Compute final metrics
        metrics = self.metrics_tracker.compute()
        metrics['loss'] = total_loss / num_batches
        metrics['num_samples'] = len(self.train_loader.dataset)
        
        return metrics
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {name: param.clone() for name, param in self.model.state_dict().items()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from server.
        
        Args:
            parameters: Dictionary of model parameters
        """
        self.model.load_state_dict(parameters)
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate local model on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                loss = self.loss_fn(outputs['logits'], batch['label'])
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                metrics_tracker.update(
                    predictions,
                    batch['label'],
                    torch.softmax(outputs['logits'], dim=1),
                    outputs.get('uncertainty', None)
                )
                
                total_loss += loss.item()
        
        metrics = metrics_tracker.compute()
        metrics['val_loss'] = total_loss / len(val_loader)
        
        return metrics


class FederatedServer:
    """Federated learning server for aggregating client updates."""
    
    def __init__(self, 
                 global_model: nn.Module,
                 config):
        """Initialize federated server.
        
        Args:
            global_model: Global model to be distributed
            config: System configuration
        """
        self.global_model = global_model
        self.config = config
        self.device = get_device(config.get("system.device", "auto"))
        self.global_model.to(self.device)
        
        self.clients: List[FederatedClient] = []
        self.round_metrics: List[Dict] = []
        
        logger.info("Federated server initialized")
    
    def add_client(self, client: FederatedClient) -> None:
        """Add a client to the federation.
        
        Args:
            client: FederatedClient instance
        """
        self.clients.append(client)
        logger.info(f"Added client {client.client_id} to federation. Total clients: {len(self.clients)}")
    
    def create_clients_from_data(self, 
                                train_loader: DataLoader,
                                num_clients: Optional[int] = None) -> List[FederatedClient]:
        """Create federated clients by partitioning training data.
        
        Args:
            train_loader: Combined training data loader
            num_clients: Number of clients to create (defaults to config)
            
        Returns:
            List of created FederatedClient instances
        """
        if num_clients is None:
            num_clients = self.config.get("federated.num_clients", 10)
        
        dataset = train_loader.dataset
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients
        
        clients = []
        
        for client_id in range(num_clients):
            # Create data partition for this client
            start_idx = client_id * samples_per_client
            if client_id == num_clients - 1:
                # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = start_idx + samples_per_client
            
            client_indices = list(range(start_idx, end_idx))
            client_subset = Subset(dataset, client_indices)
            
            client_loader = DataLoader(
                client_subset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers
            )
            
            # Create local model copy
            local_model = copy.deepcopy(self.global_model)
            
            client = FederatedClient(
                client_id=client_id,
                model=local_model,
                train_loader=client_loader,
                config=self.config
            )
            
            clients.append(client)
            self.add_client(client)
        
        logger.info(f"Created {len(clients)} federated clients with ~{samples_per_client} samples each")
        return clients
    
    def federated_averaging(self, client_updates: List[Dict[str, torch.Tensor]],
                           client_weights: Optional[List[float]] = None) -> None:
        """Perform FedAvg aggregation of client updates.
        
        Args:
            client_updates: List of client parameter dictionaries
            client_weights: Optional weights for clients (based on data size)
        """
        if not client_updates:
            return
        
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first client
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            # Weighted average of all client parameters
            aggregated_param = torch.zeros_like(client_updates[0][param_name])
            
            for client_params, weight in zip(client_updates, client_weights):
                aggregated_param += weight * client_params[param_name]
            
            aggregated_params[param_name] = aggregated_param
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        logger.debug("Global model updated with federated averaging")
    
    def train_round(self, round_num: int) -> Dict[str, float]:
        """Execute one round of federated training.
        
        Args:
            round_num: Current round number
            
        Returns:
            Aggregated metrics from all clients
        """
        logger.info(f"Starting federated training round {round_num}")
        
        # Distribute global model to all clients
        global_params = {name: param.clone() 
                        for name, param in self.global_model.state_dict().items()}
        
        for client in self.clients:
            client.set_model_parameters(global_params)
        
        # Collect client updates
        client_updates = []
        client_metrics = []
        client_weights = []
        
        for client in self.clients:
            # Local training
            metrics = client.local_train(round_num)
            client_metrics.append(metrics)
            
            # Get updated parameters
            updated_params = client.get_model_parameters()
            client_updates.append(updated_params)
            
            # Weight by number of samples
            client_weights.append(metrics['num_samples'])
        
        # Aggregate updates
        self.federated_averaging(client_updates, client_weights)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(client_metrics)
        aggregated_metrics['round'] = round_num
        
        self.round_metrics.append(aggregated_metrics)
        
        logger.info(f"Round {round_num} completed. Avg loss: {aggregated_metrics['avg_loss']:.4f}, "
                   f"Avg F1: {aggregated_metrics['avg_f1_macro']:.4f}")
        
        return aggregated_metrics
    
    def _aggregate_metrics(self, client_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from all clients.
        
        Args:
            client_metrics: List of client metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        aggregated = {}
        total_samples = sum(m['num_samples'] for m in client_metrics)
        
        # Weighted average of key metrics
        key_metrics = ['loss', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        for metric in key_metrics:
            if all(metric in m for m in client_metrics):
                weighted_sum = sum(m[metric] * m['num_samples'] for m in client_metrics)
                aggregated[f'avg_{metric}'] = weighted_sum / total_samples
        
        # Additional statistics
        aggregated['num_clients'] = len(client_metrics)
        aggregated['total_samples'] = total_samples
        
        return aggregated
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model.
        
        Returns:
            Current global model
        """
        return self.global_model
    
    def save_checkpoint(self, filepath: str, round_num: int) -> None:
        """Save federated learning checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            round_num: Current round number
        """
        checkpoint = {
            'round': round_num,
            'global_model_state_dict': self.global_model.state_dict(),
            'metrics_history': self.round_metrics,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Federated checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """Load federated learning checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Round number from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        self.round_metrics = checkpoint['metrics_history']
        
        round_num = checkpoint['round']
        logger.info(f"Federated checkpoint loaded from {filepath}, round {round_num}")
        
        return round_num


def create_federated_setup(model: nn.Module, 
                          train_loader: DataLoader,
                          config) -> Tuple[FederatedServer, List[FederatedClient]]:
    """Create complete federated learning setup.
    
    Args:
        model: Base model architecture
        train_loader: Training data loader
        config: System configuration
        
    Returns:
        Tuple of (FederatedServer, List[FederatedClient])
    """
    # Create server
    server = FederatedServer(model, config)
    
    # Create clients
    clients = server.create_clients_from_data(train_loader)
    
    logger.info(f"Federated setup complete with {len(clients)} clients")
    
    return server, clients
