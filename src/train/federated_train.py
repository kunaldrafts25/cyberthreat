"""Federated training orchestration for distributed threat detection."""

from typing import Dict, List, Optional
import torch
from loguru import logger

from ..common.config import get_config
from ..common.metrics import MetricsTracker
from ..data.federated import create_federated_setup
from ..models.threat_system import create_threat_detection_system


class FederatedTrainingOrchestrator:
    """Orchestrator for federated learning training process."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize federated trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.global_metrics_history = []
        
    def train_federated(self, num_rounds: int = 100) -> Dict:
        """Run federated training for specified rounds.
        
        Args:
            num_rounds: Number of federated rounds
            
        Returns:
            Training history dictionary
        """
        from ..data.loader import ThreatDataLoader
        
        # Initialize components
        data_loader = ThreatDataLoader(self.config)
        train_loader, val_loader, _ = data_loader.create_dataloaders()
        
        # Create federated setup
        base_model = create_threat_detection_system()
        server, clients = create_federated_setup(base_model, train_loader, self.config)
        
        logger.info(f"Starting federated training for {num_rounds} rounds with {len(clients)} clients")
        
        # Training rounds
        for round_num in range(1, num_rounds + 1):
            round_metrics = server.train_round(round_num)
            self.global_metrics_history.append(round_metrics)
            
            # Evaluation every 10 rounds
            if round_num % 10 == 0:
                eval_metrics = self._evaluate_global_model(server, val_loader)
                logger.info(f"Round {round_num} - Global F1: {eval_metrics.get('f1_macro', 0):.4f}")
                
                # Save checkpoint
                checkpoint_path = f"checkpoints/federated_round_{round_num}.pt"
                server.save_checkpoint(checkpoint_path, round_num)
        
        return {
            'rounds_completed': num_rounds,
            'global_history': self.global_metrics_history,
            'final_model': server.get_global_model()
        }
    
    def _evaluate_global_model(self, server, val_loader) -> Dict:
        """Evaluate global model on validation set."""
        model = server.get_global_model()
        model.eval()
        
        metrics_tracker = MetricsTracker(3, ['benign', 'suspicious', 'malicious'])
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch, mode="inference")
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                
                metrics_tracker.update(predictions, batch['label'], probabilities)
        
        return metrics_tracker.compute()


def create_federated_trainer(config_path: Optional[str] = None) -> FederatedTrainingOrchestrator:
    """Create federated training orchestrator."""
    return FederatedTrainingOrchestrator(config_path)
