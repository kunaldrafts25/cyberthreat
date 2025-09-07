"""Core training loop for the threat detection system."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger

from ..common.config import get_config
from ..common.utils import get_device, Timer, clip_gradient_norm
from ..common.metrics import MetricsTracker
from ..models.threat_system import create_threat_detection_system
from ..data.loader import ThreatDataLoader
from ..neuro_symbolic.fusion_validate import create_neurosymbolic_fusion


class ThreatTrainer:
    """Core trainer for the threat detection system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize threat trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.device = get_device(self.config.get("system.device", "auto"))
        
        # Initialize model
        self.model = create_threat_detection_system(config_path)
        self.model = self.model.to(self.device)
        
        # Initialize neuro-symbolic fusion
        self.neuro_symbolic = create_neurosymbolic_fusion(self.config)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("training.learning_rate", 1e-4),
            weight_decay=self.config.get("training.weight_decay", 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.epoch = 0
        self.best_f1 = 0.0
        self.training_history = []
        
        # Metrics tracking
        self.train_metrics = MetricsTracker(num_classes=3, class_names=['benign', 'suspicious', 'malicious'])
        self.val_metrics = MetricsTracker(num_classes=3, class_names=['benign', 'suspicious', 'malicious'])
        
        logger.info("ThreatTrainer initialized")
    
    def train_epoch(self, train_loader: DataLoader, sample_data: List[Dict]) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            sample_data: Raw sample data for neuro-symbolic reasoning
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass through main model
            outputs = self.model(batch, mode="training")
            
            # Apply neuro-symbolic fusion if sample data available
            if batch_idx < len(sample_data):
                batch_sample_data = sample_data[batch_idx * batch['label'].size(0):(batch_idx + 1) * batch['label'].size(0)]
                if batch_sample_data:
                    ns_outputs = self.neuro_symbolic(outputs, batch_sample_data)
                    final_logits = ns_outputs['probabilities']  # Already softmax
                    final_logits = torch.log(final_logits + 1e-8)  # Convert to log probabilities
                else:
                    final_logits = outputs['logits']
            else:
                final_logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(final_logits, batch['label'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_gradient_norm(self.model, self.config.get("training.gradient_clip", 1.0))
            
            self.optimizer.step()
            
            # Update metrics
            predictions = torch.argmax(final_logits, dim=1)
            probabilities = torch.softmax(final_logits, dim=1)
            
            self.train_metrics.update(
                predictions, 
                batch['label'],
                probabilities,
                outputs.get('uncertainty', None)
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % self.config.get("training.log_every_n_steps", 100) == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
        
        # Compute epoch metrics
        train_metrics = self.train_metrics.compute()
        train_metrics['loss'] = total_loss / num_batches
        
        return train_metrics
    
    def validate_epoch(self, val_loader: DataLoader, sample_data: List[Dict]) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            sample_data: Raw sample data for neuro-symbolic reasoning
            
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(batch, mode="inference")
                
                # Apply neuro-symbolic fusion
                if batch_idx < len(sample_data):
                    batch_sample_data = sample_data[batch_idx * batch['label'].size(0):(batch_idx + 1) * batch['label'].size(0)]
                    if batch_sample_data:
                        ns_outputs = self.neuro_symbolic(outputs, batch_sample_data)
                        final_probs = ns_outputs['probabilities']
                        final_logits = torch.log(final_probs + 1e-8)
                    else:
                        final_logits = outputs['logits']
                        final_probs = torch.softmax(final_logits, dim=1)
                else:
                    final_logits = outputs['logits']
                    final_probs = torch.softmax(final_logits, dim=1)
                
                # Compute loss
                loss = self.criterion(final_logits, batch['label'])
                
                # Update metrics
                predictions = torch.argmax(final_logits, dim=1)
                
                self.val_metrics.update(
                    predictions,
                    batch['label'],
                    final_probs,
                    outputs.get('uncertainty', None)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute epoch metrics
        val_metrics = self.val_metrics.compute()
        val_metrics['loss'] = total_loss / num_batches
        
        return val_metrics
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List]:
        """Main training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config default if None)
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.get("training.epochs", 50)
        
        # Initialize data loaders
        data_loader = ThreatDataLoader(self.config)
        train_loader, val_loader, _ = data_loader.create_dataloaders()
        
        # Get raw sample data for neuro-symbolic reasoning (simplified)
        sample_data = []  # In practice, extract from data_loader._raw_data
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        early_stopping_patience = self.config.get("training.early_stopping_patience", 10)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            with Timer(f"Epoch {self.epoch}"):
                # Training
                train_metrics = self.train_epoch(train_loader, sample_data)
                
                # Validation
                val_metrics = self.validate_epoch(val_loader, sample_data)
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['f1_macro'])
                
                # Log metrics
                logger.info(f"Epoch {self.epoch} - "
                           f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Train F1: {train_metrics['f1_macro']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val F1: {val_metrics['f1_macro']:.4f}")
                
                # Save checkpoint if best model
                if val_metrics['f1_macro'] > self.best_f1:
                    self.best_f1 = val_metrics['f1_macro']
                    self.save_checkpoint('best_model.pt')
                    patience_counter = 0
                    logger.info(f"New best model saved with F1: {self.best_f1:.4f}")
                else:
                    patience_counter += 1
                
                # Record history
                epoch_history = {
                    'epoch': self.epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                self.training_history.append(epoch_history)
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {self.epoch} epochs")
                    break
        
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        
        return {
            'history': self.training_history,
            'best_f1': self.best_f1
        }
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_dir = Path(self.config.get("paths.checkpoints_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def create_trainer(config_path: Optional[str] = None) -> ThreatTrainer:
    """Create threat trainer from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ThreatTrainer
    """
    return ThreatTrainer(config_path)
