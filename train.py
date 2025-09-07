"""Main training script for the cyber threat AI system."""

import argparse
from pathlib import Path

from src.common.config import get_config
from src.common.logging import setup_logging
from src.common.utils import set_seed
from src.train.train_core import create_trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cyber threat AI model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Setup logging
    setup_logging(args.config)
    
    # Set random seed
    set_seed(config.get("system.seed", 42))
    
    # Create trainer
    trainer = create_trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.best_f1 = checkpoint['best_f1']
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train the model
    history = trainer.train(args.epochs)
    
    print(f"Training completed. Best F1 score: {history['best_f1']:.4f}")


if __name__ == "__main__":
    main()
