"""Main evaluation script for the cyber threat AI system."""

import argparse
from pathlib import Path

from src.common.config import get_config
from src.common.logging import setup_logging
from src.common.utils import set_seed, save_json
from src.data.loader import ThreatDataLoader
from src.train.evaluate import create_evaluator


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate cyber threat AI model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="reports/artifacts/evaluation_results.json",
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Verify model exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    
    # Load configuration
    config = get_config(args.config)
    
    # Setup logging
    setup_logging(args.config)
    
    # Set random seed
    set_seed(config.get("system.seed", 42))
    
    # Create data loaders
    data_loader = ThreatDataLoader(args.config)
    _, _, test_loader = data_loader.create_dataloaders()
    
    # Create evaluator
    evaluator = create_evaluator(args.model, args.config)
    
    # Run evaluation
    results = evaluator.evaluate_model(test_loader)
    
    # Save results
    save_json(results, args.output)
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Macro: {results['f1_macro']:.4f}")
    print(f"F1-Weighted: {results['f1_weighted']:.4f}")
    if 'threat_detection_rate' in results:
        print(f"Threat Detection Rate: {results['threat_detection_rate']:.4f}")
    if 'false_positive_rate' in results:
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
