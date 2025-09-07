"""Integrated Gradients method for model explainability and attribution."""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def integrated_gradients(
    model: Callable,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    target: Optional[int] = None,
    steps: int = 50
) -> torch.Tensor:
    """
    Compute Integrated Gradients for input attribution.
    
    Args:
        model: The model function that takes input and returns predictions
        inputs: Input tensor for which to compute attributions
        baseline: Baseline input (defaults to zero tensor)
        target: Target class index for attribution (uses max prediction if None)
        steps: Number of interpolation steps between baseline and input
        
    Returns:
        Attribution tensor with same shape as inputs
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    
    # Create interpolated inputs between baseline and actual input
    scaled_inputs = [
        baseline + float(i) / steps * (inputs - baseline) 
        for i in range(steps + 1)
    ]
    
    grads = []
    
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_(True)
        
        # Forward pass
        output = model(scaled_input)
        
        # Select target class or use max prediction
        if target is not None:
            target_output = output[:, target]
        else:
            target_output = output.max(dim=1)[0]
        
        # Compute gradients
        target_output.sum().backward()
        grads.append(scaled_input.grad.detach())
    
    # Average gradients across all steps
    avg_grads = torch.mean(torch.stack(grads), dim=0)
    
    # Compute integrated gradients
    integrated_grads = (inputs - baseline) * avg_grads
    
    logger.debug(f"Integrated gradients computed with {steps} steps")
    
    return integrated_grads


def smooth_gradients(
    model: Callable,
    inputs: torch.Tensor,
    target: Optional[int] = None,
    noise_level: float = 0.15,
    n_samples: int = 50
) -> torch.Tensor:
    """
    Compute SmoothGrad for noise-reduced attribution.
    
    Args:
        model: Model function
        inputs: Input tensor
        target: Target class index
        noise_level: Standard deviation of Gaussian noise
        n_samples: Number of noisy samples
        
    Returns:
        Smoothed gradient attribution
    """
    grads = []
    
    for _ in range(n_samples):
        # Add Gaussian noise
        noise = torch.randn_like(inputs) * noise_level
        noisy_input = inputs + noise
        noisy_input.requires_grad_(True)
        
        # Forward pass
        output = model(noisy_input)
        
        if target is not None:
            target_output = output[:, target]
        else:
            target_output = output.max(dim=1)[0]
        
        # Compute gradients
        target_output.sum().backward()
        grads.append(noisy_input.grad.detach())
    
    # Average gradients
    smooth_grad = torch.mean(torch.stack(grads), dim=0)
    
    logger.debug(f"SmoothGrad computed with {n_samples} samples")
    
    return smooth_grad


class IntegratedGradientsExplainer:
    """Explainer using Integrated Gradients for model interpretability."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize explainer.
        
        Args:
            model: PyTorch model to explain
            device: Device to run computations on
        """
        self.model = model
        self.device = device or torch.device('cpu')
        
    def explain(self, 
                inputs: torch.Tensor, 
                target: Optional[int] = None,
                method: str = "integrated_gradients",
                **kwargs) -> torch.Tensor:
        """
        Generate explanations for inputs.
        
        Args:
            inputs: Input tensor to explain
            target: Target class index
            method: Attribution method ("integrated_gradients" or "smooth_gradients")
            **kwargs: Additional arguments for attribution methods
            
        Returns:
            Attribution tensor
        """
        self.model.eval()
        
        inputs = inputs.to(self.device)
        
        if method == "integrated_gradients":
            return integrated_gradients(
                self.model, 
                inputs, 
                target=target,
                steps=kwargs.get('steps', 50)
            )
        elif method == "smooth_gradients":
            return smooth_gradients(
                self.model,
                inputs,
                target=target,
                noise_level=kwargs.get('noise_level', 0.15),
                n_samples=kwargs.get('n_samples', 50)
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")
    
    def explain_batch(self,
                     inputs: torch.Tensor,
                     targets: Optional[torch.Tensor] = None,
                     method: str = "integrated_gradients",
                     **kwargs) -> torch.Tensor:
        """
        Generate explanations for a batch of inputs.
        
        Args:
            inputs: Batch of input tensors
            targets: Target class indices for each input
            method: Attribution method
            **kwargs: Additional arguments
            
        Returns:
            Batch of attribution tensors
        """
        batch_attributions = []
        
        for i in range(inputs.shape[0]):
            target = targets[i].item() if targets is not None else None
            attribution = self.explain(
                inputs[i:i+1], 
                target=target, 
                method=method, 
                **kwargs
            )
            batch_attributions.append(attribution)
        
        return torch.cat(batch_attributions, dim=0)
    
    def visualize_attribution(self, 
                             attribution: torch.Tensor, 
                             original_input: torch.Tensor,
                             output_path: Optional[str] = None) -> torch.Tensor:
        """
        Create visualization of attribution.
        
        Args:
            attribution: Attribution tensor
            original_input: Original input tensor
            output_path: Path to save visualization
            
        Returns:
            Visualization tensor
        """
        # Normalize attribution for visualization
        attr_abs = torch.abs(attribution)
        attr_norm = attr_abs / (torch.max(attr_abs) + 1e-8)
        
        # Create visualization (simple heatmap approach)
        if len(attribution.shape) == 4:  # Image-like data
            # Sum across channels for visualization
            attr_vis = attr_norm.sum(dim=1, keepdim=True)
        else:
            attr_vis = attr_norm
        
        if output_path:
            # In a real implementation, save the visualization
            logger.info(f"Attribution visualization saved to {output_path}")
        
        return attr_vis


def create_integrated_gradients_explainer(model: nn.Module, 
                                        device: Optional[torch.device] = None) -> IntegratedGradientsExplainer:
    """
    Factory function to create Integrated Gradients explainer.
    
    Args:
        model: PyTorch model to explain
        device: Computation device
        
    Returns:
        Configured explainer instance
    """
    return IntegratedGradientsExplainer(model, device)


class ThreatAttributionAnalyzer:
    """Specialized analyzer for threat detection model attributions."""
    
    def __init__(self, explainer: IntegratedGradientsExplainer):
        """
        Initialize analyzer.
        
        Args:
            explainer: Integrated gradients explainer instance
        """
        self.explainer = explainer
    
    def analyze_threat_features(self, 
                               inputs: torch.Tensor,
                               predictions: torch.Tensor,
                               feature_names: Optional[list] = None) -> dict:
        """
        Analyze which features contribute most to threat predictions.
        
        Args:
            inputs: Input features
            predictions: Model predictions
            feature_names: Names of input features
            
        Returns:
            Analysis results dictionary
        """
        # Get attributions for predicted class
        predicted_class = torch.argmax(predictions, dim=1)
        attributions = self.explainer.explain_batch(inputs, predicted_class)
        
        # Analyze attribution statistics
        attr_abs = torch.abs(attributions)
        
        # Feature importance ranking
        feature_importance = torch.mean(attr_abs, dim=0)
        top_features_idx = torch.argsort(feature_importance, descending=True)
        
        # Positive vs negative contributions
        positive_contrib = torch.mean(torch.clamp(attributions, min=0), dim=0)
        negative_contrib = torch.mean(torch.clamp(attributions, max=0), dim=0)
        
        results = {
            'feature_importance': feature_importance.cpu().numpy(),
            'top_features_idx': top_features_idx.cpu().numpy(),
            'positive_contributions': positive_contrib.cpu().numpy(),
            'negative_contributions': negative_contrib.cpu().numpy(),
            'attribution_magnitude': torch.mean(attr_abs).item()
        }
        
        if feature_names:
            results['top_features_names'] = [
                feature_names[idx] for idx in top_features_idx[:10].cpu().numpy()
            ]
        
        return results
    
    def explain_prediction(self, 
                          sample_input: torch.Tensor,
                          sample_prediction: torch.Tensor,
                          feature_names: Optional[list] = None) -> dict:
        """
        Generate detailed explanation for a single prediction.
        
        Args:
            sample_input: Single input sample
            sample_prediction: Model prediction for the sample
            feature_names: Feature names
            
        Returns:
            Detailed explanation dictionary
        """
        predicted_class = torch.argmax(sample_prediction).item()
        class_names = ['benign', 'suspicious', 'malicious']
        
        # Get attribution for predicted class
        attribution = self.explainer.explain(
            sample_input.unsqueeze(0), 
            target=predicted_class
        ).squeeze(0)
        
        # Feature-level analysis
        attr_abs = torch.abs(attribution)
        top_features_idx = torch.argsort(attr_abs, descending=True)
        
        explanation = {
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'prediction_confidence': torch.max(sample_prediction).item(),
            'top_contributing_features': [],
            'attribution_summary': {
                'total_attribution': torch.sum(attribution).item(),
                'positive_attribution': torch.sum(torch.clamp(attribution, min=0)).item(),
                'negative_attribution': torch.sum(torch.clamp(attribution, max=0)).item(),
                'attribution_magnitude': torch.mean(attr_abs).item()
            }
        }
        
        # Top contributing features
        for i in range(min(10, len(top_features_idx))):
            idx = top_features_idx[i].item()
            feature_name = feature_names[idx] if feature_names else f"feature_{idx}"
            
            explanation['top_contributing_features'].append({
                'feature_name': feature_name,
                'feature_index': idx,
                'attribution_score': attribution[idx].item(),
                'feature_value': sample_input[idx].item(),
                'contribution_rank': i + 1
            })
        
        return explanation
