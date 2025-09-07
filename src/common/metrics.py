"""Metrics calculation and tracking for the cyber threat AI system."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score
)
from loguru import logger


class MetricsTracker:
    """Track and compute various classification metrics for threat detection."""
    
    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        """Initialize metrics tracker.
        
        Args:
            num_classes: Number of classes (default: 3 for benign/suspicious/malicious)
            class_names: Names of classes for reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.probabilities: List[np.ndarray] = []
        self.uncertainties: List[float] = []
    
    def update(self, 
               preds: Union[torch.Tensor, np.ndarray],
               targets: Union[torch.Tensor, np.ndarray],
               probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
               uncertainties: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None) -> None:
        """Update metrics with batch predictions and targets.
        
        Args:
            preds: Predicted class labels
            targets: Ground truth labels
            probs: Class probabilities (optional)
            uncertainties: Prediction uncertainties (optional)
        """
        # Convert to numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if uncertainties is not None and isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().numpy()
        
        self.predictions.extend(preds.flatten())
        self.targets.extend(targets.flatten())
        
        if probs is not None:
            self.probabilities.extend(probs)
        
        if uncertainties is not None:
            if isinstance(uncertainties, (list, tuple)):
                self.uncertainties.extend(uncertainties)
            else:
                self.uncertainties.extend(uncertainties.flatten())
    
    def compute(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute all metrics from accumulated predictions and targets.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions or not self.targets:
            logger.warning("No predictions or targets available for metric computation")
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class and macro metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                per_class_metrics[f'{class_name}_precision'] = precision[i]
                per_class_metrics[f'{class_name}_recall'] = recall[i]
                per_class_metrics[f'{class_name}_f1'] = f1[i]
                per_class_metrics[f'{class_name}_support'] = support[i]
        
        metrics['per_class'] = per_class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # AUC-ROC (if probabilities available and binary/multiclass)
        if self.probabilities and len(self.probabilities) > 0:
            try:
                y_probs = np.array(self.probabilities)
                if self.num_classes == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_probs[:, 1])
                elif self.num_classes > 2:
                    # Multi-class (one-vs-rest)
                    metrics['auc_roc_ovr'] = roc_auc_score(
                        y_true, y_probs, multi_class='ovr', average='macro'
                    )
            except Exception as e:
                logger.warning(f"Could not compute AUC-ROC: {e}")
        
        # Uncertainty metrics (if available)
        if self.uncertainties:
            uncertainties = np.array(self.uncertainties)
            metrics['mean_uncertainty'] = np.mean(uncertainties)
            metrics['std_uncertainty'] = np.std(uncertainties)
            metrics['max_uncertainty'] = np.max(uncertainties)
            metrics['min_uncertainty'] = np.min(uncertainties)
            
            # Correlation between uncertainty and correctness
            correct = (y_pred == y_true).astype(float)
            if len(correct) == len(uncertainties):
                correlation = np.corrcoef(uncertainties, correct)[0, 1]
                metrics['uncertainty_correctness_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        return metrics
    
    def get_best_threshold(self, metric: str = 'f1_macro') -> Tuple[float, Dict[str, float]]:
        """Find best classification threshold for a given metric (requires probabilities).
        
        Args:
            metric: Metric to optimize ('f1_macro', 'accuracy', etc.)
            
        Returns:
            Tuple of (best_threshold, metrics_at_threshold)
        """
        if not self.probabilities:
            raise ValueError("Probabilities required for threshold optimization")
        
        y_true = np.array(self.targets)
        y_probs = np.array(self.probabilities)
        
        best_score = -1
        best_threshold = 0.5
        best_metrics = {}
        
        # Test thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 1.0, 0.05):
            y_pred_thresh = np.argmax(y_probs, axis=1)
            # For binary case, adjust based on threshold
            if self.num_classes == 2:
                y_pred_thresh = (y_probs[:, 1] >= threshold).astype(int)
            
            # Compute metrics
            temp_tracker = MetricsTracker(self.num_classes, self.class_names)
            temp_tracker.update(y_pred_thresh, y_true, y_probs)
            metrics = temp_tracker.compute()
            
            score = metrics.get(metric, 0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics


def compute_entropy(probs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Compute entropy of probability distributions.
    
    Args:
        probs: Probability distributions [batch_size, num_classes]
        
    Returns:
        Entropy values [batch_size]
    """
    if isinstance(probs, torch.Tensor):
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return entropy
    else:
        eps = 1e-8
        entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
        return entropy


def compute_confidence(probs: Union[torch.Tensor, np.ndarray],
                      uncertainties: Optional[Union[torch.Tensor, np.ndarray]] = None,
                      uncertainty_weight: float = 0.5) -> Union[torch.Tensor, np.ndarray]:
    """Compute confidence scores combining probabilities and uncertainties.
    
    Args:
        probs: Class probabilities [batch_size, num_classes]
        uncertainties: Uncertainty values [batch_size] (optional)
        uncertainty_weight: Weight for uncertainty in confidence computation
        
    Returns:
        Confidence scores [batch_size]
    """
    if isinstance(probs, torch.Tensor):
        max_probs = torch.max(probs, dim=-1)[0]
        
        if uncertainties is not None:
            # Normalize uncertainties to [0, 1] range
            norm_uncertainties = torch.sigmoid(uncertainties)
            confidence = max_probs * (1 - uncertainty_weight) + (1 - norm_uncertainties) * uncertainty_weight
        else:
            confidence = max_probs
            
        return confidence
    else:
        max_probs = np.max(probs, axis=-1)
        
        if uncertainties is not None:
            # Normalize uncertainties to [0, 1] range using sigmoid
            norm_uncertainties = 1 / (1 + np.exp(-uncertainties))
            confidence = max_probs * (1 - uncertainty_weight) + (1 - norm_uncertainties) * uncertainty_weight
        else:
            confidence = max_probs
            
        return confidence


def calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error (ECE).
    
    Args:
        y_true: True labels [n_samples]
        y_probs: Predicted probabilities [n_samples, n_classes]
        n_bins: Number of bins for calibration
        
    Returns:
        Expected calibration error
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def threat_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """Compute threat-specific metrics for cybersecurity context.
    
    Args:
        y_true: True labels (0=benign, 1=suspicious, 2=malicious)
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        
    Returns:
        Dictionary of threat-specific metrics
    """
    metrics = {}
    
    # Threat detection rate (recall for malicious class)
    if 2 in y_true:
        malicious_mask = (y_true == 2)
        threat_detection_rate = (y_pred[malicious_mask] == 2).mean()
        metrics['threat_detection_rate'] = threat_detection_rate
    
    # False positive rate for benign class
    if 0 in y_true:
        benign_mask = (y_true == 0)
        false_positive_rate = (y_pred[benign_mask] != 0).mean()
        metrics['false_positive_rate'] = false_positive_rate
    
    # Critical miss rate (malicious classified as benign)
    malicious_mask = (y_true == 2)
    if malicious_mask.sum() > 0:
        critical_miss_rate = (y_pred[malicious_mask] == 0).mean()
        metrics['critical_miss_rate'] = critical_miss_rate
    
    # Alert precision (precision for suspicious + malicious)
    alert_true = (y_true >= 1).astype(int)
    alert_pred = (y_pred >= 1).astype(int)
    
    if alert_pred.sum() > 0:
        alert_precision = (alert_true[alert_pred == 1] == 1).mean()
        metrics['alert_precision'] = alert_precision
    
    # Alert recall
    if alert_true.sum() > 0:
        alert_recall = (alert_pred[alert_true == 1] == 1).mean()
        metrics['alert_recall'] = alert_recall
    
    # Severity-weighted accuracy (higher weight for misclassifying malicious)
    weights = np.array([1.0, 2.0, 5.0])  # benign, suspicious, malicious
    correct_weights = weights[y_true] * (y_pred == y_true).astype(float)
    total_weights = weights[y_true]
    severity_weighted_accuracy = correct_weights.sum() / total_weights.sum()
    metrics['severity_weighted_accuracy'] = severity_weighted_accuracy
    
    return metrics
