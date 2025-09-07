"""Comprehensive threat report generator for explainable AI outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from loguru import logger

from ..common.utils import ensure_dir


class ThreatReportGenerator:
    """Generator for comprehensive threat analysis reports."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports (defaults to reports/explanations)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("reports/explanations")
        ensure_dir(self.output_dir)
        
    def generate_comprehensive_report(self, 
                                    sample_id: Union[int, str],
                                    threat_analysis: Dict,
                                    neural_output: Dict,
                                    rule_analysis: Dict,
                                    kg_analysis: Dict,
                                    attribution_analysis: Optional[Dict] = None) -> Path:
        """
        Generate comprehensive threat detection report.
        
        Args:
            sample_id: Unique sample identifier
            threat_analysis: Main threat detection results
            neural_output: Neural network predictions and confidence
            rule_analysis: Rule-based reasoning results
            kg_analysis: Knowledge graph analysis results
            attribution_analysis: Model attribution analysis
            
        Returns:
            Path to generated report file
        """
        report = {
            "report_metadata": {
                "sample_id": str(sample_id),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "report_version": "1.0",
                "analysis_type": "comprehensive_threat_detection"
            },
            "executive_summary": self._generate_executive_summary(threat_analysis),
            "threat_detection_results": threat_analysis,
            "neural_analysis": neural_output,
            "symbolic_reasoning": {
                "rule_based_analysis": rule_analysis,
                "knowledge_graph_analysis": kg_analysis
            },
            "explainability": attribution_analysis or {},
            "recommendations": self._generate_recommendations(threat_analysis, neural_output),
            "technical_details": self._generate_technical_details(neural_output, rule_analysis)
        }
        
        # Add confidence assessment
        report["confidence_assessment"] = self._assess_confidence(
            neural_output, rule_analysis, kg_analysis
        )
        
        # Save report
        filename = self.output_dir / f"threat_report_{sample_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Comprehensive threat report generated for sample {sample_id}: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to generate report for sample {sample_id}: {e}")
            raise
        
        return filename
    
    def generate_simple_explanation(self, 
                                   sample_id: Union[int, str],
                                   explanation: Dict) -> Path:
        """
        Generate simple explanation report.
        
        Args:
            sample_id: Sample identifier
            explanation: Explanation dictionary
            
        Returns:
            Path to generated report
        """
        simple_report = {
            "sample_id": str(sample_id),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "explanation": explanation
        }
        
        filename = self.output_dir / f"explanation_{sample_id}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(simple_report, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Simple explanation report generated for sample {sample_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate simple report for sample {sample_id}: {e}")
            raise
        
        return filename
    
    def generate_batch_report(self, 
                            batch_analyses: List[Dict],
                            report_name: str = "batch_analysis") -> Path:
        """
        Generate batch analysis report.
        
        Args:
            batch_analyses: List of analysis results
            report_name: Name for the batch report
            
        Returns:
            Path to generated batch report
        """
        batch_report = {
            "report_metadata": {
                "report_name": report_name,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "batch_size": len(batch_analyses),
                "report_type": "batch_threat_analysis"
            },
            "summary_statistics": self._compute_batch_statistics(batch_analyses),
            "individual_analyses": batch_analyses,
            "patterns_identified": self._identify_threat_patterns(batch_analyses)
        }
        
        filename = self.output_dir / f"batch_report_{report_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(batch_report, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Batch report generated: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to generate batch report: {e}")
            raise
        
        return filename
    
    def _generate_executive_summary(self, threat_analysis: Dict) -> Dict:
        """Generate executive summary of threat analysis."""
        threat_level = threat_analysis.get('threat_level', 'unknown')
        confidence = threat_analysis.get('confidence', 0.0)
        
        risk_level = "HIGH" if threat_level == "malicious" else \
                    "MEDIUM" if threat_level == "suspicious" else "LOW"
        
        summary = {
            "threat_classification": threat_level.upper(),
            "risk_level": risk_level,
            "confidence_score": round(confidence, 3),
            "requires_immediate_attention": threat_level == "malicious" and confidence > 0.8,
            "key_indicators": threat_analysis.get('key_indicators', []),
            "primary_concern": self._determine_primary_concern(threat_analysis)
        }
        
        return summary
    
    def _generate_recommendations(self, threat_analysis: Dict, neural_output: Dict) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        threat_level = threat_analysis.get('threat_level', 'benign')
        confidence = threat_analysis.get('confidence', 0.0)
        uncertainty = neural_output.get('uncertainty', 0.0)
        
        if threat_level == "malicious":
            if confidence > 0.9:
                recommendations.append({
                    "priority": "CRITICAL",
                    "action": "IMMEDIATE_ISOLATION",
                    "description": "Isolate affected systems immediately to prevent lateral movement",
                    "timeline": "Within 5 minutes"
                })
            else:
                recommendations.append({
                    "priority": "HIGH", 
                    "action": "ENHANCED_MONITORING",
                    "description": "Increase monitoring and gather additional forensic evidence",
                    "timeline": "Within 15 minutes"
                })
        
        elif threat_level == "suspicious":
            recommendations.append({
                "priority": "MEDIUM",
                "action": "INVESTIGATE",
                "description": "Conduct detailed investigation and threat hunting",
                "timeline": "Within 1 hour"
            })
        
        if uncertainty > 0.5:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "HUMAN_REVIEW",
                "description": "High uncertainty detected - requires expert human analysis",
                "timeline": "Within 2 hours"
            })
        
        return recommendations
    
    def _generate_technical_details(self, neural_output: Dict, rule_analysis: Dict) -> Dict:
        """Generate technical analysis details."""
        return {
            "neural_network_details": {
                "model_confidence": neural_output.get('confidence', 0.0),
                "prediction_uncertainty": neural_output.get('uncertainty', 0.0),
                "class_probabilities": neural_output.get('probabilities', []),
                "model_consensus": neural_output.get('consensus_score', 0.0)
            },
            "rule_engine_details": {
                "rules_fired": rule_analysis.get('fired_rules', []),
                "rule_confidence": rule_analysis.get('confidence', 0.0),
                "rule_explanations": rule_analysis.get('explanations', [])
            },
            "feature_analysis": neural_output.get('feature_importance', {}),
            "anomaly_scores": neural_output.get('anomaly_scores', {})
        }
    
    def _assess_confidence(self, neural_output: Dict, rule_analysis: Dict, kg_analysis: Dict) -> Dict:
        """Assess overall confidence in the analysis."""
        neural_conf = neural_output.get('confidence', 0.0)
        rule_conf = rule_analysis.get('confidence', 0.0)
        consensus = neural_output.get('consensus_score', 0.0)
        
        overall_confidence = (neural_conf * 0.5 + rule_conf * 0.3 + consensus * 0.2)
        
        confidence_level = "HIGH" if overall_confidence > 0.8 else \
                          "MEDIUM" if overall_confidence > 0.6 else "LOW"
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "confidence_level": confidence_level,
            "contributing_factors": {
                "neural_network_confidence": neural_conf,
                "rule_based_confidence": rule_conf,
                "multi_modal_consensus": consensus
            },
            "reliability_indicators": {
                "model_agreement": consensus > 0.7,
                "low_uncertainty": neural_output.get('uncertainty', 1.0) < 0.3,
                "strong_rule_support": len(rule_analysis.get('fired_rules', [])) > 2
            }
        }
    
    def _determine_primary_concern(self, threat_analysis: Dict) -> str:
        """Determine the primary security concern."""
        threat_level = threat_analysis.get('threat_level', 'benign')
        
        if threat_level == "malicious":
            return "Active threat detected - immediate response required"
        elif threat_level == "suspicious":
            return "Potentially malicious activity - investigation recommended"
        else:
            return "No significant threats detected"
    
    def _compute_batch_statistics(self, batch_analyses: List[Dict]) -> Dict:
        """Compute statistics for batch analysis."""
        if not batch_analyses:
            return {}
        
        threat_counts = {"benign": 0, "suspicious": 0, "malicious": 0}
        confidences = []
        
        for analysis in batch_analyses:
            threat_level = analysis.get('threat_level', 'benign')
            threat_counts[threat_level] += 1
            confidences.append(analysis.get('confidence', 0.0))
        
        return {
            "total_samples": len(batch_analyses),
            "threat_distribution": threat_counts,
            "threat_percentages": {
                level: round(count / len(batch_analyses) * 100, 2)
                for level, count in threat_counts.items()
            },
            "average_confidence": round(np.mean(confidences), 3),
            "confidence_std": round(np.std(confidences), 3),
            "high_confidence_predictions": sum(1 for c in confidences if c > 0.8),
            "low_confidence_predictions": sum(1 for c in confidences if c < 0.5)
        }
    
    def _identify_threat_patterns(self, batch_analyses: List[Dict]) -> List[Dict]:
        """Identify common threat patterns in batch."""
        patterns = []
        
        # Common rule patterns
        all_rules = []
        for analysis in batch_analyses:
            rule_analysis = analysis.get('rule_analysis', {})
            all_rules.extend(rule_analysis.get('fired_rules', []))
        
        if all_rules:
            from collections import Counter
            rule_counts = Counter(all_rules)
            common_rules = rule_counts.most_common(5)
            
            patterns.append({
                "pattern_type": "common_rules",
                "description": "Most frequently triggered security rules",
                "details": [{"rule": rule, "frequency": count} for rule, count in common_rules]
            })
        
        # High-confidence malicious samples
        malicious_samples = [
            analysis for analysis in batch_analyses 
            if analysis.get('threat_level') == 'malicious' and analysis.get('confidence', 0) > 0.8
        ]
        
        if malicious_samples:
            patterns.append({
                "pattern_type": "high_confidence_threats",
                "description": f"Detected {len(malicious_samples)} high-confidence malicious samples",
                "details": {
                    "count": len(malicious_samples),
                    "average_confidence": round(np.mean([s.get('confidence', 0) for s in malicious_samples]), 3)
                }
            })
        
        return patterns
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and torch objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def generate_html_report(self, 
                           json_report_path: Path,
                           template_path: Optional[str] = None) -> Path:
        """
        Generate HTML version of the threat report.
        
        Args:
            json_report_path: Path to JSON report
            template_path: Path to HTML template
            
        Returns:
            Path to generated HTML report
        """
        # Load JSON report
        with open(json_report_path, 'r') as f:
            report_data = json.load(f)
        
        # Simple HTML generation (in production, use proper templating)
        html_content = self._generate_html_content(report_data)
        
        html_path = json_report_path.with_suffix('.html')
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {html_path}")
        
        return html_path
    
    def _generate_html_content(self, report_data: Dict) -> str:
        """Generate HTML content from report data."""
        # Simplified HTML generation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Threat Analysis Report - {report_data.get('report_metadata', {}).get('sample_id', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 3px solid #007cba; }}
                .threat-high {{ border-left-color: #d73527; }}
                .threat-medium {{ border-left-color: #f57c00; }}
                .threat-low {{ border-left-color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cyber Threat Analysis Report</h1>
                <p><strong>Sample ID:</strong> {report_data.get('report_metadata', {}).get('sample_id', 'N/A')}</p>
                <p><strong>Generated:</strong> {report_data.get('report_metadata', {}).get('timestamp', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Threat Level:</strong> {report_data.get('executive_summary', {}).get('threat_classification', 'Unknown')}</p>
                <p><strong>Risk Level:</strong> {report_data.get('executive_summary', {}).get('risk_level', 'Unknown')}</p>
                <p><strong>Confidence:</strong> {report_data.get('executive_summary', {}).get('confidence_score', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in report_data.get('recommendations', []):
            html += f"<li><strong>{rec.get('priority', 'N/A')}:</strong> {rec.get('description', 'N/A')}</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html


def create_threat_report_generator(output_dir: Optional[str] = None) -> ThreatReportGenerator:
    """
    Factory function to create threat report generator.
    
    Args:
        output_dir: Output directory for reports
        
    Returns:
        Configured ThreatReportGenerator instance
    """
    return ThreatReportGenerator(output_dir)
