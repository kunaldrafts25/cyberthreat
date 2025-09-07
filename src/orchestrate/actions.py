"""Orchestration module: execute threat response actions and maintain audit logs."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from loguru import logger

from ..common.utils import ensure_dir


class ActionOrchestrator:
    """Coordinator to execute threat response actions and log audits."""

    def __init__(self, audit_log_dir: Optional[str] = None):
        """Initialize action orchestrator.
        
        Args:
            audit_log_dir: Directory for audit logs (defaults to reports/actions)
        """
        self.audit_log_dir = Path(audit_log_dir) if audit_log_dir else Path("reports/actions")
        ensure_dir(self.audit_log_dir)

    def execute_action(self, sample_id: int, action: int, reason: str, metadata: Optional[Dict] = None) -> None:
        """Execute the specified action on a threat sample and log it.

        Args:
            sample_id: Unique identifier of the threat sample
            action: The action to perform (0=monitor, 1=isolate, 2=block)
            reason: Explanation or reason for the action
            metadata: Optional additional data
        """
        action_map = {0: "monitor", 1: "isolate", 2: "block"}
        action_str = action_map.get(action, "unknown")

        audit_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sample_id": sample_id,
            "action": action_str,
            "reason": reason,
            "metadata": metadata or {}
        }

        audit_file = self.audit_log_dir / f"action_{sample_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        try:
            with open(audit_file, "w") as f:
                json.dump(audit_record, f, indent=2)
            logger.info(f"Action executed and logged: Sample {sample_id}, Action {action_str}")
        except Exception as e:
            logger.error(f"Failed to log action for sample {sample_id}: {e}")

    def batch_execute(self, actions: List[Dict]) -> None:
        """Execute multiple actions in batch.
        
        Args:
            actions: List of action dictionaries with keys: sample_id, action, reason, metadata
        """
        for action_dict in actions:
            self.execute_action(
                action_dict['sample_id'],
                action_dict['action'], 
                action_dict['reason'],
                action_dict.get('metadata')
            )

    def get_audit_history(self, sample_id: Optional[int] = None) -> List[Dict]:
        """Retrieve audit history for a sample or all samples.
        
        Args:
            sample_id: Specific sample ID (None for all)
            
        Returns:
            List of audit records
        """
        audit_files = []
        if sample_id:
            pattern = f"action_{sample_id}_*.json"
        else:
            pattern = "action_*.json"
        
        for audit_file in self.audit_log_dir.glob(pattern):
            try:
                with open(audit_file, 'r') as f:
                    audit_record = json.load(f)
                    audit_files.append(audit_record)
            except Exception as e:
                logger.warning(f"Failed to read audit file {audit_file}: {e}")
        
        return sorted(audit_files, key=lambda x: x['timestamp'])


class DummyOrchestrator(ActionOrchestrator):
    """Dummy orchestrator for testing, logs simulated actions without real effect."""

    def execute_action(self, sample_id: int, action: int, reason: str, metadata: Optional[Dict] = None) -> None:
        """Simulate action execution without real system changes."""
        action_map = {0: "monitor", 1: "isolate", 2: "block"}
        action_str = action_map.get(action, "unknown")
        
        logger.info(f"[Dummy] Simulating action '{action_str}' on sample {sample_id} with reason: {reason}")
        
        # Still log to audit trail for testing
        super().execute_action(sample_id, action, f"[DUMMY] {reason}", metadata)


class NetworkActionOrchestrator(ActionOrchestrator):
    """Network-specific orchestrator that interfaces with actual network security systems."""
    
    def __init__(self, audit_log_dir: Optional[str] = None, network_api_config: Optional[Dict] = None):
        """Initialize network action orchestrator.
        
        Args:
            audit_log_dir: Directory for audit logs
            network_api_config: Configuration for network security API endpoints
        """
        super().__init__(audit_log_dir)
        self.network_config = network_api_config or {}
    
    def execute_action(self, sample_id: int, action: int, reason: str, metadata: Optional[Dict] = None) -> None:
        """Execute network security action and log it."""
        action_map = {0: "monitor", 1: "isolate", 2: "block"}
        action_str = action_map.get(action, "unknown")
        
        # Execute actual network action
        success = self._execute_network_action(action, metadata)
        
        # Update metadata with execution result
        execution_metadata = (metadata or {}).copy()
        execution_metadata['execution_success'] = success
        execution_metadata['network_action'] = True
        
        # Log the action
        super().execute_action(sample_id, action, reason, execution_metadata)
    
    def _execute_network_action(self, action: int, metadata: Optional[Dict]) -> bool:
        """Execute the actual network security action.
        
        Args:
            action: Action code
            metadata: Action metadata containing network details
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if action == 1:  # Isolate
                return self._isolate_host(metadata)
            elif action == 2:  # Block
                return self._block_traffic(metadata)
            else:  # Monitor (action == 0)
                return self._enhance_monitoring(metadata)
        except Exception as e:
            logger.error(f"Network action execution failed: {e}")
            return False
    
    def _isolate_host(self, metadata: Optional[Dict]) -> bool:
        """Isolate a host on the network."""
        if not metadata or 'source_ip' not in metadata:
            logger.warning("Cannot isolate: missing source IP")
            return False
        
        source_ip = metadata['source_ip']
        logger.info(f"Isolating host {source_ip}")
        # In real implementation, this would call network security APIs
        return True
    
    def _block_traffic(self, metadata: Optional[Dict]) -> bool:
        """Block traffic from/to specific IPs or ports."""
        if not metadata:
            logger.warning("Cannot block: missing traffic details")
            return False
        
        source_ip = metadata.get('source_ip')
        dest_ip = metadata.get('dest_ip')
        port = metadata.get('port')
        
        logger.info(f"Blocking traffic: {source_ip} -> {dest_ip}:{port}")
        # In real implementation, this would update firewall rules
        return True
    
    def _enhance_monitoring(self, metadata: Optional[Dict]) -> bool:
        """Enhance monitoring for specific traffic patterns."""
        logger.info("Enhancing monitoring for detected threat pattern")
        # In real implementation, this would configure SIEM/monitoring tools
        return True


def create_orchestrator(config, orchestrator_type: str = "default") -> ActionOrchestrator:
    """Factory to create an orchestrator instance based on configuration.
    
    Args:
        config: System configuration
        orchestrator_type: Type of orchestrator ("default", "dummy", "network")
        
    Returns:
        Configured ActionOrchestrator instance
    """
    audit_dir = config.get("paths.audit_log_dir", "reports/actions")
    
    if orchestrator_type == "dummy":
        return DummyOrchestrator(audit_dir)
    elif orchestrator_type == "network":
        network_config = config.get("network_security", {})
        return NetworkActionOrchestrator(audit_dir, network_config)
    else:
        return ActionOrchestrator(audit_dir)
