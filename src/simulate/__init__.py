"""Simulation module - gated by configuration to prevent accidental synthetic data generation."""

from ..common.config import get_config
from loguru import logger

def check_simulation_enabled():
    """Check if simulation is enabled in configuration."""
    config = get_config()
    enabled = config.get("simulation.enabled", False)
    
    if not enabled:
        logger.warning("Simulation module accessed but disabled in configuration")
        raise ValueError("Simulation is disabled. Enable in configs/default.yaml if needed for research.")
    
    return enabled

# Only load simulation components if explicitly enabled
if get_config().get("simulation.enabled", False):
    logger.info("Simulation module loaded (enabled in config)")
else:
    logger.info("Simulation module disabled by configuration")
