"""Main serving script for the cyber threat AI API."""

import argparse

from src.common.config import get_config
from src.common.logging import setup_logging
from src.serve.api import run_server


def main():
    """Main serving function."""
    parser = argparse.ArgumentParser(description="Serve cyber threat AI API")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--host", type=str, default=None,
                       help="Server host")
    parser.add_argument("--port", type=int, default=None,
                       help="Server port")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Setup logging
    setup_logging(args.config)
    
    # Get server settings
    host = args.host or config.get("serve.host", "0.0.0.0")
    port = args.port or config.get("serve.port", 8080)
    
    print(f"Starting Cyber Threat AI API server on {host}:{port}")
    
    # Run server
    run_server(host=host, port=port)


if __name__ == "__main__":
    main()
