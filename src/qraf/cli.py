"""QRAF CLI for interacting with quantum-enhanced Claude instances."""

import argparse
import logging
import os
from typing import Optional

from .core.claude_network import ClaudeNetwork
from .interfaces.claude_v3_5 import ClaudeV3_5Augmenter

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def setup_claude_network() -> ClaudeNetwork:
    """Set up the Claude network with specialized instances."""
    return ClaudeNetwork(
        hidden_size=768,
        num_instances=3,
        coherence_threshold=0.7,
        phase_preservation=0.5
    )


def run(
    query: Optional[str] = None,
    model: str = "sonnet",
    stream: bool = False,
    api_key: Optional[str] = None,
    context: Optional[str] = None,
    debug: bool = False
) -> None:
    """Run the QRAF CLI."""
    if debug:
        logger.debug("Starting QRAF CLI")
        logger.debug(f"Arguments: {locals()}")
        
    # Get API key
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key is None:
        raise ValueError("No API key provided")
    if debug:
        logger.debug("API key found")
        
    # Select model
    model_name = f"claude-3-5-{model}-20241022"
    if debug:
        logger.debug(f"Selected model: {model_name}")
        
    # Initialize network
    if debug:
        logger.debug("Initializing network")
    network = setup_claude_network()
    if debug:
        logger.debug("Network initialized successfully")
        
    # Process query
    if debug:
        logger.debug("Generating response")
        logger.debug("Using non-streaming mode" if not stream else "Using streaming mode")
        
    response, metrics = network.process_query(query)
    
    # Update network metrics
    network.update_network_metrics()
    network.synchronize_phases()
    
    if debug:
        logger.debug("Response received")
        
    # Print response and metrics
    print(response)
    print("\nQuantum Enhancement Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        
    if debug:
        logger.debug("Response generation completed")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="QRAF CLI")
    parser.add_argument("query", help="Query to process")
    parser.add_argument("--model", default="sonnet", help="Model to use")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--context", help="Context for query")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main() 