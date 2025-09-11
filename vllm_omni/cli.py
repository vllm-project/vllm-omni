"""
Command-line interface for vLLM-omni.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from .async_llm import AsyncLLM
from .omni_engine import OmniEngine
from .configs import load_config


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )


async def run_server(
    config_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "INFO"
) -> None:
    """Run the vLLM-omni server."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(config_path) if config_path else {}
    
    # Initialize components
    async_llm = AsyncLLM(config)
    omni_engine = OmniEngine(config)
    
    logger.info(f"Starting vLLM-omni server on {host}:{port}")
    
    try:
        # Start the server (implementation depends on chosen framework)
        # This is a placeholder for the actual server implementation
        logger.info("Server started successfully")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


def run_gradio(
    config_path: Optional[str] = None,
    port: int = 7860,
    share: bool = False,
    log_level: str = "INFO"
) -> None:
    """Run the Gradio interface."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        import gradio as gr
        from .api_interfaces.gradio import GradioInterface
        
        # Load configuration
        config = load_config(config_path) if config_path else {}
        
        # Initialize Gradio interface
        interface = GradioInterface(config)
        
        logger.info(f"Starting Gradio interface on port {port}")
        interface.launch(server_port=port, share=share)
        
    except ImportError:
        logger.error("Gradio is not installed. Install it with: pip install gradio")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Gradio interface error: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="vLLM-omni: Multi-modality models inference and serving"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the server")
    server_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    server_parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    # Gradio command
    gradio_parser = subparsers.add_parser("gradio", help="Run the Gradio interface")
    gradio_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    gradio_parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)"
    )
    gradio_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    gradio_parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    if args.command == "server":
        asyncio.run(run_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
            log_level=args.log_level
        ))
    elif args.command == "gradio":
        run_gradio(
            config_path=args.config,
            port=args.port,
            share=args.share,
            log_level=args.log_level
        )
    elif args.command == "version":
        from . import __version__
        print(f"vLLM-omni version {__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
