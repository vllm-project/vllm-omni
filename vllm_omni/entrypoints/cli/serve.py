"""
Omni serve command for vLLM-omni.
"""

import argparse
from typing import List


class OmniServeCommand:
    """Command handler for vLLM-omni serve command."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for omni serve command."""
        parser = argparse.ArgumentParser(
            description="vLLM-omni: Multi-modality models inference and serving"
        )

        # Model arguments - make it optional with default
        parser.add_argument(
            "model",
            nargs="?",
            default="Qwen/Qwen3-0.6B",
            help="Path to the model or model name (default: Qwen/Qwen3-0.6B)",
        )

        # Server arguments
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to run the server on",
        )

        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to run the server on",
        )

        # Stage configuration arguments
        parser.add_argument("--ar-stage", type=str, help="AR stage model path")

        parser.add_argument("--dit-stage", type=str, help="DiT stage model path")

        parser.add_argument(
            "--dit-steps",
            type=int,
            default=50,
            help="Number of DiT inference steps",
        )

        parser.add_argument(
            "--dit-guidance-scale",
            type=float,
            default=7.5,
            help="DiT guidance scale",
        )

        parser.add_argument(
            "--use-diffusers",
            action="store_true",
            help="Use diffusers pipeline for DiT stage",
        )

        # Other arguments
        parser.add_argument(
            "--log-stats", action="store_true", help="Enable logging statistics"
        )

        return parser

    def run(self, args: List[str]) -> None:
        """Run the omni serve command."""
        pass

    def _create_stage_configs(self, args) -> List:
        """Create stage configurations based on arguments."""
        pass

    async def _start_server(self, omni_llm, args) -> None:
        """Start the API server."""
        try:
            # Import here to avoid circular imports
            from vllm_omni.entrypoints.api_server import run_server

            await run_server(omni_llm_instance=omni_llm, host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"Error starting server: {e}")
            raise
