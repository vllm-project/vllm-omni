"""
CLI entry point for vLLM-omni that intercepts vLLM commands.
"""

import sys
import argparse
from typing import List, Optional
from vllm_omni.entrypoints.omni import OmniServeCommand


def main():
    """Main CLI entry point that intercepts vLLM commands."""
    # Check if --omni flag is present
    if "--omni" in sys.argv:
        # Remove --omni flag and process with vLLM-omni
        omni_args = [arg for arg in sys.argv[1:] if arg != "--omni"]
        omni_serve = OmniServeCommand()
        omni_serve.run(omni_args)
    else:
        # Forward to original vLLM CLI
        from vllm.entrypoints.cli.main import main as vllm_main
        vllm_main()


if __name__ == "__main__":
    main()
