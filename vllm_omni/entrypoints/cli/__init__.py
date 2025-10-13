"""CLI helpers for vLLM-omni entrypoints."""

from .serve import OmniServeCommand
from .main import main

__all__ = ["OmniServeCommand", "main"]
