"""
vLLM-omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

from . import patch  # noqa: F401
from .config import OmniModelConfig

# Main entry points
from .entrypoints.omni_llm import OmniLLM

__all__ = [
    # Main components
    "OmniLLM",
    # "AsyncOmniLLM", # TODO: add AsyncOmniLLM back when it is implemented
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
