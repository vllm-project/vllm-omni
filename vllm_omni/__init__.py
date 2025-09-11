"""
vLLM-omni: Multi-modality models inference and serving with non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive generation
to support multi-modality models with non-autoregressive structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support  
- 🔴 Added: New components for multimodal and non-autoregressive processing
"""

__version__ = "0.1.0"
__author__ = "vLLM-omni Team"
__email__ = "hsliuustc@gmail.com"

# Main entry points
from .async_llm import AsyncLLM
from .omni_engine import OmniEngine
from .diffusion_engine import DiffusionEngine

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Main components
    "AsyncLLM",
    "OmniEngine",
    "DiffusionEngine",
    
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]