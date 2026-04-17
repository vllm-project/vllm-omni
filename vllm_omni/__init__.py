"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: F401

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402

from .config import OmniModelConfig


def __getattr__(name: str):
    # Lazy import for AsyncOmni and Omni to avoid pulling in heavy
    # dependencies (vllm model_loader → fused_moe → pynvml) at package
    # import time.  This prevents crashes in lightweight subprocesses
    # (e.g. model-architecture inspection) that lack a CUDA context.
    # See: https://github.com/vllm-project/vllm-omni/issues/1793
    if name == "AsyncOmni":
        from .entrypoints.async_omni import AsyncOmni

        return AsyncOmni
    if name == "Omni":
        from .entrypoints.omni import Omni

        return Omni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
