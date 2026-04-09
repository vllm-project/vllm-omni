from .registry import OmniModelRegistry  # noqa: F401

# Model classes are lazily loaded via OmniModelRegistry.
# Do NOT eagerly import model classes here — it triggers heavy transitive
# imports (CUDA, pynvml, bitsandbytes, etc.) that crash in subprocess
# environments used by vLLM's model inspection.

__all__ = [
    "OmniModelRegistry",
]
