from importlib import import_module

from .registry import OmniModelRegistry  # noqa: F401

# Model classes are lazily loaded via OmniModelRegistry.
# Do NOT eagerly import model classes here — it triggers heavy transitive
# imports (CUDA, pynvml, bitsandbytes, etc.) that crash in subprocess
# environments used by vLLM's model inspection.


def __getattr__(name: str):
    if name == "FunAudioChatForConditionalGeneration":
        module = import_module("vllm_omni.model_executor.models.funaudiochat")
        return module.FunAudioChatForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OmniModelRegistry",
    "FunAudioChatForConditionalGeneration",
]
