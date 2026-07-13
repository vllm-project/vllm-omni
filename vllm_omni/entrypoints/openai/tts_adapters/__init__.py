# SPDX-License-Identifier: Apache-2.0
"""Registry of TTS serving adapters.

Adapters register themselves by their ``name`` (the model-type discriminator
returned by ``OmniOpenAIServingSpeech._detect_tts_model_type``) via
``@register_tts_adapter``. Resolution is by that name: detection already encodes
the (order-sensitive, partly ``model_arch``-based) stage logic, so keying the
registry on the resolved type is both simpler and correct by construction.
``stage_keys`` is retained on each adapter as documentation metadata.
"""

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    DiffusionTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    SpeechServingContext,
    TTSModelAdapter,
)

logger = init_logger(__name__)

TTS_ADAPTER_REGISTRY: dict[str, type[TTSModelAdapter]] = {}


def register_tts_adapter(cls: type[TTSModelAdapter]) -> type[TTSModelAdapter]:
    """Class decorator: index ``cls`` under its ``name`` (model-type)."""
    if cls.name in TTS_ADAPTER_REGISTRY:
        raise ValueError(
            f"TTS adapter name {cls.name!r} already registered to "
            f"{TTS_ADAPTER_REGISTRY[cls.name].__qualname__}; {cls.__qualname__} conflicts."
        )
    TTS_ADAPTER_REGISTRY[cls.name] = cls
    return cls


def all_tts_model_types() -> frozenset[str]:
    """All registered model-type names."""
    return frozenset(TTS_ADAPTER_REGISTRY)


def resolve_adapter(model_type: str | None) -> type[TTSModelAdapter] | None:
    """Return the adapter for a detected model-type, or ``None``."""
    if model_type is None:
        return None
    return TTS_ADAPTER_REGISTRY.get(model_type)


# Import adapter modules for their registration side effects. Keep at the bottom
# so the registry helpers above are defined first.
from vllm_omni.entrypoints.openai.tts_adapters import (  # noqa: E402,F401
    cosyvoice3,
    covo_audio,
    fish_speech,
    glm_tts,
    higgs_audio_v2,
    higgs_audio_v3,
    indextts2,
    ming_tts,
    moss_tts,
    omnivoice,
    qwen3_tts,
    step_audio2,
    voxcpm2,
    voxtral,
)

__all__ = [
    "ARTTSAdapter",
    "DiffusionTTSAdapter",
    "OutputPolicy",
    "PreparedRequest",
    "SpeechServingContext",
    "TTSModelAdapter",
    "TTS_ADAPTER_REGISTRY",
    "all_tts_model_types",
    "register_tts_adapter",
    "resolve_adapter",
]
