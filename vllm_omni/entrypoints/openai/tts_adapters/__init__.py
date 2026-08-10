# SPDX-License-Identifier: Apache-2.0
"""Registry of TTS serving adapters, and the model detection built on it.

Adapters register themselves by their ``name`` (the model-type discriminator)
via ``@register_tts_adapter``, and declare the deployment metadata that
identifies them: ``stage_keys``, plus ``model_archs`` for the few models a stage
key alone cannot distinguish.

That metadata is the single source of truth. :func:`detect_tts_model_type` and
:func:`all_tts_stage_keys` derive the whole stage -> model-type mapping from it,
so a new TTS model is one adapter file and needs no edit to ``serving_speech.py``.
Models still awaiting an adapter are listed explicitly in
:data:`LEGACY_TTS_DETECTORS`.
"""

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    DiffusionTTSAdapter,
    LegacyDetector,
    OutputPolicy,
    PreparedRequest,
    SpeechServingContext,
    TTSModelAdapter,
)

logger = init_logger(__name__)

TTS_ADAPTER_REGISTRY: dict[str, type[TTSModelAdapter]] = {}

#: Model types the serving layer detects that have no adapter yet. See
#: :class:`LegacyDetector`. This list must only ever shrink.
LEGACY_TTS_DETECTORS: tuple[LegacyDetector, ...] = (
    # Ming *flash-omni* claims the ``ming_tts`` stage key, while Ming *dense*
    # (which does have an adapter) is identified by architecture. The stage key
    # must therefore be tested first, hence the explicit priority.
    LegacyDetector(
        name="ming_flash_omni_tts",
        stage_keys=frozenset({"ming_tts"}),
        detect_priority=50,
    ),
)


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


def iter_tts_detectors() -> list[type[TTSModelAdapter] | LegacyDetector]:
    """Every detector, in resolution order.

    Sorted by ``detect_priority`` then ``name``, so the order is total and does
    not depend on adapter import order.
    """
    detectors: list[type[TTSModelAdapter] | LegacyDetector] = [
        *TTS_ADAPTER_REGISTRY.values(),
        *LEGACY_TTS_DETECTORS,
    ]
    detectors.sort(key=lambda d: (d.detect_priority, d.name))
    return detectors


def detect_tts_model_type(model_stage: str | None, model_arch: str | None) -> str | None:
    """Resolve a deployed stage to its TTS model-type, or ``None``.

    The first detector that matches wins; ties are impossible for distinct model
    types because overlapping detectors must declare an explicit
    ``detect_priority`` (enforced by ``test_tts_detection.py``).
    """
    for detector in iter_tts_detectors():
        if detector.matches(model_stage, model_arch):
            return detector.name
    return None


def all_tts_stage_keys() -> frozenset[str]:
    """Every ``model_stage`` value served by ``/v1/audio/speech``."""
    keys: set[str] = set()
    for detector in iter_tts_detectors():
        keys |= detector.stage_keys
    return frozenset(keys)


def tts_entry_stage_archs() -> frozenset[str]:
    """``model_arch`` values that identify an AR entry stage on their own.

    Only for models that own no ``model_stage`` key; stage discovery falls back
    to these. Deliberately *not* every declared ``model_arch``: a model that also
    owns a stage key is found by that key, and widening this would change which
    stage is selected in mixed deployments.
    """
    archs: set[str] = set()
    for detector in iter_tts_detectors():
        if detector.arch_identifies_entry_stage:
            archs |= detector.model_archs
    return frozenset(archs)


# Import adapter modules for their registration side effects. Keep at the bottom
# so the registry helpers above are defined first.
from vllm_omni.entrypoints.openai.tts_adapters import (  # noqa: E402,F401
    audex,
    audex_tta,
    cosyvoice3,
    covo_audio,
    fish_speech,
    glm_tts,
    higgs_audio_v2,
    higgs_audio_v3,
    indextts2,
    ming_flash_omni_tts,
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
    "LegacyDetector",
    "OutputPolicy",
    "PreparedRequest",
    "SpeechServingContext",
    "TTSModelAdapter",
    "LEGACY_TTS_DETECTORS",
    "TTS_ADAPTER_REGISTRY",
    "all_tts_model_types",
    "all_tts_stage_keys",
    "detect_tts_model_type",
    "iter_tts_detectors",
    "register_tts_adapter",
    "resolve_adapter",
    "tts_entry_stage_archs",
]
