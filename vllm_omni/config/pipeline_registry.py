# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline registry and factory for vllm-omni.

``OMNI_PIPELINES`` maps each ``model_type`` to either a ``PipelineConfig``
instance or a resolver callable that accepts an optional HF config and returns
a ``PipelineConfig``.

To add a new pipeline:
    1. Define the ``PipelineConfig`` instance as a module-level variable in
       ``vllm_omni/.../pipeline.py``.
    2. If the model needs to support several configurations, e.g., because some
       stages are optional, implement a resolver that consumes the HF config
       and returns a ``PipelineConfig``.
    3. Update the registry to map the key to the new config object (in the case
       of new keys) or to the resolver func.

Out of tree pipeline configs or resolvers can also be registered with register_pipeline.

NOTE: Single-stage diffusion models continue to use the
``_create_default_diffusion_stage_cfg`` fallback in
``async_omni_engine.py``; for now we do not add them to registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from transformers import PretrainedConfig
from vllm.logger import init_logger

from vllm_omni.config.stage_config import (
    PipelineConfig,
)
from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_PIPELINE
from vllm_omni.model_executor.models.bagel.pipeline import (
    BAGEL_PIPELINE,
    BAGEL_SINGLE_STAGE_PIPELINE,
    BAGEL_THINK_PIPELINE,
)
from vllm_omni.model_executor.models.cosyvoice3.pipeline import COSYVOICE3_PIPELINE
from vllm_omni.model_executor.models.covo_audio.pipeline import COVO_AUDIO_PIPELINE
from vllm_omni.model_executor.models.dreamzero.pipeline import DREAMZERO_PIPELINE
from vllm_omni.model_executor.models.dynin_omni.pipeline import DYNIN_OMNI_PIPELINE
from vllm_omni.model_executor.models.fish_speech.pipeline import FISH_SPEECH_PIPELINE
from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE
from vllm_omni.model_executor.models.glm_tts.pipeline import GLM_TTS_PIPELINE
from vllm_omni.model_executor.models.gr00t.pipeline import GR00T_N1D7_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import HIGGS_AUDIO_V2_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v3.pipeline import HIGGS_AUDIO_V3_PIPELINE
from vllm_omni.model_executor.models.hunyuan_image3.pipeline import (
    HUNYUAN_IMAGE3_AR_PIPELINE,
    HUNYUAN_IMAGE3_DIT_PIPELINE,
    HUNYUAN_IMAGE3_PIPELINE,
)
from vllm_omni.model_executor.models.hunyuan_video.pipeline import HUNYUAN_VIDEO_15_PIPELINE
from vllm_omni.model_executor.models.indextts2.pipeline import INDEXTTS2_PIPELINE
from vllm_omni.model_executor.models.lance.pipeline import LANCE_PIPELINE
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import (
    MAMMOTH_MODA2_AR_PIPELINE,
    MAMMOTH_MODA2_PIPELINE,
)
from vllm_omni.model_executor.models.mimo_audio.pipeline import MIMO_AUDIO_PIPELINE
from vllm_omni.model_executor.models.ming_flash_omni.pipeline import (
    MING_FLASH_OMNI_IMAGE_PIPELINE,
    MING_FLASH_OMNI_PIPELINE,
    MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    MING_FLASH_OMNI_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.ming_tts.pipeline import (
    MING_TTS_MOE_PIPELINE,
    MING_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.model_executor.models.moss_tts.pipeline import (
    MOSS_TTS_LOCAL_PIPELINE,
    MOSS_TTS_PIPELINE,
    MOSS_TTS_REALTIME_PIPELINE,
)
from vllm_omni.model_executor.models.moss_tts_nano.pipeline import MOSS_TTS_NANO_PIPELINE
from vllm_omni.model_executor.models.omnivoice.pipeline import OMNIVOICE_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import (
    QWEN2_5_OMNI_PIPELINE,
    QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import resolve_qwen3_omni_pipeline
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.step_audio2.pipeline import (
    STEP_AUDIO2_ASR_PIPELINE,
    STEP_AUDIO2_PIPELINE,
)
from vllm_omni.model_executor.models.voxcpm2.pipeline import VOXCPM2_PIPELINE
from vllm_omni.model_executor.models.voxtral_tts.pipeline import VOXTRAL_TTS_PIPELINE
from vllm_omni.model_executor.models.wan2_2.pipeline import WAN2_2_TI2V_PIPELINE

logger = init_logger(__name__)

PipelineResolverFunc: TypeAlias = Callable[[PretrainedConfig | None], PipelineConfig | None]

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc] = {
    "aura_omni": AURA_OMNI_PIPELINE,
    "qwen2_5_omni": QWEN2_5_OMNI_PIPELINE,
    "qwen2_5_omni_thinker_only": QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
    "qwen3_omni_moe": resolve_qwen3_omni_pipeline,
    "qwen3_tts": QWEN3_TTS_PIPELINE,
    "step_audio_2": STEP_AUDIO2_PIPELINE,
    "step_audio_2_asr": STEP_AUDIO2_ASR_PIPELINE,
    "covo_audio": COVO_AUDIO_PIPELINE,
    "bagel": BAGEL_PIPELINE,
    "bagel_think": BAGEL_THINK_PIPELINE,
    "bagel_single_stage": BAGEL_SINGLE_STAGE_PIPELINE,
    "lance": LANCE_PIPELINE,
    "dreamzero": DREAMZERO_PIPELINE,
    "Gr00tN1d7": GR00T_N1D7_PIPELINE,
    "glm_image": GLM_IMAGE_PIPELINE,
    "hunyuan_image_3_moe": HUNYUAN_IMAGE3_PIPELINE,
    "hunyuan_image3_ar": HUNYUAN_IMAGE3_AR_PIPELINE,
    "hunyuan_image3_dit": HUNYUAN_IMAGE3_DIT_PIPELINE,
    "hunyuan_video_15": HUNYUAN_VIDEO_15_PIPELINE,
    "wan2_2_ti2v": WAN2_2_TI2V_PIPELINE,
    "voxcpm2": VOXCPM2_PIPELINE,
    "cosyvoice3": COSYVOICE3_PIPELINE,
    "mimo_audio": MIMO_AUDIO_PIPELINE,
    "ming_tts": MING_TTS_PIPELINE,
    "ming_tts_moe": MING_TTS_MOE_PIPELINE,
    "voxtral_tts": VOXTRAL_TTS_PIPELINE,
    "glm_tts": GLM_TTS_PIPELINE,
    "fish_qwen3_omni": FISH_SPEECH_PIPELINE,
    "ming_flash_omni": MING_FLASH_OMNI_PIPELINE,
    "ming_flash_omni_tts": MING_FLASH_OMNI_TTS_PIPELINE,
    "ming_flash_omni_thinker_only": MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    "ming_flash_omni_image": MING_FLASH_OMNI_IMAGE_PIPELINE,
    "moss_tts_nano": MOSS_TTS_NANO_PIPELINE,
    "omnivoice": OMNIVOICE_PIPELINE,
    "mammoth_moda2": MAMMOTH_MODA2_PIPELINE,
    "mammoth_moda2_ar": MAMMOTH_MODA2_AR_PIPELINE,
    "moss_tts_delay": MOSS_TTS_PIPELINE,
    "moss_tts_realtime": MOSS_TTS_REALTIME_PIPELINE,
    "moss_tts_local": MOSS_TTS_LOCAL_PIPELINE,
    "minicpmo_4_5": MINICPMO_4_5_PIPELINE,
    "higgs_audio_v2": HIGGS_AUDIO_V2_PIPELINE,
    "higgs_multimodal_qwen3": HIGGS_AUDIO_V3_PIPELINE,
    "dynin_omni": DYNIN_OMNI_PIPELINE,
    "indextts2": INDEXTTS2_PIPELINE,
}


def register_pipeline(pipeline: PipelineConfig | PipelineResolverFunc, model_type: str | None = None):
    """Register an out of tree pipeline or PipelineResolverFunc to a model_type key.
    If a PipelineConfig is provided, model_type is optional, and pipeline.model_type
    will be used by default. If a callable is provided, model_type must be provided,
    since resolvers can return multiple different PipelineConfigs depending on the
    consumed config.
    """
    errors: list[str] = []
    if isinstance(pipeline, PipelineConfig):
        errors = pipeline.validate()
        model_type = model_type if model_type is not None else pipeline.model_type
    else:
        if model_type is None:
            raise ValueError("Model type must be explicitly provided when registering a pipeline resolver")

    if model_type in OMNI_PIPELINES:
        errors.append(f"Model type {model_type} is already registered; the old mapping will be clobbered")
    if errors:
        logger.warning("Registration for pipeline of type %s produced the following issues: %s", model_type, errors)
    OMNI_PIPELINES[model_type] = pipeline


def resolve_pipeline_config(
    model_type: str,
    hf_config: PretrainedConfig | None = None,
) -> PipelineConfig | None:
    """Resolve a registry key to a concrete pipeline config."""
    if model_type not in OMNI_PIPELINES:
        logger.warning("Model type %s is not registered to OMNI_PIPELINES", model_type)
        return None
    pipeline = OMNI_PIPELINES[model_type]
    return pipeline(hf_config) if callable(pipeline) else pipeline
