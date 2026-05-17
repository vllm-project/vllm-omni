# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Central declarative registry of all vllm-omni pipelines.

Mirrors the pattern in ``vllm/model_executor/models/registry.py``: each entry
is ``model_type -> (module_path, variable_name)``, and the module is imported
lazily on first lookup (see ``_LazyPipelineRegistry`` in
``vllm_omni/config/stage_config.py``). Keeping every pipeline declared in one
file makes it easy to spot a missing registration, which was the original
motivation in https://github.com/vllm-project/vllm-omni/issues/2887 (item 4).

Per-model ``pipeline.py`` modules still define the ``PipelineConfig`` instance;
they just no longer need to self-register via ``register_pipeline(...)``.

Adding a new pipeline:
    1. Define the ``PipelineConfig`` instance as a module-level variable in
       ``vllm_omni/.../pipeline.py``.
    2. Add one line to ``_OMNI_PIPELINES`` below.

Single-stage diffusion models continue to use the
``_create_default_diffusion_stage_cfg`` fallback in
``async_omni_engine.py`` — they don't need a registry entry. The empty
``_DIFFUSION_PIPELINES`` placeholder previously here (#2915) was removed
once #2987 (which would have populated it) was deferred.

``register_pipeline(config)`` in ``stage_config`` is still supported for
out-of-tree plugins and tests that create pipelines at runtime; those override
the entries declared here.
"""

from __future__ import annotations

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
_OMNI_PIPELINES: dict[str, tuple[str, str]] = {
    # model_type -> (module_path, variable_name)
    "qwen2_5_omni": (
        "vllm_omni.model_executor.models.qwen2_5_omni.pipeline",
        "QWEN2_5_OMNI_PIPELINE",
    ),
    "qwen2_5_omni_thinker_only": (
        "vllm_omni.model_executor.models.qwen2_5_omni.pipeline",
        "QWEN2_5_OMNI_THINKER_ONLY_PIPELINE",
    ),
    "qwen3_omni_moe": (
        "vllm_omni.model_executor.models.qwen3_omni.pipeline",
        "QWEN3_OMNI_PIPELINE",
    ),
    "qwen3_tts": (
        "vllm_omni.model_executor.models.qwen3_tts.pipeline",
        "QWEN3_TTS_PIPELINE",
    ),
    "covo_audio": (
        "vllm_omni.model_executor.models.covo_audio.pipeline",
        "COVO_AUDIO_PIPELINE",
    ),
    "bagel": (
        "vllm_omni.model_executor.models.bagel.pipeline",
        "BAGEL_PIPELINE",
    ),
    "bagel_think": (
        "vllm_omni.model_executor.models.bagel.pipeline",
        "BAGEL_THINK_PIPELINE",
    ),
    "bagel_single_stage": (
        "vllm_omni.model_executor.models.bagel.pipeline",
        "BAGEL_SINGLE_STAGE_PIPELINE",
    ),
    "glm_image": (
        "vllm_omni.model_executor.models.glm_image.pipeline",
        "GLM_IMAGE_PIPELINE",
    ),
    "hunyuan_image_3_moe": (
        "vllm_omni.model_executor.models.hunyuan_image3.pipeline",
        "HUNYUAN_IMAGE3_PIPELINE",
    ),
    "hunyuan_image3_ar": (
        "vllm_omni.model_executor.models.hunyuan_image3.pipeline",
        "HUNYUAN_IMAGE3_AR_PIPELINE",
    ),
    "hunyuan_image3_dit": (
        "vllm_omni.model_executor.models.hunyuan_image3.pipeline",
        "HUNYUAN_IMAGE3_DIT_PIPELINE",
    ),
    "voxcpm2": (
        "vllm_omni.model_executor.models.voxcpm2.pipeline",
        "VOXCPM2_PIPELINE",
    ),
    "cosyvoice3": (
        "vllm_omni.model_executor.models.cosyvoice3.pipeline",
        "COSYVOICE3_PIPELINE",
    ),
    "mimo_audio": (
        "vllm_omni.model_executor.models.mimo_audio.pipeline",
        "MIMO_AUDIO_PIPELINE",
    ),
    "voxtral_tts": (
        "vllm_omni.model_executor.models.voxtral_tts.pipeline",
        "VOXTRAL_TTS_PIPELINE",
    ),
    "fish_qwen3_omni": (
        "vllm_omni.model_executor.models.fish_speech.pipeline",
        "FISH_SPEECH_PIPELINE",
    ),
    "ming_flash_omni": (
        "vllm_omni.model_executor.models.ming_flash_omni.pipeline",
        "MING_FLASH_OMNI_PIPELINE",
    ),
    "ming_flash_omni_tts": (
        "vllm_omni.model_executor.models.ming_flash_omni.pipeline",
        "MING_FLASH_OMNI_TTS_PIPELINE",
    ),
    "ming_flash_omni_thinker_only": (
        "vllm_omni.model_executor.models.ming_flash_omni.pipeline",
        "MING_FLASH_OMNI_THINKER_ONLY_PIPELINE",
    ),
    "moss_tts_nano": (
        "vllm_omni.model_executor.models.moss_tts_nano.pipeline",
        "MOSS_TTS_NANO_PIPELINE",
    ),
}
