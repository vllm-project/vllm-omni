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
    2. Add one line to ``_OMNI_PIPELINES`` or ``_DIFFUSION_PIPELINES`` below.

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
}

# --- Single-stage diffusion pipelines (populated in PR 3/N) ---
_DIFFUSION_PIPELINES: dict[str, tuple[str, str]] = {}

# Union view used by ``_LazyPipelineRegistry``; don't mutate at runtime.
_VLLM_OMNI_PIPELINES: dict[str, tuple[str, str]] = {
    **_OMNI_PIPELINES,
    **_DIFFUSION_PIPELINES,
}
