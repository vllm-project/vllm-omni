# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo single-stage AR pipeline.

Mirrors sglang's design: ONE model class (``Alpamayo15ForConditionalGeneration``)
contains the vLLM Qwen3-VL + flow-matching expert + action heads, so the engine
runs a single LLM-AR stage that internally dispatches to flow matching when the
trajectory trigger token appears. No two-stage / no diffusion pipeline.

Reference template: ``QWEN2_5_OMNI_THINKER_ONLY_PIPELINE``.
"""

from __future__ import annotations

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ALPAMAYO_PIPELINE = PipelineConfig(
    model_type="alpamayo1_5",
    model_arch="Alpamayo15ForConditionalGeneration",
    hf_architectures=("Alpamayo1_5",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            # "latent" (not "text"): signals the worker to route the model's
            # multimodal_outputs payload (sampled action trajectory tensor) to
            # the engine output processor so the client sees it under
            # multimodal_output["actions"]. The AR text path is independent of
            # this field and continues to flow through the sampler.
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

__all__ = ["ALPAMAYO_PIPELINE"]
