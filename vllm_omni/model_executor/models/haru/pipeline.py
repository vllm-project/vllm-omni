# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru_vllm/haru_vllm/pipeline.py @ d3612b0.
"""Haru-4B pipeline topology: one autoregressive stage that also produces the audio.

The audio latent of a frame is an input of the next frame, so the MeanFlow head
and the codec run inside the language-model stage, right after sampling. Served
full duplex over ``/v1/realtime?duplex=1``: each conversation is one resumable
engine request (``duplex/plugin.py``).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

HARU_PIPELINE = PipelineConfig(
    model_type="haru",
    model_arch="HaruForConditionalGeneration",
    default_deploy_config_name="haru.yaml",
    duplex_plugin="vllm_omni.model_executor.models.haru.duplex.plugin.HaruDuplexPlugin",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="haru",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            owns_tokenizer=True,
            sampling_constraints={"detokenize": False},
            # Conversation state lives in the worker and cannot be recomputed.
            recompute_preemption="fail",
        ),
    ),
)
