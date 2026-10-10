# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Turbo and Original pipeline topologies.

Stage 0: T3     - text and reference conditioning to S3 speech tokens.
Stage 1: S3Gen  - S3 speech tokens to 24 kHz audio.

With ``async_chunk: true`` stage 0 streams token chunks to stage 1 through
the shared-memory connector (``t3_to_s3gen_async_chunk``). With
``async_chunk: false`` stage 1 receives the finished utterance from
``t3_to_s3gen``.

Both stages run on either model runner. On Model Runner V2 they use its
native data plane, where each stage's runner owns the connector.

The config-less checkpoints resolve by repository name. The factory prefers
the longer ``chatterbox_turbo`` key over Original's ``chatterbox`` key.
Original replaces GPT-2 with Llama and schedules guidance branches atomically.
"""

from dataclasses import replace

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

_PROC = "vllm_omni.model_executor.stage_input_processors"

CHATTERBOX_TURBO_PIPELINE = PipelineConfig(
    model_type="chatterbox_turbo",
    default_deploy_config_name="chatterbox_turbo.yaml",
    model_arch="ChatterboxForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="chatterbox_t3",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.chatterbox.t3_to_s3gen_async_chunk",
            supports_native_mrv2_data_plane=True,
            sampling_constraints={
                "stop_token_ids": [ChatterboxConfig().stop_speech_token],
                # Speech ids mean nothing to the text tokenizer.
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="chatterbox_s3gen",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.chatterbox.t3_to_s3gen",
            supports_native_mrv2_data_plane=True,
        ),
    ),
)

CHATTERBOX_ORIGINAL_PIPELINE = replace(
    CHATTERBOX_TURBO_PIPELINE,
    model_type="chatterbox",
    default_deploy_config_name="chatterbox.yaml",
    stages=(
        replace(
            CHATTERBOX_TURBO_PIPELINE.stages[0],
            model_stage="chatterbox_original_t3",
            scheduler_cls="vllm_omni.core.sched.omni_cfg_ar_scheduler.OmniCFGARScheduler",
            prompt_expand_func=f"{_PROC}.chatterbox.expand_original_cfg_prompts",
        ),
        CHATTERBOX_TURBO_PIPELINE.stages[1],
    ),
)
