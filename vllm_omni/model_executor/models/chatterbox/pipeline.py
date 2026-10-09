# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Turbo pipeline topology.

Stage 0: T3     - text and reference conditioning to S3 speech tokens.
Stage 1: S3Gen  - S3 speech tokens to 24 kHz audio.

With ``async_chunk: true`` stage 0 streams token chunks to stage 1 through
the shared-memory connector (CosyVoice3's processor; the codec is the same).
With ``async_chunk: false`` stage 1 receives the finished utterance from
``t3_to_s3gen``.

The key is ``chatterbox_turbo``, not ``chatterbox``: the checkpoint has no
config.json, so the serving factory matches the repo name against the
registered keys, and a bare ``chatterbox`` key would also claim
``ResembleAI/chatterbox``, a different model.
"""

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
            async_chunk_process_next_stage_input_func=f"{_PROC}.cosyvoice3.talker2code2wav_async_chunk",
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
        ),
    ),
)
