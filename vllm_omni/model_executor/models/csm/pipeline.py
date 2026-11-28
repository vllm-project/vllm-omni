# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sesame CSM pipeline: talker (text to Mimi codes) then code2wav (codes to 24 kHz audio).

The codec stage keeps its decoder state per request, so the stages are always
connected by chunks; there is no whole-request handoff.
"""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.csm"

# The talker forces this id on the step after a stop frame. It is the
# checkpoint's codebook_pad_token_id: inside the first-codebook vocabulary
# (0..2050), which vLLM requires of a stop token id, and never an audio code.
_CSM_STOP_TOKEN_ID = 2050

CSM_PIPELINE = PipelineConfig(
    model_type="csm",
    default_deploy_config_name="csm.yaml",
    model_arch="CsmTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="csm_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            supports_native_mrv2_data_plane=True,
            sampling_constraints={"detokenize": False, "stop_token_ids": [_CSM_STOP_TOKEN_ID]},
            # Each decode step feeds the previous frame's 32 codes, which the token stream does not hold.
            recompute_preemption="fail",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="csm_code2wav",
            model_arch="CsmCode2Wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            supports_native_mrv2_data_plane=True,
            retains_state_across_chunks=True,
            sampling_constraints={"detokenize": True},
        ),
    ),
)
