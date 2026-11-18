# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vllm-Omni project
"""Magenta RealTime 2 pipeline: Talker (Stage 0 MRv2 AR) -> SpectroStream Code2Wav (Stage 1 MRv2 Generation)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.magenta_rt2"

MAGENTA_RT2_PIPELINE = PipelineConfig(
    model_type="magenta_realtime",
    model_arch="MagentaRT2ForConditionalGeneration",
    default_deploy_config_name="magenta_rt2_mrv2.yaml",
    duplex_plugin="vllm_omni.model_executor.models.magenta_rt2.duplex.plugin.MagentaRT2DuplexPlugin",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=False,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={"detokenize": False},
            recompute_preemption="fail",
            supports_native_mrv2_data_plane=True,
            supports_duplex_mrv2=True,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="SpectroStreamCode2Wav",
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            retains_state_across_chunks=True,
            supports_native_mrv2_data_plane=True,
            supports_duplex_mrv2=True,
        ),
    ),
)

__all__ = ["MAGENTA_RT2_PIPELINE"]
