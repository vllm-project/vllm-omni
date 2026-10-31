# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 pipeline: Talker (text -> 9-codebook DAC tokens) -> Code2Wav (DAC 44.1kHz PCM).

Supports sync completion and native DAC overlap-add streaming.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.zonos2"

ZONOS2_PIPELINE = PipelineConfig(
    model_type="zonos2",
    default_deploy_config_name="zonos2.yaml",
    model_arch="Zonos2ForConditionalGeneration",
    hf_architectures=("Zonos2ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="zonos2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2dac_async_chunk",
            # EOS handling is owned by the model-side sampler (any codebook
            # hitting eoa starts the n_codebooks+1 countdown); the LM token
            # stream is a lifecycle channel only.
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [1],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dac_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Zonos2Code2WavForConditionalGeneration",
            sync_process_input_func=f"{_PROC}.talker2dac",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
