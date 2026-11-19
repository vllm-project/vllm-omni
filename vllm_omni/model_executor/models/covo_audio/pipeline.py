# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Covo-Audio pipeline topology (frozen).

Stage 0: fused_thinker_talker — multimodal understanding + interleaved text/audio
         token generation (AR).
Stage 1: code2wav              — audio codes → waveform (generation).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    TEXT_RESPONSE,
    TEXT_TO_SPEECH,
    declare_capabilities,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.covo_audio"

COVO_AUDIO_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_RESPONSE,
        AUDIO_RESPONSE,
        TEXT_TO_SPEECH,
        sources=("vllm_omni/model_executor/models/covo_audio/pipeline.py",),
    ),
    model_type="covo_audio",
    default_deploy_config_name="covo_audio.yaml",
    model_arch="CovoAudioForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.llm2code2wav_full_payload",
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [151645],
                "ignore_eos": True,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.llm2code2wav_token_only",
            sampling_constraints={"detokenize": False},
            requires_full_payload_input=True,
        ),
    ),
)
