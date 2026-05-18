# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 pipeline topology (frozen).

Stage 0: Thinker  — multimodal understanding + text generation.
Stage 1: Talker   — MiniCPMTTS over thinker hidden states + tokens.
Stage 2: Token2Wav — wraps the waveform produced by the talker.

The thinker -> talker bridge passes the hidden states + token ids extracted
from the thinker output through ``minicpmo_4_5_omni.llm2tts``; the talker ->
token2wav bridge passes the waveform through ``minicpmo_4_5_omni.tts2t2w``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni"


MINICPMO_4_5_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    # MiniCPM-o reports model_type="minicpmo" in its HF config.  Declare both
    # the registry key (this PipelineConfig.model_type) and the architecture
    # names so StageConfigFactory can route either way.
    hf_architectures=("MiniCPMO45OmniForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="tts",
            # Talker is a self-contained MiniCPMTTS + Token2wav module; it
            # consumes thinker hidden states + token ids rather than sharing
            # the thinker backbone, so it carries its own model_arch.
            model_arch="MiniCPMO45OmniTTSForConditionalGeneration",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=False,
            hf_config_name="tts_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.llm2tts",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="t2w",
            model_arch="MiniCPMO45OmniT2WForConditionalGeneration",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="tts_config",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.tts2t2w",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
