# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 2.6 pipeline topology (frozen).

Stage 0: Thinker --- multimodal understanding + text generation.
Stage 1: Talker  --- ConditionalChatTTS + Vocos, emits the final audio waveform.

The thinker -> talker bridge passes the hidden states + token ids extracted
from the thinker output through ``minicpmo_2_6_omni.llm2tts``; the talker
runs ConditionalChatTTS + DVAE + Vocos vocoder in the same process and
returns the waveform directly as the pipeline's final audio output.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo_2_6_omni"


MINICPMO_2_6_PIPELINE = PipelineConfig(
    model_type="minicpmo_2_6",
    model_arch="MiniCPMO26OmniForConditionalGeneration",
    hf_architectures=("MiniCPMO", "MiniCPMO26OmniForConditionalGeneration"),
    hf_config_predicate=lambda c: str(getattr(c, "version", "")) == "2.6",
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
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="tts_config",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.llm2tts",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
