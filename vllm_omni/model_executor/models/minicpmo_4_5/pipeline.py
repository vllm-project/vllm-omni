# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 pipeline topology.

Stage 0: Thinker — multimodal understanding + text generation.
Stage 1: Talker  — MiniCPMTTS, emits generated audio tokens.
Stage 2: Token2Wav — decodes audio tokens into the final waveform.

The thinker -> talker bridge passes the hidden states + token ids extracted
from the thinker output through ``minicpmo_4_5_omni.llm2talker``. The
talker -> token2wav bridge sends a complete ``codes.audio`` payload; async
chunking remains disabled for PR1.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni"


MINICPMO_4_5_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5",
    default_deploy_config_name="minicpmo_4_5.yaml",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    # MiniCPM-o 2.6 and 4.5 both advertise ``architectures=["MiniCPMO"]``.
    # The version predicate keeps 2.6 checkpoints out of the 4.5 pipeline.
    hf_architectures=("MiniCPMO", "MiniCPMO45OmniForConditionalGeneration"),
    hf_config_predicate=lambda c: str(getattr(c, "version", "")) == "4.5",
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
            model_stage="talker",
            # Keep the wrapper here so runner-side runtime metadata reaches
            # the Talker; wiring the standalone TTS module directly would hit
            # the dummy path.
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="tts_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.llm2talker",
            custom_process_next_stage_input_func=f"{_PROC}.talker2token2wav_full_payload",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="token2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="tts_config",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.talker2token2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
