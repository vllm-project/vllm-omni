# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming-flash-omni-2.0 pipeline topology (frozen).

Stage 0: Thinker — multimodal understanding + text generation.
Stage 1: Talker  — text -> audio waveform via CFM + AudioVAE.

The thinker -> talker bridge passes the detokenized text rather than
hidden states through `ming_flash_omni.thinker2talker`; the talker has a
self-contained Qwen2 LLM that retokenizes the string itself.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.ming_flash_omni"


MING_FLASH_OMNI_PIPELINE = PipelineConfig(
    model_type="ming_flash_omni",
    model_arch="MingFlashOmniForConditionalGeneration",
    # Upstream HF config applies model_type="bailingmm_moe_v2_lite"
    # (the thinker sub-config name) rather than "ming_flash_omni".
    # Declare the architectures here explicitly for routing
    hf_architectures=(
        "MingFlashOmniForConditionalGeneration",
        "BailingMM2NativeForConditionalGeneration",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            # Thinker reads the LLM sub-config of BailingMM2Config
            hf_config_name="llm_config",
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="ming_tts",
            # Talker is a self-contained Qwen2 LLM + CFM + AudioVAE;
            # it does not share the thinker backbone, so it gets its own arch.
            model_arch="MingFlashOmniTalkerForConditionalGeneration",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="talker_config",
            engine_output_type="audio",
            tokenizer_subdir="talker/llm",
            custom_process_input_func=f"{_PROC}.thinker2talker",
        ),
    ),
)


# Standalone TTS variant: talker only.
MING_FLASH_OMNI_TTS_PIPELINE = PipelineConfig(
    model_type="ming_flash_omni_tts",
    model_arch="MingFlashOmniTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ming_tts",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            hf_config_name="talker_config",
            engine_output_type="audio",
            tokenizer_subdir="talker/llm",
        ),
    ),
)


# Thinker-only variant: multimodal understanding with text output
MING_FLASH_OMNI_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="ming_flash_omni_thinker_only",
    model_arch="MingFlashOmniForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="llm_config",
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
