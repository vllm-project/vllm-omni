# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMind-Omni pipeline topology.

Stage 0: Thinker  - multimodal understanding + text generation
Stage 1: Talker   - thinker hidden states -> speech codec tokens
Stage 2: Code2Wav - speech codec tokens -> audio waveform
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minimind_omni"

MINIMIND_OMNI_PIPELINE = PipelineConfig(
    model_type="minimind-o",
    model_arch="MiniMindOmniForConditionalGeneration",
    hf_architectures=(
        "MiniMindOmniForConditionalGeneration",
        "MiniMindOmniThinkerForConditionalGeneration",
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
            engine_output_type="latent",
            sampling_constraints={
                "detokenize": True,
                "ignore_eos": True,
                "stop_token_ids": [17],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.thinker2talker",
            sampling_constraints={"detokenize": False, "stop_token_ids": [2052]},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
