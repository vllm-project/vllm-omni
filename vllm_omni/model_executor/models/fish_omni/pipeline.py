# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Audio Drama 3 pipeline topology (frozen).

Stage 0: fish_omni   — text + voice references → one sampled 10-codebook frame per
         step (Qwen3.5 MoE backbone + codebook tower, Model Runner V2).
Stage 1: dac_decoder — frames → 44.1 kHz audio; the same Fish DAC codec as S2 Pro.
"""

from dataclasses import replace

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    TEXT_TO_SPEECH,
    declare_capabilities,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.fish_omni"

FISH_OMNI_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/fish_omni/pipeline.py", "docs/serving/speech_api.md"),
    ),
    model_type="fish_omni",
    default_deploy_config_name="fish_omni.yaml",
    model_arch="FishOmniForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fish_omni",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.fish_omni_to_dac_decoder_async_chunk",
            sampling_constraints={
                "detokenize": False,
                # Drama 3 end of speech (config eos_token_id).
                "stop_token_ids": [248046],
            },
            # MRv2 saves generated codebooks on preemption and restores the full
            # frame rows on resume, so retain the default recompute policy.
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dac_decoder",
            model_arch="FishSpeechDACDecoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
