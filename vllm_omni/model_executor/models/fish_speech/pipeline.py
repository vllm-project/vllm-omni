# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Speech S2 Pro pipeline topology (frozen).

Stage 0: slow_ar     — text → RVQ codec tokens (LLM autoregressive).
Stage 1: dac_decoder — RVQ tokens → audio waveform (LLM_GENERATION).

The HF config top-level reports ``model_type = "fish_qwen3_omni"`` (the
OmniConfig that bundles slow-AR and fast-AR sub-configs), which is why the
registry key follows the HF name rather than the human-readable "fish_speech".
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

_PROC = "vllm_omni.model_executor.stage_input_processors.fish_speech"

FISH_SPEECH_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/fish_speech/pipeline.py", "docs/serving/speech_api.md"),
    ),
    model_type="fish_qwen3_omni",
    default_deploy_config_name="fish_qwen3_omni.yaml",
    model_arch="FishSpeechSlowARForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fish_speech_slow_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.slow_ar_to_dac_decoder_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                # <|im_end|> — stop when the model emits end-of-turn.
                "stop_token_ids": [151645],
            },
            # talker_mtp adds the sampled codebook embeddings to this step's
            # inputs_embeds; a replay pads the unbuilt rows (vllm-project/vllm-omni#6179).
            recompute_preemption="fail",
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
