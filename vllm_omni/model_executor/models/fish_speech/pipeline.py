# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fish Speech S2 Pro pipeline topology (frozen).

Stage 0: slow_ar     — text → RVQ codec tokens (LLM autoregressive).
Stage 1: dac_decoder — RVQ tokens → audio waveform (LLM_GENERATION).

The HF config top-level reports ``model_type = "fish_qwen3_omni"`` (the
OmniConfig that bundles slow-AR and fast-AR sub-configs), which is why the
registry key follows the HF name rather than the human-readable "fish_speech".
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.fish_speech"

FISH_SPEECH_PIPELINE = PipelineConfig(
    model_type="fish_qwen3_omni",
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
