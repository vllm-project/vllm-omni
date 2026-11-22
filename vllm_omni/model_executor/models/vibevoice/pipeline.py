# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice pipeline topology.

Single-stage autoregressive TTS: text and an optional reference audio are
consumed by Qwen2, while CFG diffusion, acoustic decoding, semantic re-encoding,
and continuous-embedding feedback stay inside the same decode stage. The acoustic
decoder cannot be split into a second stage because every waveform chunk feeds
the semantic embedding consumed by the next AR step.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# audio BOS / audio EOS / audio placeholder / <|endoftext|>
VIBEVOICE_VALID_TOKEN_IDS = [151652, 151653, 151654, 151643]

VIBEVOICE_PIPELINE = PipelineConfig(
    model_type="vibevoice",
    default_deploy_config_name="vibevoice.yaml",
    model_arch="VibeVoiceForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vibevoice",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            # Keeps startup encoder profiling for the worst-case reference; the reference stays optional.
            requires_multimodal_data=True,
            prompt_transform_func="vllm_omni.model_executor.models.vibevoice.prompt.check_reference_prompt",
            sampling_constraints={
                "temperature": 0.0,
                "detokenize": False,
                "allowed_token_ids": VIBEVOICE_VALID_TOKEN_IDS,
                # <|endoftext|>; audio_eos is an AR state transition, not a
                # request stop.
                "stop_token_ids": [151643],
            },
        ),
    ),
)

__all__ = ["VIBEVOICE_PIPELINE", "VIBEVOICE_VALID_TOKEN_IDS"]
