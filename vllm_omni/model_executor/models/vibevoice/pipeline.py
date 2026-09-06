# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice pipeline topology (frozen).

Single-stage autoregressive TTS: text and optional reference audio are consumed
by Qwen2, while CFG diffusion, acoustic decoding, semantic re-encoding, and
continuous-embedding feedback remain inside the same decode stage.

The acoustic decoder must not be split into a Code2Wav stage because every
waveform chunk contributes the semantic embedding used by the next AR step.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

VIBEVOICE_VALID_TOKEN_IDS = [151652, 151653, 151654, 151643]

VIBEVOICE_PIPELINE = PipelineConfig(
    model_type="vibevoice",
    default_deploy_config_name="vibevoice.yaml",
    model_arch="VibeVoiceForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            # Keep a distinct serving discriminator; reusing VoxCPM2's
            # "latent_generator" stage name routes speech requests to the
            # wrong model-specific adapter.
            model_stage="vibevoice",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            # Voice cloning passes reference waveforms to the Acoustic Encoder.
            requires_multimodal_data=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # vLLM's built-in allowed-token mask is equivalent to the PR's
                # VibeVoiceTokenConstraintProcessor and runs in the stock v1
                # sampler on every decode step.
                "allowed_token_ids": VIBEVOICE_VALID_TOKEN_IDS,
                # Qwen2 <|im_end|>; audio_eos is an AR state transition and is
                # intentionally not treated as the request stopping token.
                "stop_token_ids": [151643],
            },
        ),
    ),
)

__all__ = ["VIBEVOICE_PIPELINE", "VIBEVOICE_VALID_TOKEN_IDS"]
