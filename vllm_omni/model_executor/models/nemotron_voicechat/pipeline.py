# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronVoiceChat pipeline: thinker (speech -> frame-locked text) -> talker
(text timeline -> 31-quantizer RVQ code stacks) -> code2wav (codes -> 22.05 kHz PCM).

``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`` is an 11B full-duplex speech-to-speech
model. Offline it is served as a qwen3-omni-style 3-stage pipeline:

* Stage 0 (``thinker``, LLM_AR): vendored FastConformer perception + additive
  channel fusion feeding the NemotronH backbone on vLLM; emits the frame-locked
  agent text channel (also the final text output) and carries the full frame
  timeline + metadata to the talker as a latent payload.
* Stage 1 (``talker``, LLM_AR): the vendored Gemma3-based EAR-TTS model stepped
  per frame (classifier-free guidance + MoG sampling run inside the vendored
  forward); emits one 31-code stack per frame as latents.
* Stage 2 (``code2wav``, LLM_GENERATION): the vendored RVQ-VAE codec decoder
  (fp32) producing the final 22.05 kHz waveform.

The STT->TTS coupling is one-directional and token-ids-only (verified against
the NeMo reference), which is what makes the offline cascade exact.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.nemotron_voicechat"

NEMOTRON_VOICECHAT_PIPELINE = PipelineConfig(
    model_type="nemotron_voicechat",
    model_arch="NemotronVoiceChatThinkerForConditionalGeneration",
    duplex_runtime_extension=(
        "vllm_omni.experimental.fullduplex.nemotron_voicechat.runtime.NemotronVoiceChatDuplexRuntimeExtension"
    ),
    duplex_serving_adapter=(
        "vllm_omni.experimental.fullduplex.nemotron_voicechat.serving_adapter.NemotronVoiceChatServingRuntimeAdapter"
    ),
    duplex_control_enabled=True,
    # Named after the alias key so the deploy-yaml stem substring-matches the
    # checkpoint directory name (NVIDIA-NemotronLabs-VoiceChat-11B) during
    # bare-path auto-detection.
    default_deploy_config_name="nemotron_labs_voicechat.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            final_output=True,
            final_output_type="text",
            engine_output_type="latent",
            # thinker -> talker sync mode is token-path only: the frame-locked
            # text timeline rides the talker's prompt_token_ids (built by
            # thinker2talker_token_only on the talker stage); no connector
            # payload and no full-payload wait for stage 1. With
            # ``async_chunk: true`` in the deploy yaml, the timeline instead
            # streams incrementally over the connector (cumulative ids.all per
            # thinker step) so the talker starts before the thinker finishes.
            async_chunk_process_next_stage_input_func=f"{_PROC}.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            model_arch="NemotronVoiceChatTalkerForConditionalGeneration",
            hf_config_name="talker_config",
            input_sources=(0,),
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            sync_process_input_func=f"{_PROC}.thinker2talker_token_only",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_arch="NemotronVoiceChatCode2Wav",
            hf_config_name="code2wav_config",
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": False},
            requires_full_payload_input=True,
        ),
    ),
)
