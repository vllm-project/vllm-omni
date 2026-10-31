# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM-Omni pipeline topology for DRVEXO-speech-8b.

The Thinker includes the audio encoder and trained input projector. Its
generated lexical tokens pass through the native Qwen3-TTS text projection
to the Talker; Code2Wav decodes the acoustic tokens. The rejected learned
hidden-state bridge is not the accepted speech route. Registering a topology
does not establish component quality or final model acceptance.
"""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"
_DRVEXO = "vllm_omni.model_executor.stage_input_processors.drvexo_speech_8b"

DRVEXO_SPEECH_8B_PIPELINE = PipelineConfig(
    model_type="drvexo_speech_8b",
    default_deploy_config_name="drvexo_speech_8b.yaml",
    model_arch="DRVEXOSpeech8BForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            # Generated-token mode hands text through the orchestrator.
            # Only Talker -> Codec uses asynchronous connector chunks.
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            model_arch="DRVEXOSpeech8BTalker",
            model_subdir="talker",
            engine_output_type="latent",
            # The incoming Thinker edge is synchronous even when the outgoing
            # Talker-to-Codec edge uses async chunks. Keep its processor in both modes.
            custom_process_input_func=f"{_DRVEXO}.thinker2talker_token_only",
            sync_process_input_func=f"{_DRVEXO}.thinker2talker_token_only",
            custom_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={"detokenize": False, "stop_token_ids": [2150]},
            requires_full_payload_input=True,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            model_arch="DRVEXOSpeech8BCode2Wav",
            model_subdir="talker",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)

# A single-stage variant isolates input and decision integration tests from
# Talker/Codec. It loads the shared Qwen3 backbone, audio tower and projector.
DRVEXO_SPEECH_8B_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="drvexo_speech_8b_thinker_only",
    model_arch="DRVEXOSpeech8BForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=False,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
