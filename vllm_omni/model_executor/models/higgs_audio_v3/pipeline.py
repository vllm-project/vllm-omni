# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""higgs-audio v3 pipeline: Talker (text -> 8-codebook codec) -> Code2Wav (codec -> 24 kHz PCM).

Two delivery modes are wired here:

* ``sync_process_input_func`` runs when the deploy YAML has
  ``async_chunk: false``. The orchestrator collects the entire Stage-0 emit,
  reverts the delay pattern once, and hands a single payload to Stage 1.
* ``async_chunk_process_next_stage_input_func`` runs when the deploy YAML
  has ``async_chunk: true``. Stage 0 dispatches per AR step; the streaming
  adapter buffers raw delay-pattern rows, slides a window with left context
  and right holdback, and emits codec-ready frames per chunk. Stage 1
  trims the overlap on both ends so the client sees a coherent PCM stream.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.higgs_audio_v3"

HIGGS_AUDIO_V3_PIPELINE = PipelineConfig(
    model_type="higgs_multimodal_qwen3",
    model_arch="HiggsAudioV3TalkerForConditionalGeneration",
    hf_architectures=("HiggsMultimodalQwen3ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="higgs_audio_v3",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            # stop_token_ids: the model-owned sampler forces eos at ramp-down
            # completion. Safety stops from the actual V3 checkpoint:
            #   151643 = <|endoftext|> (eos_token_id from config.json)
            #   151671 = <|audio_end|> (audio generation end marker)
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [151643, 151671],
            },
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="HiggsAudioV3Code2WavForConditionalGeneration",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
