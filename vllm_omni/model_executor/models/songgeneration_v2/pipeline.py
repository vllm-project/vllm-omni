# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SongGeneration v2-large pipeline: Stage 0 (LeLM AR) -> Stage 1 (Flow1dVAE)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# Stage-input processor module path.
_PROC = "vllm_omni.model_executor.stage_input_processors.songgeneration_v2"


_LELM_MODEL_ARCH = "SongGenerationV2LeLMForConditionalGeneration"


SONGGENERATION_V2_PIPELINE = PipelineConfig(
    model_type="songgeneration_v2",
    model_arch=_LELM_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="songgeneration_v2_lelm",
            model_arch=_LELM_MODEL_ARCH,
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            # ``owns_tokenizer=True`` marks this as a "comprehension" stage in
            # AsyncOmniEngine so ``supported_tasks`` includes ``generate``.
            # Lyric tokenization uses ``QwTokenizerConditioner`` internally.
            owns_tokenizer=True,
            tokenizer_subdir="third_party/Qwen2-7B",
            engine_output_type="codes",
            # No async-chunk hook: Stage 0 emits the full codec tensor once at
            # finish (not per-step frames), so async-chunk streaming is
            # unsupported (merge_pipeline_deploy rejects async_chunk: true).
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [16384],
            },
            extras={
                "per_stream_top_k": [5000, 1, 1],
                "cfg_coef": 1.5,
                "code_depth": 3,
                "delay_pattern": [0, 250, 250],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="songgeneration_v2_flow1dvae",
            model_arch="SongGenerationV2Flow1dVAESeparateDecoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            owns_tokenizer=False,
            input_sources=(0,),
            sync_process_input_func=f"{_PROC}.lelm_to_flow1dvae",
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
            extras={
                "decode_window_frames": 1000,
                "decode_hop_frames": 750,
                "decode_overlap_frames": 250,
                "guidance_scale": 1.5,
                "num_steps": 10,
                "output_channels": 2,
                "sample_rate": 48000,
            },
        ),
    ),
)
