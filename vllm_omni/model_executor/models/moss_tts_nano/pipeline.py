# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS-Nano pipeline topology (frozen).

Single-stage AR TTS: text -> speech waveform in one pass. The 0.1B AR LM
and the MOSS-Audio-Tokenizer-Nano codec both run inside
``MossTTSNanoForGeneration.forward()``, which uses the VoxCPM-style
generator pattern (``inference_stream()`` stored per-request; one audio
chunk yielded per forward) to drive progressive streaming through the
AR scheduler.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MOSS_TTS_NANO_PIPELINE = PipelineConfig(
    model_type="moss_tts_nano",
    model_arch="MossTTSNanoForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts_nano",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # compute_logits() forces EOS (token id 2) when the last
                # streaming chunk is yielded; keep a hard backstop here.
                "stop_token_ids": [2],
            },
        ),
    ),
)
