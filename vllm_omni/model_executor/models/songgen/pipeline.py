# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SongGen pipeline topology (frozen).

Single-stage AR TTS: lyrics + description text -> 16 kHz song waveform in one
pass. The 1.3B SongGen AR model and the X-Codec decoder both run inside
``SongGenForGeneration.forward()``, which uses the VoxCPM-style generator
pattern (one blocking ``generate()`` call per request that yields the complete
waveform as a single final chunk).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

SONGGEN_PIPELINE = PipelineConfig(
    model_type="songgen",
    model_arch="SongGenMixedForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="songgen",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # compute_logits() forces EOS (token id 2) after the single
                # waveform chunk is yielded; keep a hard backstop here.
                "stop_token_ids": [2],
            },
        ),
    ),
)
