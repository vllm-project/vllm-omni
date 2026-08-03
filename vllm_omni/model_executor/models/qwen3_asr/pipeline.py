# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Qwen3-ASR pipeline topology.

A single comprehension stage: audio in, transcript text out.

``Qwen3ASRForConditionalGeneration`` already implements vLLM's
``SupportsTranscription``, so this topology is what ``AsyncOmniEngine``'s
capability probe looks for -- a transcription-capable comprehension stage that
is also the text terminal -- and it is what makes ``/v1/audio/transcriptions``
reachable.

Contrast ``aura_omni``, which runs the same model at stage 0 but as an
intermediate feeding a Qwen3-VL stage. A request there resolves to the *last*
text stage, so the ASR stage cannot be addressed on its own.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN3_ASR_PIPELINE = PipelineConfig(
    model_type="qwen3_asr",
    model_arch="Qwen3ASRForConditionalGeneration",
    hf_architectures=("Qwen3ASRForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3ASRForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
