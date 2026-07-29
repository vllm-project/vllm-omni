# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vevo2 pipeline topology (frozen).

Single-stage AR TTS: text -> speech waveform in one pass. The full
upstream pipeline (Qwen2.5-0.5B AR LM, 350M flow-matching transformer,
250M Vocos vocoder) runs inside :class:`Vevo2ForCausalLM.forward`, which
uses the MOSS-TTS-Nano-style generator pattern: ``inference_ar_and_fm()``
yields the full waveform once per request, then a sentinel that lets
``compute_logits()`` finish the AR scheduler.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

VEVO2_PIPELINE = PipelineConfig(
    model_type="vevo2",
    default_deploy_config_name="vevo2.yaml",
    model_arch="Vevo2ForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vevo2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # ``compute_logits()`` forces EOS (token id 2) when the
                # last streaming chunk is yielded; keep a hard backstop
                # here so misaligned batches still terminate.
                "stop_token_ids": [2],
            },
        ),
    ),
)
