# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS-Nano pipeline topology (frozen).

Single-stage AR TTS: text -> speech waveform in one pass. The 0.1B AR LM
and the MOSS-Audio-Tokenizer-Nano codec both run inside
``MossTTSNanoForGeneration.forward()``, which uses the VoxCPM-style
generator pattern (``inference_stream()`` stored per-request; one audio
chunk yielded per forward) to drive progressive streaming through the
AR scheduler.
"""

from dataclasses import replace

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    TEXT_TO_SPEECH,
    declare_capabilities,
)

MOSS_TTS_NANO_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/moss_tts_nano/pipeline.py", "docs/serving/speech_api.md"),
    ),
    model_type="moss_tts_nano",
    default_deploy_config_name="moss_tts_nano.yaml",
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
