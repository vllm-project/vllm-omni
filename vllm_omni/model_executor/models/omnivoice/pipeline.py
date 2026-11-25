# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice pipeline topology (frozen).

Single-stage diffusion model for text-to-speech.
Stage 0: DiT — text → iterative unmasking → codebook tokens → DAC decode → 24kHz audio
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

OMNIVOICE_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/omnivoice/pipeline.py", "docs/serving/speech_api.md"),
    ),
    model_type="omnivoice",
    default_deploy_config_name="omnivoice.yaml",
    model_arch="OmniVoicePipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            engine_output_type="audio",
        ),
    ),
)
