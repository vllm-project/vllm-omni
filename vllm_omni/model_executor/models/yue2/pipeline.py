# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B pipeline topology.

Single-stage native-AR text-to-music (gepard pattern). The whole
Mixture-of-Transformers checkpoint — AR path, NAR path, projection heads —
lives in one ``model.safetensors``, so both the autoregressive sampling and
the terminal flow-matching + VAE pass run inside one stage and no weights are
duplicated across stages. The VAE decoder is a separate repository
(``m-a-p/YuE2-Vae``) loaded lazily at the first song finish, from
``$YUE2_VAE`` or the default hub id.

Sampling is model-owned. ``sampling_constraints`` pins ``detokenize`` and the
union of both phase end tokens; ``extra_args`` carry the phase and the
request-local sampling preset, and must be present in the stage's
``default_sampling_params`` so ``has_sampling_extra_args`` turns on and the
per-request args reach the model's forward.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

from .constants import STOP_TOKEN_IDS

YUE2_PIPELINE = PipelineConfig(
    model_type="yue2",
    default_deploy_config_name="yue2.yaml",
    model_arch="Yue2ForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="yue2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            # Whole-song audio ships once, on the finishing step; the audio
            # accumulator concatenates over time and a single tensor just
            # passes through.
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": list(STOP_TOKEN_IDS),
            },
        ),
    ),
)

__all__ = ["YUE2_PIPELINE"]
