# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenVLA single-stage action-policy topology.

OpenVLA is the first robot policy here whose actions come out of an
autoregressive decoder rather than a denoiser: the seven action dimensions are
seven discretised tokens drawn from the tail of the Llama-2 vocabulary, decoded
one per step. That makes it a plain ``LLM_AR`` stage — the model class is
upstream vLLM's ``OpenVLAForActionPrediction``, which ``OmniModelRegistry``
already resolves — and everything robot-specific lives in the token→action
decode on the serving side.

``engine_output_type`` is deliberately unset. ``final_output_type`` is a free
string (GR00T already uses ``"actions"``), but ``engine_output_type`` is parsed
by ``OutputModality.from_string``, which knows only text/image/audio/latent and
raises on anything else.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# One token per action dimension. Every one of the 25 embodiments in
# openvla-7b's norm_stats has a 7-dimensional action space.
OPENVLA_ACTION_DIM = 7

OPENVLA_PIPELINE = PipelineConfig(
    model_type="openvla",
    default_deploy_config_name="openvla.yaml",
    # No hf_architectures: vLLM ships an `OpenVLAConfig` for model_type
    # "openvla", so the primary model_type match already resolves this pipeline
    # and the architecture fallback would never be reached.
    model_arch="OpenVLAForActionPrediction",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="actions",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="OpenVLAForActionPrediction",
            robot_adapter=("vllm_omni.model_executor.models.openvla.robot_adapter.OpenVLARobotAdapter"),
            # The action head is argmax over 256 bins, so sampling is greedy and
            # the length is fixed. ``ignore_eos`` matters because an action bin
            # id could otherwise be cut short by an EOS, and ``detokenize`` is
            # off because the tokens are bin indices, not text.
            sampling_constraints={
                "temperature": 0.0,
                "max_tokens": OPENVLA_ACTION_DIM,
                "min_tokens": OPENVLA_ACTION_DIM,
                "ignore_eos": True,
                "detokenize": False,
            },
        ),
    ),
)
