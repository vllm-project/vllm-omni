# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for cross-stage LoRA routing in the orchestrator."""

from __future__ import annotations

import pytest
import torch
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.engine.serialization import deserialize_model_intermediate_buffer
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestBuildEngineCoreRequestLoRA:
    """Verify build_engine_core_request_from_tokens passes LoRA from params."""

    def test_lora_extracted_from_diffusion_params(self):
        lr = LoRARequest(lora_name="test", lora_int_id=1, lora_path="/tmp/fake")
        params = OmniDiffusionSamplingParams(lora_request=lr)

        # OmniDiffusionSamplingParams is not a SamplingParams, so
        # build_engine_core_request_from_tokens takes the pooling path.
        # We only care that lora_request is extracted via getattr.
        request = build_engine_core_request_from_tokens(
            request_id="req-1",
            prompt={"prompt_token_ids": [1, 2, 3]},
            params=params,
            model_config=None,
        )
        assert request.lora_request is lr

    def test_no_lora_on_sampling_params(self):
        params = SamplingParams(max_tokens=10)

        request = build_engine_core_request_from_tokens(
            request_id="req-2",
            prompt={"prompt_token_ids": [1, 2, 3]},
            params=params,
            model_config=None,
        )
        assert request.lora_request is None

    def test_model_buffer_tensor_is_prepared_for_engine_transport(self):
        hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        request = build_engine_core_request_from_tokens(
            request_id="req-buffer",
            prompt={
                "prompt_token_ids": [0, 0],
                "model_intermediate_buffer": {
                    "hidden_states": {"tts": hidden},
                    "legacy": [[0.0, 1.0]],
                },
            },
            params=SamplingParams(max_tokens=1),
            model_config=None,
        )

        restored = deserialize_model_intermediate_buffer(request.model_intermediate_buffer)
        assert restored is not None
        assert restored["legacy"] == [[0.0, 1.0]]
        assert torch.equal(restored["hidden_states"]["tts"], hidden)
