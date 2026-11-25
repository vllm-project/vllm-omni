# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared test fixtures for scheduler tests."""

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest


def make_engine_core_request(request_id: str = "req") -> EngineCoreRequest:
    """Minimal EngineCoreRequest for scheduler tests."""
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
