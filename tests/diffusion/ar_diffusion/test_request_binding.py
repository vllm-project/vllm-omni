# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV ownership follows upstream Session ticks, not a second parser."""

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.request_binding import KVRequestBinding, kv_request_binding
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_typed_tick_owns_pool_identity_reset_close_and_position():
    tick = ARDiffusionTickRequest("session", "turn", chunk_index=5, reset=True, close_session=True)
    extra = {"session_id": "ignored", "chunk_base": 99, **tick.to_extra_args()}
    req = OmniDiffusionRequest(
        request_id="transport",
        prompt="test",
        sampling_params=OmniDiffusionSamplingParams(extra_args=extra),
    )
    assert kv_request_binding(req) == KVRequestBinding("session", False, True, True, 5)


def test_ephemeral_requests_are_isolated_and_explicit_session_persists():
    req = OmniDiffusionRequest(
        request_id="one",
        prompt="test",
        sampling_params=OmniDiffusionSamplingParams(),
    )
    assert kv_request_binding(req) == KVRequestBinding("one", False, False, False, 0)
    req.sampling_params.extra_args = {"session_id": "s", "chunk_base": 2}
    assert kv_request_binding(req) == KVRequestBinding("s", True, False, False, 2)
    req.sampling_params.extra_args["chunk_base"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        kv_request_binding(req)
