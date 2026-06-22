# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OmniRequest substitutability with the base Request.

vllm-omni rebinds ``vllm.v1.request.Request`` to ``OmniRequest`` at import
time (see ``vllm_omni/patch.py``). vLLM core constructs ``Request`` positionally
in some paths (notably ``vllm/v1/worker/gpu/warmup.py::warmup_kernels`` for V2
model-runner architectures like Qwen3ForCausalLM). The omni-specific params
must therefore be keyword-only so positional construction stays Liskov-
substitutable with the base class — including base-style calls that pass
``prompt_embeds`` (itself a positional-capable base param) positionally.
"""

import inspect

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from vllm_omni.engine import PromptEmbedsPayload
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_params_are_keyword_only():
    """The three omni params must be keyword-only after ``*args``.

    Guards against re-introducing the import-time rebind bug where omni params
    came first positionally and broke positional ``Request(...)`` construction.
    """
    params = inspect.signature(OmniRequest.__init__).parameters
    for name in ("prompt_embeds", "external_req_id", "additional_information"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_positional_construction_matches_base_request():
    """Reproduces the warmup_kernels call: Request(id, tokens, sp, pp)."""
    req = OmniRequest("req-0", [1, 2, 3], SamplingParams(), None)

    assert isinstance(req, Request)
    assert req.request_id == "req-0"
    assert req.prompt_token_ids == [1, 2, 3]
    # Omni params default cleanly when constructed positionally.
    assert req.external_req_id is None
    assert req.additional_information is None
    assert req.prompt_embeds_payload is None


def test_positional_prompt_embeds_does_not_collide():
    """Base-style positional call that includes ``prompt_embeds``.

    ``prompt_embeds`` is a positional-capable base param (after
    ``arrival_time``). A base-style call must not trip
    ``got multiple values for argument 'prompt_embeds'`` — the subclass must
    not also inject its keyword override when the value arrived positionally.
    Positional order: request_id, prompt_token_ids, sampling_params,
    pooling_params, client_index, arrival_time, prompt_embeds.
    """
    embeds = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    req = OmniRequest("req-pe", [1, 2], SamplingParams(), None, 0, None, embeds)

    assert isinstance(req, Request)
    assert torch.equal(req.prompt_embeds, embeds)
    # Positional tensor is not a serialized payload.
    assert req.prompt_embeds_payload is None
    assert req.external_req_id is None


def test_keyword_omni_params_round_trip():
    """Keyword omni params are preserved; serialized embeds are decoded."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = PromptEmbedsPayload(data=arr.tobytes(), shape=[2, 3], dtype="float32")

    req = OmniRequest(
        request_id="req-1",
        prompt_token_ids=[7, 8],
        sampling_params=SamplingParams(),
        pooling_params=None,
        prompt_embeds=payload,
        external_req_id="ext-1",
    )

    assert req.external_req_id == "ext-1"
    # The serialized payload is retained and decoded into a tensor on the base.
    assert req.prompt_embeds_payload is payload
    assert torch.equal(req.prompt_embeds, torch.from_numpy(arr))
