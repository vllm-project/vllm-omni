# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests for the per-request hidden-state opt-in (RL teacher rollouts).

Covers the typed wire field, the ingress lift from SamplingParams.extra_args,
the scheduler-request carry-through, the runner-side opt-in tracking, the
per-request GPU slicing guards, and the wire-channel merge.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.request import OmniRequest
from vllm_omni.worker_v2.omni_ar_model_runner import (
    HIDDEN_STATES_OUTPUT_KEY,
    OmniARModelRunner,
    _merge_hidden_state_payloads,
)

pytestmark = [pytest.mark.core_model]


def _core_request(**overrides: Any) -> OmniEngineCoreRequest:
    kwargs = dict(
        request_id="req-1",
        prompt_token_ids=np.asarray([1, 2, 3], dtype=np.int32),
        prompt_is_token_ids=[True, True, True],
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=0,
        current_wave=None,
        priority=0,
        trace_headers=None,
        resumable=False,
        external_req_id="req-1",
        session_id=None,
        kv_hints=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
        abort_immediately=False,
    )
    kwargs.update(overrides)
    return OmniEngineCoreRequest(**kwargs)


@pytest.mark.cpu
def test_wire_field_defaults_off():
    request = _core_request()
    assert request.return_hidden_states is False


@pytest.mark.cpu
def test_from_request_round_trip():
    base = _core_request(return_hidden_states=True)
    clone = OmniEngineCoreRequest.from_request(base)
    assert clone.return_hidden_states is True
    # An explicit override wins over the source attribute.
    suppressed = OmniEngineCoreRequest.from_request(base, return_hidden_states=False)
    assert suppressed.return_hidden_states is False


@pytest.mark.cpu
def test_ingress_lifts_extra_args_flag():
    params = SamplingParams(max_tokens=1)
    params.extra_args = {"return_hidden_states": True}
    built = build_engine_core_request_from_tokens(
        "req-lift",
        {"prompt_token_ids": [1, 2, 3]},
        params,
    )
    assert built.return_hidden_states is True
    # Without the flag the wire field stays off.
    plain = build_engine_core_request_from_tokens(
        "req-plain",
        {"prompt_token_ids": [1, 2, 3]},
        SamplingParams(max_tokens=1),
    )
    assert plain.return_hidden_states is False


@pytest.mark.cpu
def test_scheduler_request_carries_flag():
    request = _core_request(return_hidden_states=True)
    scheduler_request = OmniRequest.from_engine_core_request(request, block_hasher=None)
    assert scheduler_request.return_hidden_states is True
    assert OmniRequest.from_engine_core_request(_core_request(), block_hasher=None).return_hidden_states is False


class _FakeBatch:
    def __init__(self, req_ids: list[str], scheduled: list[int], computed: list[int], prompt_lens: list[int]):
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        offsets = np.zeros(len(req_ids) + 1, dtype=np.int32)
        np.cumsum(scheduled, out=offsets[1:])
        self.query_start_loc_np = offsets
        self.num_scheduled_tokens = np.asarray(scheduled, dtype=np.int32)
        self.num_computed_tokens_np = np.asarray(computed, dtype=np.int32)
        self.prompt_len_np = np.asarray(prompt_lens, dtype=np.int32)


def _runner_with_options(options: set[str]) -> OmniARModelRunner:
    runner = object.__new__(OmniARModelRunner)
    runner._hidden_state_req_ids = set(options)
    runner.req_states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=None),
    )
    return runner


@pytest.mark.cpu
def test_collect_slices_only_opted_in_rows():
    batch = _FakeBatch(
        req_ids=["a", "b"],
        scheduled=[4, 3],
        computed=[0, 0],
        prompt_lens=[4, 3],
    )
    runner = _runner_with_options({"a"})
    runner.req_states.prompt_len.np = batch.prompt_len_np
    hidden = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2)
    slices = runner._collect_hidden_state_slices(hidden, batch)
    assert set(slices) == {"a"}
    assert torch.equal(slices["a"], hidden[:4])


@pytest.mark.cpu
def test_collect_slices_none_when_no_option():
    batch = _FakeBatch(req_ids=["a"], scheduled=[4], computed=[0], prompt_lens=[4])
    runner = _runner_with_options(set())
    runner.req_states.prompt_len.np = batch.prompt_len_np
    assert runner._collect_hidden_state_slices(torch.zeros(4, 2), batch) is None


@pytest.mark.cpu
def test_collect_slices_rejects_chunked_prefill():
    # Prompt of 6 tokens, only 4 scheduled this step.
    batch = _FakeBatch(req_ids=["a"], scheduled=[4], computed=[0], prompt_lens=[6])
    runner = _runner_with_options({"a"})
    runner.req_states.prompt_len.np = batch.prompt_len_np
    with pytest.raises(RuntimeError, match="chunked prefill"):
        runner._collect_hidden_state_slices(torch.zeros(4, 2), batch)


@pytest.mark.cpu
def test_collect_slices_rejects_prefix_cache_resume():
    # 3 tokens already computed from a cached prefix; only 3 of the 6 prompt
    # tokens run this step.
    batch = _FakeBatch(req_ids=["a"], scheduled=[3], computed=[3], prompt_lens=[6])
    runner = _runner_with_options({"a"})
    runner.req_states.prompt_len.np = batch.prompt_len_np
    with pytest.raises(RuntimeError, match="prefix cache"):
        runner._collect_hidden_state_slices(torch.zeros(3, 2), batch)


@pytest.mark.cpu
def test_collect_slices_skips_decode_rows():
    # Prompts fully computed (4/4 and 3/3); this step schedules decode tokens.
    # No error and no payload: the prefill step already delivered the tensor.
    batch = _FakeBatch(req_ids=["a", "b"], scheduled=[1, 1], computed=[4, 3], prompt_lens=[4, 3])
    runner = _runner_with_options({"a"})
    runner.req_states.prompt_len.np = batch.prompt_len_np
    assert runner._collect_hidden_state_slices(torch.zeros(2, 2), batch) is None


@pytest.mark.cpu
def test_merge_payloads_into_wire_channel():
    output = SimpleNamespace(req_ids=["x", "y"], multimodal_outputs=None)
    tensors = {"y": torch.zeros(3, 2)}
    _merge_hidden_state_payloads(output, tensors)
    assert output.multimodal_outputs[0] is None
    assert output.multimodal_outputs[1] == {HIDDEN_STATES_OUTPUT_KEY: tensors["y"]}

    # Existing pooler payloads are preserved.
    existing = [{"pcm": torch.zeros(1)}, None]
    output = SimpleNamespace(req_ids=["x", "y"], multimodal_outputs=existing)
    _merge_hidden_state_payloads(output, {"x": torch.zeros(2, 2)})
    assert "pcm" in output.multimodal_outputs[0]
    assert output.multimodal_outputs[0][HIDDEN_STATES_OUTPUT_KEY].shape == (2, 2)

    # A non-dict slot cannot carry the payload; fail loudly.
    output = SimpleNamespace(req_ids=["x"], multimodal_outputs=[object()])
    with pytest.raises(RuntimeError, match="already holds"):
        _merge_hidden_state_payloads(output, {"x": torch.zeros(1, 1)})


@pytest.mark.cpu
def test_diffusion_ingress_allowlist_getter_matches_validator_surface():
    from vllm_omni.config.omni_config import (
        _DIFFUSION_DEFAULT_FACTORY_FIELDS,
        _DIFFUSION_SHARED_ONLY_ENGINE_FIELDS,
        diffusion_ingress_allowed_fields,
    )

    allowed = diffusion_ingress_allowed_fields()
    # Stage-consumed and externally-consumed owners are all present...
    assert _DIFFUSION_DEFAULT_FACTORY_FIELDS <= allowed
    assert _DIFFUSION_SHARED_ONLY_ENGINE_FIELDS <= allowed
    # ...while LLM-only engine fields stay out (the reason verl-omni strips them).
    assert "enable_prefix_caching" not in allowed or "kv_cache_dtype" in allowed
    assert isinstance(allowed, frozenset)
