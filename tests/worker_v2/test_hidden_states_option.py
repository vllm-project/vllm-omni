# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests for the per-request hidden-state opt-in (RL teacher rollouts).

Covers the typed wire field from the engine-core request to the scheduler's
new-request data, the runner-side prefill-row slicing guards, and the
wire-channel merge. The opt-in lift in ``StageEngineCoreProc`` is covered in
``tests/engine/test_stage_engine_core_proc.py``, the opt-in bookkeeping in
``OmniGPUModelRunner.add_requests`` in ``test_omni_gpu_model_runner.py``, and
the diffusion ingress getters in ``tests/config/test_omni_config.py``.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.data_entry_keys import _NESTED_KEYS
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.request import OmniRequest
from vllm_omni.worker_v2.omni_ar_model_runner import (
    TEACHER_HIDDEN_STATES_KEY,
    OmniARModelRunner,
    _merge_hidden_state_payloads,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


# ---------------------------------------------------------------------------
# Wire field
# ---------------------------------------------------------------------------


def test_wire_field_defaults_off():
    request = _core_request()
    assert request.return_hidden_states is False


def test_from_request_round_trip():
    base = _core_request(return_hidden_states=True)
    clone = OmniEngineCoreRequest.from_request(base)
    assert clone.return_hidden_states is True
    # An explicit override wins over the source attribute.
    suppressed = OmniEngineCoreRequest.from_request(base, return_hidden_states=False)
    assert suppressed.return_hidden_states is False


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


def test_scheduler_request_carries_flag():
    request = _core_request(return_hidden_states=True)
    scheduler_request = OmniRequest.from_engine_core_request(request, block_hasher=None)
    assert scheduler_request.return_hidden_states is True
    assert OmniRequest.from_engine_core_request(_core_request(), block_hasher=None).return_hidden_states is False


def test_scheduler_new_request_data_carries_flag():
    # The scheduler rewraps vLLM's NewRequestData; the runner reads the flag there.
    data = NewRequestData(
        req_id="req-1",
        prompt_token_ids=np.asarray([1, 2, 3], dtype=np.int32),
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_ids=((),),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[1, 2, 3],
    )
    request = OmniRequest.from_engine_core_request(_core_request(return_hidden_states=True), block_hasher=None)
    assert OmniNewRequestData.from_base(data, request).return_hidden_states is True
    assert OmniNewRequestData.from_base(data, None).return_hidden_states is False


# ---------------------------------------------------------------------------
# Runner: prefill-row slicing
# ---------------------------------------------------------------------------


class _FakeBatch:
    """Batch layout for one step: ``[qsl[i], qsl[i] + scheduled[i])`` are req i's rows."""

    def __init__(self, req_ids: list[str], scheduled: list[int], computed_prefill: list[int], prefill_len: list[int]):
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        offsets = np.zeros(len(req_ids) + 1, dtype=np.int32)
        np.cumsum(scheduled, out=offsets[1:])
        self.query_start_loc_np = offsets
        self.num_scheduled_tokens = np.asarray(scheduled, dtype=np.int32)
        self.num_computed_prefill_tokens_np = np.asarray(computed_prefill, dtype=np.int32)
        self.prefill_len_np = np.asarray(prefill_len, dtype=np.int32)


def _runner_with_options(options: set[str]) -> OmniARModelRunner:
    runner = object.__new__(OmniARModelRunner)
    runner._hidden_state_req_ids = set(options)
    return runner


def test_collect_slices_only_opted_in_rows_of_a_mixed_batch():
    batch = _FakeBatch(
        req_ids=["a", "b", "c"],
        scheduled=[4, 3, 2],
        computed_prefill=[0, 0, 0],
        prefill_len=[4, 3, 2],
    )
    hidden = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)
    slices = _runner_with_options({"a", "c"})._collect_hidden_state_slices(hidden, batch)
    assert set(slices) == {"a", "c"}
    assert torch.equal(slices["a"], hidden[0:4])
    assert torch.equal(slices["c"], hidden[7:9])


def test_collect_slices_none_when_nothing_opted_in():
    batch = _FakeBatch(req_ids=["a"], scheduled=[4], computed_prefill=[0], prefill_len=[4])
    assert _runner_with_options(set())._collect_hidden_state_slices(torch.zeros(4, 2), batch) is None


def test_collect_slices_cuts_prefill_rows_when_more_are_scheduled():
    # One extra scheduled row (e.g. a draft); the payload is the prefill rows only.
    batch = _FakeBatch(req_ids=["a"], scheduled=[6], computed_prefill=[0], prefill_len=[5])
    hidden = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    assert torch.equal(_runner_with_options({"a"})._collect_hidden_state_slices(hidden, batch)["a"], hidden[:5])


def test_collect_slices_skips_decode_rows():
    # Prefill done (4/4 and 3/3); this step schedules one decode token each.
    # No error and no payload: the prefill step already delivered the tensor.
    batch = _FakeBatch(req_ids=["a", "b"], scheduled=[1, 1], computed_prefill=[4, 3], prefill_len=[4, 3])
    assert _runner_with_options({"a", "b"})._collect_hidden_state_slices(torch.zeros(2, 2), batch) is None


def test_collect_slices_uses_expanded_prefill_len_for_multimodal_prompts():
    # Multimodal placeholders expand, so prefill_len counts tokens the runner
    # sees. The payload must cover all of them, not the unexpanded id count.
    batch = _FakeBatch(req_ids=["a"], scheduled=[12], computed_prefill=[0], prefill_len=[12])
    hidden = torch.arange(12 * 2, dtype=torch.float32).reshape(12, 2)
    assert torch.equal(_runner_with_options({"a"})._collect_hidden_state_slices(hidden, batch)["a"], hidden)


def test_collect_slices_rejects_first_chunk_of_chunked_prefill():
    # 12 expanded prefill tokens, only 5 scheduled this step.
    batch = _FakeBatch(req_ids=["a"], scheduled=[5], computed_prefill=[0], prefill_len=[12])
    with pytest.raises(RuntimeError, match="chunked prefill"):
        _runner_with_options({"a"})._collect_hidden_state_slices(torch.zeros(5, 2), batch)


@pytest.mark.parametrize("computed", [3, 5])
def test_collect_slices_rejects_later_prefill_chunk_or_prefix_hit(computed):
    # Part of the prefill was computed before this step (a later chunk or a
    # prefix-cache hit): this step's tensor cannot supply the full prompt.
    batch = _FakeBatch(req_ids=["a"], scheduled=[12 - computed], computed_prefill=[computed], prefill_len=[12])
    with pytest.raises(RuntimeError, match="computed before this step"):
        _runner_with_options({"a"})._collect_hidden_state_slices(torch.zeros(12 - computed, 2), batch)


@pytest.mark.parametrize("missing", ["prefill_len_np", "num_computed_prefill_tokens_np"])
@pytest.mark.parametrize("row", ["prefill", "decode"])
def test_collect_slices_fails_closed_without_prefill_counters(missing, row):
    # No prefill counters means a decode row cannot be told from a prefill row;
    # never emit a slice on a guess, on either kind of step.
    batch = _FakeBatch(req_ids=["a"], scheduled=[4], computed_prefill=[0], prefill_len=[4])
    if row == "decode":
        batch.num_computed_prefill_tokens_np = np.asarray([4], dtype=np.int32)
    delattr(batch, missing)
    with pytest.raises(RuntimeError, match="needs InputBatch.prefill_len_np"):
        _runner_with_options({"a"})._collect_hidden_state_slices(torch.zeros(4, 2), batch)


# ---------------------------------------------------------------------------
# Wire channel
# ---------------------------------------------------------------------------


def test_wire_key_is_not_a_reserved_payload_nest():
    assert TEACHER_HIDDEN_STATES_KEY == "teacher_hidden_states"
    assert TEACHER_HIDDEN_STATES_KEY not in _NESTED_KEYS
    assert "." not in TEACHER_HIDDEN_STATES_KEY


def test_merge_payloads_into_wire_channel():
    output = SimpleNamespace(req_ids=["x", "y"], multimodal_outputs=None)
    tensors = {"y": torch.zeros(3, 2)}
    _merge_hidden_state_payloads(output, tensors)
    assert output.multimodal_outputs[0] is None
    assert output.multimodal_outputs[1] == {TEACHER_HIDDEN_STATES_KEY: tensors["y"]}


def test_merge_keeps_pooler_payload_in_the_same_slot():
    # A thinker->talker stage publishes its reserved hidden_states.* nest in the
    # same slot; the teacher payload is added beside it, not over it.
    pooler = {"hidden_states.output": torch.zeros(2, 4), "embed.prefill": torch.zeros(2, 4)}
    output = SimpleNamespace(req_ids=["x"], multimodal_outputs=[pooler])
    teacher = torch.zeros(2, 2)
    _merge_hidden_state_payloads(output, {"x": teacher})
    merged = output.multimodal_outputs[0]
    assert merged[TEACHER_HIDDEN_STATES_KEY] is teacher
    assert set(merged) == {"hidden_states.output", "embed.prefill", TEACHER_HIDDEN_STATES_KEY}
    assert merged["hidden_states.output"] is pooler["hidden_states.output"]
    # The caller's dict is left untouched.
    assert TEACHER_HIDDEN_STATES_KEY not in pooler


def test_merge_rejects_slot_that_cannot_carry_the_payload():
    output = SimpleNamespace(req_ids=["x"], multimodal_outputs=[object()])
    with pytest.raises(RuntimeError, match="already holds"):
        _merge_hidden_state_payloads(output, {"x": torch.zeros(1, 1)})


# ---------------------------------------------------------------------------
# Diffusion ingress allowlist
# ---------------------------------------------------------------------------


def test_diffusion_ingress_allowlist_excludes_llm_only_engine_fields():
    from vllm_omni.config import omni_config as omni_config_module

    allowed = omni_config_module.diffusion_ingress_allowed_fields()
    # LLM-only engine args have no owner on a diffusion stage; this is the set
    # external embedders strip before submitting diffusion engine kwargs.
    for llm_only in ("enable_lora", "max_loras", "max_lora_rank", "block_size", "swap_space", "preemption_mode"):
        assert llm_only not in allowed, llm_only
    # Diffusion-owned and externally consumed fields stay accepted.
    assert {"dtype", "cache_backend", "enable_prefix_caching", "seed", "max_model_len", "omni_master_port"} <= allowed
