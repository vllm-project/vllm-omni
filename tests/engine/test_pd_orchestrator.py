# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for PD metadata routing in the orchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import FinishReason

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.pd_continuation import PD_PREFILL_KEY, PD_RESUME_KEY, PDContinuation
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pd_orchestrator(kv_params: dict | None) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._pd_kv_params = {"req": kv_params} if kv_params is not None else {}
    orchestrator._pd_bootstrap_addr = "http://127.0.0.1:25201"
    orchestrator._pd_prefill_engine_id = "prefill-engine"
    return orchestrator


def test_pd_decode_params_allow_missing_prefill_output() -> None:
    orchestrator = _make_pd_orchestrator(None)
    result = orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))

    kv_params = result.extra_args["kv_transfer_params"]
    assert kv_params["transfer_id"] == "xfer-req"
    assert kv_params["remote_bootstrap_addr"] == "http://127.0.0.1:25201"
    assert kv_params["remote_engine_id"] == "prefill-engine"
    assert kv_params["do_remote_prefill"] is True
    assert kv_params["do_remote_decode"] is False
    assert "remote_request_id" not in kv_params


def test_pd_decode_params_preserve_optional_prefill_output() -> None:
    orchestrator = _make_pd_orchestrator(
        {
            "kv_ready": True,
            "remote_request_id": "prefill-request",
            "connector_metadata": "kept",
        }
    )
    result = orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))

    kv_params = result.extra_args["kv_transfer_params"]
    assert kv_params["remote_request_id"] == "prefill-request"
    assert kv_params["connector_metadata"] == "kept"


@pytest.mark.parametrize(
    ("attribute", "missing_field"),
    [
        ("_pd_prefill_engine_id", "remote_engine_id"),
        ("_pd_bootstrap_addr", "remote_bootstrap_addr"),
    ],
)
def test_pd_decode_params_require_mooncake_routing_fields(attribute: str, missing_field: str) -> None:
    orchestrator = _make_pd_orchestrator(None)
    setattr(orchestrator, attribute, None)

    with pytest.raises(RuntimeError, match=missing_field):
        orchestrator._build_pd_decode_params("req", SamplingParams(max_tokens=2))


@pytest.mark.parametrize("prompt_type", [dict, SimpleNamespace])
@pytest.mark.parametrize("ignore_eos", [False, True])
@pytest.mark.parametrize("generation_eos", [None, 3, [2, 3, 4]])
def test_pd_decode_inherits_processed_stop_metadata(prompt_type, ignore_eos, generation_eos):
    logical = SamplingParams(
        temperature=0,
        max_tokens=8,
        min_tokens=3,
        stop=["STOP"],
        stop_token_ids=[7],
        ignore_eos=ignore_eos,
        extra_args={PD_RESUME_KEY: True},
    )
    generation_config = {} if generation_eos is None else {"eos_token_id": generation_eos}
    # InputProcessor applies this normalization to its own clone, not to the
    # original stage sampling list retained by the orchestrator.
    non_pd = logical.clone()
    non_pd.update_from_generation_config(generation_config, eos_token_id=2)
    producer = PDDisaggregationMixin._prepare_prefill_sampling_params("req", logical)
    producer.update_from_generation_config(generation_config, eos_token_id=2)

    decode = _make_pd_orchestrator(None)._build_pd_decode_params("req", logical, prompt_type(sampling_params=producer))

    assert decode.eos_token_id == non_pd.eos_token_id
    assert set(decode.stop_token_ids) == set(non_pd.stop_token_ids)
    assert decode.all_stop_token_ids == non_pd.all_stop_token_ids
    assert 2 in decode.all_stop_token_ids  # Must be masked before min_tokens even with ignore_eos.
    assert (decode.max_tokens, decode.min_tokens, decode.stop, decode.detokenize) == (8, 3, ["STOP"], True)
    assert PD_PREFILL_KEY not in decode.extra_args
    assert decode.extra_args["kv_transfer_params"]["do_remote_prefill"] is True
    assert decode.extra_args["kv_transfer_params"]["do_remote_decode"] is False
    assert producer.stop == [] and not producer.detokenize
    assert producer.extra_args["kv_transfer_params"]["do_remote_decode"] is True
    assert logical.eos_token_id is None
    assert logical.stop_token_ids == [7] and logical.all_stop_token_ids == {7}
    assert logical.extra_args == {PD_RESUME_KEY: True}

    # Per-request stop sets/lists must not alias either the defaults or P.
    decode.stop_token_ids.append(99)
    decode.all_stop_token_ids.add(99)
    assert 99 not in producer.stop_token_ids and 99 not in producer.all_stop_token_ids
    assert 99 not in logical.stop_token_ids and 99 not in logical.all_stop_token_ids


@pytest.mark.parametrize("ignore_eos", [False, True])
@pytest.mark.parametrize("token", [2, 3, 7])
@pytest.mark.parametrize("first_token", [False, True])
def test_pd_decode_obeys_processed_eos_and_explicit_stop_tokens(ignore_eos, token, first_token):
    logical = SamplingParams(
        temperature=0,
        max_tokens=8,
        ignore_eos=ignore_eos,
        stop_token_ids=[7],
        extra_args={PD_RESUME_KEY: True},
    )
    producer = PDDisaggregationMixin._prepare_prefill_sampling_params("req", logical)
    producer.update_from_generation_config({"eos_token_id": [2, 3]}, eos_token_id=2)
    decode = _make_pd_orchestrator(None)._build_pd_decode_params(
        "req", logical, SimpleNamespace(sampling_params=producer)
    )
    request = OmniRequest(
        "req", [10, 11], decode, None, pd_continuation=PDContinuation(2, [token if first_token else 5])
    )
    if not first_token:
        request.append_output_token_ids(token)

    should_stop = not ignore_eos or token == 7  # Explicit user stops still apply with ignore_eos.
    assert check_stop(request, 4096) is should_stop
    if should_stop:
        assert request.get_finished_reason() == FinishReason.STOP


def test_legacy_pd_decode_keeps_its_logical_limits_and_stops():
    logical = SamplingParams(temperature=0, max_tokens=8, stop=["STOP"], stop_token_ids=[7])
    producer = PDDisaggregationMixin._prepare_prefill_sampling_params("req", logical)
    producer.update_from_generation_config({"eos_token_id": [2, 3]}, eos_token_id=2)
    decode = _make_pd_orchestrator(None)._build_pd_decode_params(
        "req", logical, SimpleNamespace(sampling_params=producer)
    )
    assert producer.max_tokens == 1 and producer.stop == []
    assert decode.max_tokens == 8 and decode.stop == ["STOP"] and decode.detokenize
    assert decode.eos_token_id == 2
    assert set(decode.stop_token_ids) == {3, 7}
    assert decode.all_stop_token_ids == {2, 3, 7}


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_pd_forward_carries_processed_stop_metadata(resume):
    logical = SamplingParams(
        temperature=0, max_tokens=8, stop_token_ids=[7], extra_args={PD_RESUME_KEY: True} if resume else None
    )
    producer = PDDisaggregationMixin._prepare_prefill_sampling_params("req", logical)
    producer.update_from_generation_config({"eos_token_id": [2, 3]}, eos_token_id=2)
    orchestrator = _make_pd_orchestrator(None)
    orchestrator._pd_pair = (0, 1)
    decode_pool = SimpleNamespace(
        stage_type="llm",
        stage_client=SimpleNamespace(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
        live_replica_ids=lambda: [0],
        submit_initial=AsyncMock(return_value=0),
    )
    orchestrator.stage_pools = [None, decode_pool]
    orchestrator._next_stage_already_submitted = Mock(return_value=False)
    orchestrator._record_duplex_stage_submission = Mock()
    orchestrator._emit_tx_edge = Mock()
    state = SimpleNamespace(
        sampling_params_list=[logical, logical],
        pd_prefill_prompt=SimpleNamespace(prompt_token_ids=[10, 11], sampling_params=producer),
        stage_submit_ts={},
    )
    output = SimpleNamespace(outputs=[SimpleNamespace(finish_reason="length", token_ids=[5])])

    await orchestrator._forward_to_next_stage_unguarded("req", 0, output, state)

    decode_pool.submit_initial.assert_awaited_once()
    decode_request = decode_pool.submit_initial.call_args.args[2]
    assert decode_request.prompt_token_ids == [10, 11]
    assert decode_request.sampling_params.eos_token_id == 2
    assert set(decode_request.sampling_params.stop_token_ids) == {3, 7}
    assert decode_request.sampling_params.all_stop_token_ids == {2, 3, 7}
    assert decode_request.sampling_params.max_tokens == 8
    assert logical.eos_token_id is None and logical.stop_token_ids == [7]
    assert decode_request.sampling_params.extra_args["kv_transfer_params"]["do_remote_prefill"] is True
    if resume:
        assert decode_request.pd_continuation.token_ids == [5]
    else:
        assert decode_request.pd_continuation is None
