# SPDX-License-Identifier: Apache-2.0
"""Runner-side validation tests for the multi-step decode window.

``validate_multi_step_plan`` re-checks every window invariant against
runner-local state before the K-step replay runs (fail-closed: any
refusal falls back to the normal single-step path).  These tests drive
the validator with duck-typed fakes, no NPU device required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vllm_omni.platforms.npu.worker.multi_step_decode as msd_module
from vllm_omni.platforms.npu.worker.multi_step_decode import validate_multi_step_plan


@pytest.fixture(autouse=True)
def _stub_parallel_groups(monkeypatch):
    """Single-process stand-ins for the distributed state accessors."""
    group = SimpleNamespace(world_size=1, is_last_rank=True)
    monkeypatch.setattr(msd_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(msd_module, "get_tp_group", lambda: group)


def make_scheduler_output(req_ids, *, window_k=8, scheduled=None):
    scheduled = scheduled or {rid: window_k for rid in req_ids}
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_scheduled_tokens=scheduled,
        multi_step_plan={rid: window_k for rid in req_ids},
    )


def make_runner(req_ids, *, model=None, input_batch=None, **overrides):
    num_reqs = len(req_ids)
    runner = SimpleNamespace(
        use_async_scheduling=True,
        num_spec_tokens=0,
        speculative_config=None,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
        ),
        pcp_size=1,
        dcp_size=1,
        use_cp=False,
        lora_config=None,
        is_pooling_model=False,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            enable_return_routed_experts=False,
            enforce_eager=False,
        ),
        supports_mm_inputs=False,
        omni_prefix_cache=None,
        cache_config=SimpleNamespace(mamba_cache_mode="off"),
        calculate_kv_scales=False,
        dynamic_eplb=False,
        debugger=None,
        ascend_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(enabled=False),
        ),
        model=model or make_model(),
        input_batch=input_batch or make_input_batch(req_ids),
        _resolve_duplex_sampling_hook=lambda: None,
        num_prompt_logprobs=0,
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def make_model(**overrides):
    model = SimpleNamespace(
        has_preprocess=True,
        make_omni_output=lambda *a, **k: None,
        prefer_model_sampler=False,
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def make_input_batch(req_ids):
    num_reqs = len(req_ids)
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=num_reqs,
        num_computed_tokens_cpu=np.full(num_reqs, 200, dtype=np.int32),
        num_prompt_tokens=np.full(num_reqs, 100, dtype=np.int32),
        sampling_metadata=SimpleNamespace(
            no_penalties=True,
            max_num_logprobs=None,
        ),
        bad_words_token_ids=None,
        no_allowed_token_ids=True,
        logprob_token_ids=None,
    )


def test_validate_accepts_eligible_window():
    runner = make_runner(["r0", "r1"])
    output = make_scheduler_output(["r0", "r1"])
    assert validate_multi_step_plan(runner, output) == {"r0": 8, "r1": 8}


def test_validate_refuses_missing_or_empty_plan():
    runner = make_runner(["r0"])
    output = make_scheduler_output(["r0"])
    output.multi_step_plan = {}
    assert validate_multi_step_plan(runner, output) is None
    output.multi_step_plan = None
    assert validate_multi_step_plan(runner, output) is None


def test_validate_refuses_non_uniform_window():
    runner = make_runner(["r0", "r1"])
    output = make_scheduler_output(["r0", "r1"])
    output.multi_step_plan = {"r0": 8, "r1": 4}
    output.num_scheduled_tokens = {"r0": 8, "r1": 4}
    assert validate_multi_step_plan(runner, output) is None


def test_validate_refuses_mismatched_schedule():
    # num_scheduled_tokens disagrees with the plan (scheduler bug or a
    # rewritten output that was not meant for this runner).
    runner = make_runner(["r0"])
    output = make_scheduler_output(["r0"])
    output.num_scheduled_tokens = {"r0": 1}
    assert validate_multi_step_plan(runner, output) is None


def test_validate_refuses_sync_scheduling():
    runner = make_runner(["r0"], use_async_scheduling=False)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_validate_refuses_spec_decode():
    runner = make_runner(["r0"], num_spec_tokens=2)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_validate_refuses_model_without_contract():
    runner = make_runner(["r0"], model=make_model(has_preprocess=False))
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_validate_refuses_penalties():
    input_batch = make_input_batch(["r0"])
    input_batch.sampling_metadata.no_penalties = False
    runner = make_runner(["r0"], input_batch=input_batch)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_accepts_stale_input_batch_views():
    # The runner-side validation must NOT re-check per-step mutable state:
    # under async scheduling the input_batch view lags the scheduler (the
    # plan is validated before _update_states syncs it), so a stale view
    # would produce false refusals.  A refusal after the scheduler planned
    # follow-up windows on top of this one is not recoverable -- the
    # follow-up plans' KV positions would skip this window's reserved
    # slots.  Phase and batch composition are gated scheduler-side with
    # authoritative state instead.
    input_batch = make_input_batch(["r0"])
    input_batch.num_computed_tokens_cpu[0] = 50  # stale prefill view
    runner = make_runner(["r0"], input_batch=input_batch)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is not None


def test_accepts_plan_covering_requests_missing_from_input_batch():
    runner = make_runner(["r0"])
    output = make_scheduler_output(["r0", "r1"])
    output.multi_step_plan = {"r0": 8, "r1": 8}
    output.num_scheduled_tokens = {"r0": 8, "r1": 8}
    # input_batch still shows only r0 (async lag); the plan is authoritative.
    assert validate_multi_step_plan(runner, output) is not None


def test_validate_returns_none_on_internal_error():
    class ExplodingRunner(SimpleNamespace):
        # any attribute access on use_async_scheduling raises
        def __getattribute__(self, name):
            if name == "use_async_scheduling":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

    runner = ExplodingRunner()
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None
