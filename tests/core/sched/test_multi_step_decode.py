# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the multi-step decode window (PR: multi-step merged decode).

Pure-CPU tests for the scheduler-side planner in
``vllm_omni.core.sched.multi_step_decode``: window-size resolution
(config + env override), the static capability gate, the schedule-output
rewrite (request admission, budget clamp, KV slot allocation, in-flight
fence accounting) and the shortfall reconcile that keeps
``num_computed_tokens`` consistent with the KV slots actually written
when a window exits early or falls back to single steps.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.multi_step_decode as msd
from vllm_omni.core.sched.multi_step_decode import (
    ENV_MULTI_STEP_STEPS,
    MAX_MULTI_STEP_STEPS,
    model_supports_multi_step,
    plan_multi_step_window,
    reconcile_window_shortfall,
    resolve_multi_step_steps,
    scheduler_allows_multi_step,
)

# The one architecture that declares ``supports_multi_step_decode`` today.
WINDOWED_ARCH = "MiniCPMO45OmniTTSForConditionalGeneration"


@pytest.fixture(autouse=True)
def _windowed_platform(monkeypatch):
    """Pin the platform gate to a window-capable (NPU) platform.

    The window executor ships with the NPU AR runner, so the static gate
    refuses other platforms.  These tests exercise the gate's *other*
    conditions, so pin device_type to "npu" regardless of the host machine.
    """
    monkeypatch.setattr(msd, "current_platform", SimpleNamespace(device_type="npu"))


# ---------------------------------------------------------------- fakes ----


def make_sampling_params(**overrides):
    params = SimpleNamespace(
        logprobs=None,
        prompt_logprobs=None,
        bad_words_token_ids=None,
        allowed_token_ids=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=2048,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def make_request(req_id="r0", *, computed=None, prompt=100, output_len=5, max_tokens=2048):
    # ``computed`` models the post-base-schedule state: for steady decode the
    # async scheduler has optimistically counted prompt + output_len tokens.
    if computed is None:
        computed = prompt + output_len
    request = SimpleNamespace(
        request_id=req_id,
        status=RequestStatus.RUNNING,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=prompt + output_len,
        num_output_placeholders=1,
        output_token_ids=[0] * output_len,
        has_encoder_inputs=False,
        use_structured_output=False,
        pooling_params=None,
        sampling_params=make_sampling_params(max_tokens=max_tokens),
        is_finished=lambda: False,
    )
    return request


def make_kv_manager(fail_ids=()):
    """allocate_slots fake: extends blocks unless the request id is in fail_ids."""

    def allocate_slots(request, num_new_tokens, num_lookahead_tokens=0):
        if request.request_id in fail_ids:
            return None
        blocks = ([f"blk-{request.request_id}-{num_new_tokens}"],)
        request.allocated = getattr(request, "allocated", 0) + num_new_tokens
        return SimpleNamespace(get_block_ids=lambda ids=blocks: ids)

    return SimpleNamespace(allocate_slots=allocate_slots)


def make_scheduler_output(requests, *, scheduled=1):
    req_ids = [r.request_id for r in requests]
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=req_ids,
            new_block_ids=[None] * len(req_ids),
        ),
        num_scheduled_tokens={rid: scheduled for rid in req_ids},
        total_num_scheduled_tokens=scheduled * len(requests),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        multi_step_plan={},
    )


def make_scheduler(requests, *, max_model_len=4096, fail_ids=()):
    return SimpleNamespace(
        requests={r.request_id: r for r in requests},
        kv_cache_manager=make_kv_manager(fail_ids=fail_ids),
        max_model_len=max_model_len,
    )


def make_model_config(**overrides):
    config = SimpleNamespace(
        architectures=[WINDOWED_ARCH],
        model_impl="vllm",
        is_encoder_decoder=False,
        enable_return_routed_experts=False,
        multi_step_decode_steps=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_vllm_config(model_config=None, **overrides):
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pcp_size=1,
        dcp_size=1,
    )
    config = SimpleNamespace(
        model_config=model_config or make_model_config(),
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="off"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_configured_scheduler(**scheduler_overrides):
    scheduler = SimpleNamespace(
        vllm_config=make_vllm_config(),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        num_spec_tokens=0,
        kv_transfer_criteria=None,
        lora_config=None,
        max_model_len=4096,
        waiting=[],
    )
    for key, value in scheduler_overrides.items():
        setattr(scheduler, key, value)
    return scheduler


# ------------------------------------------------- window-size resolution --


def test_resolve_steps_reads_model_config(monkeypatch):
    monkeypatch.delenv(ENV_MULTI_STEP_STEPS, raising=False)
    assert resolve_multi_step_steps(make_model_config(multi_step_decode_steps=8)) == 8
    assert resolve_multi_step_steps(make_model_config(multi_step_decode_steps=0)) == 0


def test_resolve_steps_env_overrides_config(monkeypatch):
    monkeypatch.setenv(ENV_MULTI_STEP_STEPS, "4")
    assert resolve_multi_step_steps(make_model_config(multi_step_decode_steps=8)) == 4
    monkeypatch.setenv(ENV_MULTI_STEP_STEPS, "0")
    assert resolve_multi_step_steps(make_model_config(multi_step_decode_steps=8)) == 0


def test_resolve_steps_clamps_to_max(monkeypatch):
    monkeypatch.delenv(ENV_MULTI_STEP_STEPS, raising=False)
    assert resolve_multi_step_steps(make_model_config(multi_step_decode_steps=99)) == MAX_MULTI_STEP_STEPS


def test_resolve_steps_invalid_env_falls_back_to_config(monkeypatch):
    monkeypatch.setenv(ENV_MULTI_STEP_STEPS, "not-a-number")
    assert resolve_multi_step_steps(make_model_config(multi_step_decode_steps=6)) == 6


# ----------------------------------------------------- capability gate -----


def test_model_capability_resolved_through_registry():
    assert model_supports_multi_step(make_model_config()) is True


def test_model_capability_refused_for_undeclared_arch():
    assert model_supports_multi_step(make_model_config(architectures=["SomeOtherArch"])) is False


def test_model_capability_stage_qualified_wrapper():
    """A wrapper arch backing several stages qualifies only for listed stages.

    Regression: the omni wrapper class is what the registry resolves for the
    talker stage, so a flag declared only on the inner stage model left the
    gate permanently closed.  The wrapper declares capability per stage.
    """
    WRAPPER = "MiniCPMO45OmniForConditionalGeneration"
    tts = make_model_config(architectures=[WRAPPER], model_stage="tts")
    llm = make_model_config(architectures=[WRAPPER], model_stage="llm")
    assert model_supports_multi_step(tts) is True
    assert model_supports_multi_step(llm) is False


def test_model_capability_stage_flag_beats_blanket_flag():
    """``supports_multi_step_stages`` takes precedence over a blanket flag."""
    WRAPPER = "MiniCPMO45OmniForConditionalGeneration"
    llm = make_model_config(architectures=[WRAPPER], model_stage="llm")
    assert model_supports_multi_step(llm) is False


def test_scheduler_allows_multi_step_platform_refusal(monkeypatch):
    """Platforms without the window executor must be refused (fail-closed)."""
    monkeypatch.setattr(msd, "current_platform", SimpleNamespace(device_type="cuda"))
    assert scheduler_allows_multi_step(make_configured_scheduler()) is False


def test_scheduler_allows_multi_step_positive():
    assert scheduler_allows_multi_step(make_configured_scheduler()) is True


def test_scheduler_allows_multi_step_refusals():
    refusals = [
        {"num_spec_tokens": 2},
        {"kv_transfer_criteria": {"type": "prefill_finished"}},
        {"lora_config": object()},
        {"scheduler_config": SimpleNamespace(async_scheduling=False)},
        {"vllm_config": make_vllm_config(make_model_config(is_encoder_decoder=True))},
        {
            "vllm_config": make_vllm_config(
                make_model_config(enable_return_routed_experts=True)
            )
        },
    ]
    for overrides in refusals:
        assert scheduler_allows_multi_step(make_configured_scheduler(**overrides)) is False, overrides


def test_scheduler_allows_multi_step_parallel_refusals():
    for field in ("pipeline_parallel_size", "tensor_parallel_size", "data_parallel_size", "pcp_size", "dcp_size"):
        scheduler = make_configured_scheduler()
        setattr(scheduler.vllm_config.parallel_config, field, 2)
        assert scheduler_allows_multi_step(scheduler) is False, field


def test_scheduler_allows_multi_step_mamba_refusal():
    scheduler = make_configured_scheduler()
    scheduler.vllm_config.cache_config.mamba_cache_mode = "align"
    assert scheduler_allows_multi_step(scheduler) is False


def test_scheduler_allows_multi_step_arch_refusal():
    scheduler = make_configured_scheduler()
    scheduler.vllm_config.model_config.architectures = ["SomeOtherArch"]
    assert scheduler_allows_multi_step(scheduler) is False


# -------------------------------------------------------- plan / patch ----


def test_plan_window_rewrites_single_decode_step():
    requests = [make_request("r0"), make_request("r1")]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 8) is True
    assert output.num_scheduled_tokens == {"r0": 8, "r1": 8}
    assert output.multi_step_plan == {"r0": 8, "r1": 8}
    assert output.total_num_scheduled_tokens == 16
    for request in requests:
        # K-1 extra placeholders + K-1 optimistic computed tokens on top of
        # the base single-token schedule.
        assert request.num_output_placeholders == 8
        assert request.num_computed_tokens == request.num_prompt_tokens + 5 + 7
        assert request.allocated == 7
    # New block rows communicated to the worker for both requests.
    assert all(row is not None for row in output.scheduled_cached_reqs.new_block_ids)


def test_plan_window_refused_when_waiting_queue_non_empty():
    # Admission gate: new work must be admitted before any window opens.
    requests = [make_request("r0")]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    scheduler.waiting = [make_request("r_wait")]
    assert plan_multi_step_window(scheduler, output, 8) is False
    assert output.multi_step_plan == {}
    assert requests[0].num_output_placeholders == 1


def test_plan_window_refused_when_new_reqs_scheduled():
    requests = [make_request("r0")]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    output.scheduled_new_reqs = [SimpleNamespace(req_id="new")]
    assert plan_multi_step_window(scheduler, output, 8) is False
    assert output.multi_step_plan == {}
    assert requests[0].num_output_placeholders == 1


def test_plan_window_refused_for_mixed_batch():
    # A prefill chunk scheduled 5 tokens for one request -> no uniform decode.
    requests = [make_request("r0"), make_request("r1")]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    output.num_scheduled_tokens["r1"] = 5
    assert plan_multi_step_window(scheduler, output, 8) is False


def test_plan_window_refused_for_prefill_phase_request():
    # num_computed_tokens < num_prompt_tokens => chunked prefill in flight.
    requests = [make_request("r0", computed=10, prompt=100)]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 8) is False


def test_plan_window_refused_for_logprobs_or_penalties():
    requests = [make_request("r0", max_tokens=2048)]
    requests[0].sampling_params.logprobs = 3
    scheduler = make_scheduler(requests)
    assert plan_multi_step_window(scheduler, make_scheduler_output(requests), 8) is False

    requests[0].sampling_params.logprobs = None
    requests[0].sampling_params.repetition_penalty = 1.2
    assert plan_multi_step_window(scheduler, make_scheduler_output(requests), 8) is False


def test_plan_window_budget_clamped_by_max_tokens():
    # Only 3 tokens remain before max_tokens -> K clamps to 3.
    requests = [make_request("r0", output_len=2045, max_tokens=2048)]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 8) is True
    assert output.multi_step_plan["r0"] == 3
    assert requests[0].num_output_placeholders == 3
    assert requests[0].num_computed_tokens == 100 + 2045 + 2


def test_plan_window_budget_clamped_by_model_len():
    # Context room leaves space for only 4 more tokens.
    requests = [make_request("r0", computed=4090, prompt=4085, output_len=5)]
    scheduler = make_scheduler(requests, max_model_len=4094)
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 8) is True
    assert output.multi_step_plan["r0"] == 4


def test_plan_window_refused_when_budget_below_two():
    requests = [make_request("r0", output_len=2048, max_tokens=2048)]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 8) is False


def test_plan_window_refused_when_allocation_fails_but_blocks_synced():
    # One request cannot host the extra slots: no window at all, yet the
    # successful sibling allocation is still communicated as look-ahead so
    # scheduler/worker block tables never diverge.
    requests = [make_request("r0"), make_request("r1")]
    scheduler = make_scheduler(requests, fail_ids={"r1"})
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 8) is False
    assert output.multi_step_plan == {}
    assert output.num_scheduled_tokens == {"r0": 1, "r1": 1}
    # r0's successful probe allocation is appended as a look-ahead row.
    assert output.scheduled_cached_reqs.new_block_ids[0] is not None
    assert output.scheduled_cached_reqs.new_block_ids[1] is None
    # Account state untouched on refusal.
    assert requests[0].num_output_placeholders == 1
    assert requests[1].num_output_placeholders == 1


def test_plan_window_preserves_confirmed_count_invariant():
    """The patch must preserve ``computed - placeholders == confirmed KV``.

    That identity is what keeps every subsequent schedule's
    ``num_new_tokens`` exact: while a window is in flight the async
    scheduler optimistically counts the K in-flight samples as
    placeholders, and each received token consumes exactly one
    placeholder, so the engine-side confirmed length always matches the
    KV slots actually written.
    """
    requests = [make_request("r0", output_len=5)]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    confirmed_before = requests[0].num_computed_tokens - requests[0].num_output_placeholders
    assert plan_multi_step_window(scheduler, output, 8) is True
    request = requests[0]
    assert request.num_computed_tokens - request.num_output_placeholders == confirmed_before
    # Both counters move by K-1; placeholders now hold the K in-flight
    # samples the window will produce (base 1 + patch 7).
    assert request.num_output_placeholders == 8
    assert request.num_computed_tokens == confirmed_before + 8
    # A subsequent schedule stays a normal single-token step: the base
    # formula num_tokens_with_spec + placeholders - computed == 1.
    num_new_tokens = request.num_tokens + request.num_output_placeholders - request.num_computed_tokens
    assert num_new_tokens == 1


def test_plan_window_disabled_for_k_below_two():
    requests = [make_request("r0")]
    scheduler = make_scheduler(requests)
    output = make_scheduler_output(requests)
    assert plan_multi_step_window(scheduler, output, 0) is False
    assert plan_multi_step_window(scheduler, output, 1) is False


# ------------------------------------------------------- reconcile --------


def test_reconcile_noop_when_window_fully_reported():
    request = make_request("r0", output_len=5)  # post-patch computed = 105 + 7
    request.num_computed_tokens = 105 + 7
    request.num_output_placeholders = 8
    reconcile_window_shortfall(request, 8, 8)
    assert request.num_output_placeholders == 8
    assert request.num_computed_tokens == 112


def test_reconcile_rolls_back_shortfall():
    # Runner produced 3 of the planned 8 tokens (early stop / fallback).
    # True computed after the window = (prompt + L - 1) + 3 = 104 + 3 = 107:
    # the window processed positions 104..106 of the 8 reserved slots.
    request = make_request("r0", output_len=5)
    request.num_output_placeholders = 5  # K - reported, consumed by the update
    request.num_computed_tokens = 112  # post-patch optimistic count
    reconcile_window_shortfall(request, 8, 3)
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == 107


def test_reconcile_clamps_to_available_placeholders():
    request = make_request("r0", output_len=5)
    request.num_output_placeholders = 1
    request.num_computed_tokens = 112
    reconcile_window_shortfall(request, 8, 3)
    # Only 1 placeholder was available; computed rolls back by the same
    # amount to preserve the confirmed-length invariant.
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == 111


def test_reconcile_handles_empty_report():
    request = make_request("r0", output_len=5)
    request.num_output_placeholders = 8
    request.num_computed_tokens = 112
    reconcile_window_shortfall(request, 8, 0)
    assert request.num_output_placeholders == 0
    # Nothing was processed: back to the pre-step confirmed count
    # (prompt + L - 1 = 104); the reserved slots stay allocated as
    # look-ahead and the next schedule re-plans a normal single step.
    assert request.num_computed_tokens == 104
