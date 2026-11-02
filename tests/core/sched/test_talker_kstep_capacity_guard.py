# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Capacity guard on the Talker K-step draft drop (PR #7929 review).

A busy stream keeps a request queued on most steps. Treating "something is
waiting" as "a prefill row joins this step" cleared the Talker's continuation
drafts on nearly every decode step, so a large share of steps ran the
single-frame path. vLLM only admits a waiting request while
``len(running) + num_waiting_for_streaming_input < max_num_running_reqs``
(vllm v1/core/sched/scheduler.py:864), so a full running batch admits
nothing and its uniform decode spans may keep the drafts.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(spec_token_ids: list[int], *, num_computed_tokens: int = 100, num_tokens: int = 101) -> SimpleNamespace:
    return SimpleNamespace(
        spec_token_ids=list(spec_token_ids),
        num_computed_tokens=num_computed_tokens,
        num_tokens=num_tokens,
        prompt_token_ids=[0] * 50,
    )


def _sched(*, running: list, waiting: list, max_num_running_reqs: int | None, num_spec_tokens: int = 7):
    sched = SimpleNamespace(
        _talker_kstep_armed=lambda: True,
        num_spec_tokens=num_spec_tokens,
        max_model_len=4096,
        num_waiting_for_streaming_input=0,
        running=running,
        waiting=waiting,
    )
    if max_num_running_reqs is not None:
        sched.max_num_running_reqs = max_num_running_reqs
    sched._talker_waiting_prefill_may_run = lambda: OmniARScheduler._talker_waiting_prefill_may_run(sched)
    return sched


def test_waiting_admitted_drops_the_drafts():
    """Capacity available: the waiting prefill joins the step, spans would be
    uneven, so the K-step drafts must be dropped."""
    running = [_req([1, 2, 3, 4, 5, 6, 7]), _req([1, 2, 3, 4, 5, 6, 7])]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=8)
    assert sched._talker_waiting_prefill_may_run() is True
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [] for req in running)


def test_full_running_batch_keeps_the_drafts():
    """Capacity full: nothing new is admitted, the step is a pure decode batch
    with uniform spans, and the K-step drafts survive."""
    running = [_req([1, 2, 3, 4, 5, 6, 7]) for _ in range(4)]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=4)
    assert sched._talker_waiting_prefill_may_run() is False
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [1, 2, 3, 4, 5, 6, 7] for req in running)


def test_waiting_for_streaming_input_counts_towards_capacity():
    """vLLM counts the streaming-waiting requests in the same guard, so they
    can fill the batch on their own."""
    running = [_req([1, 2, 3]) for _ in range(3)]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=4)
    sched.num_waiting_for_streaming_input = 1
    assert sched._talker_waiting_prefill_may_run() is False
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [1, 2, 3] for req in running)


def test_no_waiting_keeps_the_drafts():
    running = [_req([1, 2, 3, 4, 5, 6, 7])]
    sched = _sched(running=running, waiting=[], max_num_running_reqs=8)
    assert sched._talker_waiting_prefill_may_run() is False
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert running[0].spec_token_ids == [1, 2, 3, 4, 5, 6, 7]


def test_chunked_prefill_row_in_running_still_drops_the_drafts():
    """A running request that is still consuming its prompt (a streaming or
    chunked input) puts a foreign row width in the batch regardless of the
    waiting queue, so the drafts go."""
    running = [
        _req([1, 2, 3, 4, 5, 6, 7]),
        _req([1, 2, 3, 4, 5, 6, 7], num_computed_tokens=10, num_tokens=60),
    ]
    sched = _sched(running=running, waiting=[], max_num_running_reqs=8)
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [] for req in running)


def test_unknown_capacity_keeps_the_pre_drop_behaviour():
    """No capacity information: assume the waiting request joins, which is
    the direction that cannot feed the runner a mixed span."""
    running = [_req([1, 2, 3, 4, 5, 6, 7])]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=None)
    assert sched._talker_waiting_prefill_may_run() is True
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert running[0].spec_token_ids == []


def _dynamic_sched(*, device="cuda", eager=True):
    sched = _sched(running=[], waiting=[], max_num_running_reqs=16)
    sched.num_spec_tokens = 7
    from vllm.config import SpeculativeConfig
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=7,
        prompt_lookup_min=1,
        prompt_lookup_max=1,
        num_speculative_tokens_per_batch_size=[[1, 1, 7], [2, 2, 3], [3, 16, 0]],
    )
    sched.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
        config.num_speculative_tokens_per_batch_size, 16, config.num_speculative_tokens
    )
    sched.vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device_type=device),
        model_config=SimpleNamespace(enforce_eager=eager),
    )
    return sched


def _booked_output(widths):
    return SimpleNamespace(
        num_scheduled_tokens={rid: n + 1 for rid, n in enumerate(widths)},
        scheduled_spec_decode_tokens={rid: list(range(n)) for rid, n in enumerate(widths)},
        total_num_scheduled_tokens=sum(n + 1 for n in widths),
        num_spec_tokens_to_schedule=7,
    )


def test_dynamic_cuda_trims_current_step_not_only_next_proposal():
    sched = _dynamic_sched()
    output = _booked_output([7, 7])
    OmniARScheduler._enforce_kstep_span_uniformity(sched, output)
    assert output.num_scheduled_tokens == {0: 4, 1: 4}
    assert output.scheduled_spec_decode_tokens == {0: [0, 1, 2], 1: [0, 1, 2]}
    assert output.total_num_scheduled_tokens == 8
    assert output.num_spec_tokens_to_schedule == 3
    # The schedule() backstop must not change the already-booked output.
    OmniARScheduler._enforce_kstep_span_uniformity(sched, output)
    assert output.total_num_scheduled_tokens == 8


def test_dynamic_cuda_can_return_to_full_k_on_next_step():
    sched = _dynamic_sched()
    short = _booked_output([7, 7])
    OmniARScheduler._enforce_kstep_span_uniformity(sched, short)
    full = _booked_output([7])
    OmniARScheduler._enforce_kstep_span_uniformity(sched, full)
    assert full.num_scheduled_tokens == {0: 8}
    assert full.num_spec_tokens_to_schedule == 7


def test_dynamic_cuda_zero_drafts_books_single_frames():
    output = _booked_output([7, 7, 7])
    OmniARScheduler._enforce_kstep_span_uniformity(_dynamic_sched(), output)
    assert output.num_scheduled_tokens == {0: 1, 1: 1, 2: 1}
    assert output.scheduled_spec_decode_tokens == {}
    assert output.total_num_scheduled_tokens == 3
    assert output.num_spec_tokens_to_schedule == 0


def test_dynamic_cuda_insufficient_budget_keeps_single_frame_fallback():
    output = _booked_output([7, 2])
    OmniARScheduler._enforce_kstep_span_uniformity(_dynamic_sched(), output)
    assert output.num_scheduled_tokens == {0: 1, 1: 1}
    assert output.scheduled_spec_decode_tokens == {}


def test_async_mixed_folds_placeholder_and_draft_rows_preserving_prefill():
    """Mixed async steps fold decode rows, never truncate prompt extension."""
    sched = _dynamic_sched()
    sched._spec_token_placeholders = [-1] * 7
    output = _booked_output([7, 7, 7])
    # Request 0 is a real drafted decode row, request 1 is an async
    # placeholder-armed K-width row with no scheduler spec entry, and request 2
    # is a genuine prompt extension that must remain fully booked.
    output.scheduled_spec_decode_tokens = {0: list(range(7))}
    output.num_scheduled_tokens[2] = 20
    output.total_num_scheduled_tokens = 7 + 8 + 20

    OmniARScheduler._enforce_kstep_span_uniformity(sched, output)

    assert output.num_scheduled_tokens == {0: 1, 1: 1, 2: 20}
    assert output.scheduled_spec_decode_tokens == {}
    assert output.total_num_scheduled_tokens == 22


def test_dynamic_cuda_does_not_trim_prefill_rows():
    output = _booked_output([7])
    output.num_scheduled_tokens[1] = 20
    output.total_num_scheduled_tokens += 20
    OmniARScheduler._enforce_kstep_span_uniformity(_dynamic_sched(), output)
    assert output.num_scheduled_tokens == {0: 1, 1: 20}
    assert output.total_num_scheduled_tokens == 21


def test_dynamic_lookup_does_not_enable_narrow_static_graph_or_npu():
    for device, eager in [("cuda", False), ("npu", True)]:
        output = _booked_output([7, 7])
        OmniARScheduler._enforce_kstep_span_uniformity(_dynamic_sched(device=device, eager=eager), output)
        assert output.num_scheduled_tokens == {0: 8, 1: 8}
        assert output.num_spec_tokens_to_schedule == 7


def test_dynamic_width_is_booked_by_real_upstream_accounting():
    class RequestState(SimpleNamespace):
        __hash__ = object.__hash__

    sched = object.__new__(OmniARScheduler)
    fixture = _dynamic_sched()
    sched.vllm_config = fixture.vllm_config
    sched.dynamic_sd_lookup = fixture.dynamic_sd_lookup
    sched.num_spec_tokens = fixture.num_spec_tokens
    sched._talker_kstep_armed = lambda: True
    sched.defer_block_free = False
    sched._inflight_prefills = set()
    sched.requests = {
        rid: RequestState(
            num_computed_tokens=100,
            num_in_flight_tokens=0,
            num_tokens=101,
            num_output_placeholders=0,
            use_structured_output=False,
        )
        for rid in range(2)
    }
    output = _booked_output([7, 7])
    output.has_structured_output_requests = False
    sched._update_after_schedule(output)
    assert [r.num_computed_tokens for r in sched.requests.values()] == [104, 104]
    assert [r.num_in_flight_tokens for r in sched.requests.values()] == [4, 4]
    assert output.total_num_scheduled_tokens == 8
