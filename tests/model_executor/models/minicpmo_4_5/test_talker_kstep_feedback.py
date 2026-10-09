# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""No-device regression tests for the MiniCPM-o Talker K-step feedback chain.

Layers pinned down here:

- ``make_omni_output`` must record each frame's codec sample in the request
  state (``last_code``) while the multi-frame loop owns sampling: the rows
  vLLM schedules for the next step carry placeholder continue ids, so
  without the record every frame after the first embeds a placeholder
  instead of the real previous frame (silent audio corruption).
- The decode preprocess must prefer that state record over input_ids.
- The scheduler-side K-frame guard must drop continuation drafts whenever
  prefill work is pending, so no step ever mixes prefill rows with
  multi-row decode spans (the mixed batch that crashed a 64/4 run).
- The multi-frame gate matrix: non-uniform decode spans stay blocked at
  ``applies()``; the runner raise remains the assertion of last resort for
  a combination the guard makes unschedulable.
- The draft rule: a stop-truncated row (a request that accepted fewer
  tokens than the step ran) folds the whole batch to a draft-free step,
  the same way an empty row already did.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _codec_int_param,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_AUDIO_TOKENS = 6562
_EOS_ID = _NUM_AUDIO_TOKENS - 1


def _make_talker(*, k_step_frames: int, scripted_samples: list[int]):
    """Bare talker instance: no config, no weights, deterministic sampler.

    ``emb_code`` is crafted so the placeholder fallback (row 0) is all
    zeros while real codec rows are not -- a sharp contrast for asserting
    which id got embedded.
    """
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._k_step_frames = k_step_frames
    model.supports_multi_frame_decode = k_step_frames > 0
    model._codec_temperature = 0.0
    model._codec_min_tokens = 0
    model._codec_max_tokens = 4032
    model._num_audio_tokens = _NUM_AUDIO_TOKENS
    model._codec_eos_id = _EOS_ID
    model._request_audio_states = {}
    model._request_codec_history = {}
    model._request_generators = {}
    emb = nn.Embedding(_NUM_AUDIO_TOKENS, 4)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[42] = 1.0
        emb.weight[43] = 2.0
    model.emb_code = nn.ModuleList([emb])
    queue = iter(scripted_samples)

    def _greedy(_hidden, _codes, _request_id, _step, _min_tokens, _max_tokens, _eos_window_masked=False):
        return torch.tensor(next(queue), dtype=torch.long)

    model._sample_audio_code_greedy = _greedy
    return model


def _frame_call(model, hidden):
    return model.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "r1"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )


def test_kstep_frame_sample_is_recorded_for_next_frame():
    model = _make_talker(k_step_frames=8, scripted_samples=[42, 43, _EOS_ID])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state
    hidden = torch.randn(1, 8)

    out0 = _frame_call(model, hidden)
    assert state["last_code"] == 42
    assert state["step"] == 1
    assert out0.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [42]

    out1 = _frame_call(model, hidden)
    assert state["last_code"] == 43
    assert out1.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [43]

    # The EOS frame terminates the request: the record keeps the last real
    # sample (there is no next frame to feed), and the delta goes empty.
    out2 = _frame_call(model, hidden)
    assert state["last_code"] == 43
    assert state["finished"] is True
    assert out2.multimodal_outputs["codes"]["audio"][0].numel() == 0
    # Only confirmed frames are recorded: the terminating frame carries no
    # codec id for the next step to embed, so streaming prompt recompute must
    # not see it in the history either.
    assert model._request_codec_history["r1"] == [42, 43]


def test_legacy_single_frame_path_does_not_touch_last_code():
    model = _make_talker(k_step_frames=0, scripted_samples=[42])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state

    _frame_call(model, torch.randn(1, 8))
    assert "last_code" not in state


def test_decode_preprocess_embeds_state_last_code():
    model = _make_talker(k_step_frames=8, scripted_samples=[])
    state = {"step": 3, "codes": torch.tensor([42]), "last_code": 42}
    model._request_audio_states["r1"] = state
    # The scheduled row carries the placeholder continue id (0), not the
    # real previous frame's sample.
    input_ids = torch.zeros(1, dtype=torch.long)

    _, embeds, out = model.preprocess(
        input_ids,
        None,
        request_id="r1",
        audio_state=state,
        _omni_is_prefill=False,
    )
    assert torch.equal(embeds, model.emb_code[0](torch.tensor([42])))
    # Row 0 (the placeholder fallback) is all zeros in this fixture, so a
    # non-zero embed proves the state record won.
    assert torch.count_nonzero(embeds) == embeds.numel()
    assert out["codes"]["audio"].reshape(-1).tolist() == [42]


def _make_scheduler(*, num_spec: int, waiting=(), running=(), stage: str = "tts"):
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omni_talker_kstep_cache = None
    # The armed check reads the stage and the n-gram fingerprint off the engine
    # config; the real Scheduler exposes neither as a plain attribute.
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage),
        speculative_config=SimpleNamespace(method="ngram", num_speculative_tokens=num_spec),
    )
    # vLLM's Scheduler stores the count here (vllm_config.num_speculative_tokens);
    # the guard predicts row widths from it.
    sched.num_spec_tokens = num_spec
    sched.max_model_len = 40960
    sched.waiting = list(waiting)
    sched.running = list(running)
    return sched


def _req(*, computed: int, prompt: int, spec: list[int], total: int | None = None, req_id: str = "r0"):
    return SimpleNamespace(
        request_id=req_id,
        num_computed_tokens=computed,
        prompt_token_ids=[0] * prompt,
        # num_tokens is prompt + generated; the guard reads the difference to
        # predict how many rows this request will schedule this step.
        num_tokens=prompt if total is None else total,
        spec_token_ids=list(spec),
    )


def test_buffer_alignment_clears_a_stale_short_buffer():
    """A request whose draft buffer survived from an earlier state (a duplex
    takeover snapshot, a step whose schedule clipped it mid-batch) sits next
    to neighbours with full buffers; the next ``schedule()`` would book 8-
    and 4-wide spans in one step and the runner dies on the refusal. The
    alignment pass must clear every buffer so the step schedules a uniform
    padded single-frame span, from which the K-step resumes on its own."""
    good = _req(computed=100, prompt=100, spec=[0] * 7, total=108, req_id="good")
    stale = _req(computed=100, prompt=100, spec=[0] * 3, total=104, req_id="stale")
    sched = _make_scheduler(num_spec=7, waiting=[], running=[good, stale])

    sched.update_draft_token_ids(SimpleNamespace(req_ids=[], draft_token_ids=[]))
    assert good.spec_token_ids == []
    assert stale.spec_token_ids == []


def test_buffer_alignment_clears_a_zero_and_full_mix():
    """A full buffer next to an empty one is itself uneven: the next
    ``schedule()`` books a 1+num_spec-wide row for the full buffer next to
    a 1-wide row for the prefill chunk's empty one. The alignment pass must
    clear the full buffer so the step stays uniform."""
    good = _req(computed=100, prompt=100, spec=[0] * 7, total=108, req_id="good")
    chunk = _req(computed=50, prompt=100, spec=[], total=107, req_id="chunk")
    sched = _make_scheduler(num_spec=7, waiting=[], running=[good, chunk])

    sched.update_draft_token_ids(SimpleNamespace(req_ids=[], draft_token_ids=[]))
    assert good.spec_token_ids == []
    assert chunk.spec_token_ids == []


def test_buffer_alignment_keeps_a_single_value_batch():
    """Every buffer holding the same length -- all full, or all empty -- is
    uniform and must be left untouched."""
    full_a = _req(computed=100, prompt=100, spec=[0] * 7, total=108, req_id="a")
    full_b = _req(computed=110, prompt=100, spec=[0] * 7, total=118, req_id="b")
    sched = _make_scheduler(num_spec=7, waiting=[], running=[full_a, full_b])
    sched.update_draft_token_ids(SimpleNamespace(req_ids=[], draft_token_ids=[]))
    assert full_a.spec_token_ids == [0] * 7
    assert full_b.spec_token_ids == [0] * 7

    empty_a = _req(computed=100, prompt=100, spec=[], total=101, req_id="a")
    empty_b = _req(computed=110, prompt=100, spec=[], total=111, req_id="b")
    sched = _make_scheduler(num_spec=7, waiting=[], running=[empty_a, empty_b])
    sched.update_draft_token_ids(SimpleNamespace(req_ids=[], draft_token_ids=[]))
    assert empty_a.spec_token_ids == []
    assert empty_b.spec_token_ids == []


def _sched_out(num_scheduled: dict, spec: dict):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=spec,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
    )


def test_enforce_drops_uneven_decode_spans_to_single_frame():
    """The live repro: five decode requests leave upstream schedule() with
    span widths 4/5/5/6/6 (per-request spec frame counts 3/4/4/5/5 derived
    from each request's own token budget). The static-shape decode backend
    has no graph for a narrower K-step (10-09: a uniform 3-frame rewrite
    stalled the runner silently; stage 2 then waited 600s for chunks that
    never came), so the batch drops to a single-frame step -- the shape the
    backend ran at warmup -- and constant_drafts re-arms the K-step on the
    next draftless step."""
    reqs = {f"r{i}": r for i, r in enumerate([
        _req(computed=100, prompt=100, spec=[0] * 3, total=104, req_id="r0"),
        _req(computed=100, prompt=100, spec=[0] * 4, total=105, req_id="r1"),
        _req(computed=100, prompt=100, spec=[0] * 4, total=105, req_id="r2"),
        _req(computed=100, prompt=100, spec=[0] * 5, total=106, req_id="r3"),
        _req(computed=100, prompt=100, spec=[0] * 5, total=106, req_id="r4"),
    ])}
    sched = _make_scheduler(num_spec=5, waiting=[], running=list(reqs.values()))
    out = _sched_out(
        {"r0": 4, "r1": 5, "r2": 5, "r3": 6, "r4": 6},
        {"r0": [0] * 3, "r1": [0] * 4, "r2": [0] * 4, "r3": [0] * 5, "r4": [0] * 5},
    )

    sched._enforce_kstep_span_uniformity(out)

    assert out.num_scheduled_tokens == {r: 1 for r in reqs}
    # The dict itself must be emptied, not just the per-request lists: a
    # non-empty dict still builds spec-decode metadata and sends the
    # draftless step through the rejection sampler, which faults on NPU
    # (10-09: temp_q.exponential_ crash -> EngineDeadError).
    assert out.scheduled_spec_decode_tokens == {}
    assert out.total_num_scheduled_tokens == 5


def test_enforce_drops_drafts_when_a_plain_row_joins():
    """A width-1 decode row (cold or drained buffer) booked next to drafted
    rows is still uneven; the whole batch goes single-frame."""
    cold = _req(computed=100, prompt=100, spec=[], total=101, req_id="cold")
    armed_req = _req(computed=100, prompt=100, spec=[0] * 4, total=105, req_id="armed")
    sched = _make_scheduler(num_spec=4, waiting=[], running=[cold, armed_req])
    out = _sched_out({"cold": 1, "armed": 5}, {"armed": [0] * 4})

    sched._enforce_kstep_span_uniformity(out)

    assert out.num_scheduled_tokens == {"cold": 1, "armed": 1}
    assert out.scheduled_spec_decode_tokens == {}
    assert out.total_num_scheduled_tokens == 2


def test_enforce_drops_drafts_in_a_mixed_extend_step():
    """A streaming extend row (width > 1, no drafts) only tolerates
    single-frame decode rows beside it."""
    extend = _req(computed=100, prompt=100, spec=[], total=104, req_id="extend")
    armed_req = _req(computed=100, prompt=100, spec=[0] * 4, total=105, req_id="armed")
    sched = _make_scheduler(num_spec=4, waiting=[], running=[extend, armed_req])
    out = _sched_out({"extend": 4, "armed": 5}, {"armed": [0] * 4})

    sched._enforce_kstep_span_uniformity(out)

    assert out.num_scheduled_tokens == {"extend": 4, "armed": 1}
    assert out.scheduled_spec_decode_tokens == {}
    assert out.total_num_scheduled_tokens == 5


def test_enforce_keeps_a_uniform_batch_untouched():
    """A batch that already books uniform spans must pass through with no
    rewrite at all (no trim, no re-summary)."""
    reqs = [_req(computed=100 + i, prompt=100, spec=[0] * 4, total=105 + i, req_id=f"r{i}") for i in range(3)]
    sched = _make_scheduler(num_spec=4, waiting=[], running=reqs)
    out = _sched_out(
        {"r0": 5, "r1": 5, "r2": 5},
        {"r0": [0] * 4, "r1": [0] * 4, "r2": [0] * 4},
    )

    sched._enforce_kstep_span_uniformity(out)

    assert out.num_scheduled_tokens == {"r0": 5, "r1": 5, "r2": 5}
    assert out.total_num_scheduled_tokens == 15
    assert all(len(v) == 4 for v in out.scheduled_spec_decode_tokens.values())


def test_enforce_noop_when_not_armed():
    """Off the talker stage the pass must not rewrite upstream's output."""
    a = _req(computed=100, prompt=100, spec=[0] * 4, total=105, req_id="a")
    b = _req(computed=100, prompt=100, spec=[0] * 2, total=103, req_id="b")
    sched = _make_scheduler(num_spec=4, waiting=[], running=[a, b], stage="thinker")
    out = _sched_out({"a": 5, "b": 3}, {"a": [0] * 4, "b": [0] * 2})

    sched._enforce_kstep_span_uniformity(out)

    assert out.num_scheduled_tokens == {"a": 5, "b": 3}
    assert out.total_num_scheduled_tokens == 8


def test_guard_finishes_a_request_the_padded_width_cannot_fit():
    """A request too close to max_model_len to carry the padded 1+num_spec
    width schedules 1 row against the neighbours' 1+num_spec -- and padding
    does not depend on the drafts, so dropping drafts cannot defuse that
    mix; the runner would die on the refusal anyway. The starved request is
    within a handful of single-frame steps of the engine's own
    FINISHED_LENGTH_CAPPED, so the guard finishes it and the rest of the
    batch schedules a uniform span."""
    healthy = _req(computed=100, prompt=100, spec=[0] * 7, total=101, req_id="healthy")
    # 4090 + 1 + 7 + 1 > 4096: the padded width does not fit.
    starved = _req(computed=4090, prompt=100, spec=[0] * 7, total=4091, req_id="starved")
    sched = _make_scheduler(num_spec=7, waiting=[], running=[healthy, starved])
    sched.max_model_len = 4096
    finished = []
    sched.finish_requests = lambda ids, status: finished.append((ids, status))

    sched._drop_talker_drafts_if_prefill_pending()
    assert finished and finished[0][0] == ("starved",)
    assert starved not in sched.running
    # Only the starved request left: the batch is uniform again, so the
    # healthy request keeps its drafts and the K-step resumes.
    assert healthy.spec_token_ids == [0] * 7


def test_guard_drops_drafts_when_waiting_request_pending():
    decode_req = _req(computed=100, prompt=100, spec=[0] * 7)
    sched = _make_scheduler(num_spec=7, waiting=[object()], running=[decode_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decode_req.spec_token_ids == []


def test_guard_keeps_drafts_when_no_prefill_pending():
    decode_req = _req(computed=100, prompt=100, spec=[0] * 7, total=101)
    sched = _make_scheduler(num_spec=7, waiting=[], running=[decode_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decode_req.spec_token_ids == [0] * 7


def test_guard_drops_drafts_when_chunked_prefill_in_flight():
    decoding_req = _req(computed=100, prompt=100, spec=[0] * 7, total=101)
    chunking_req = _req(computed=50, prompt=100, spec=[])
    sched = _make_scheduler(num_spec=7, waiting=[], running=[decoding_req, chunking_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decoding_req.spec_token_ids == []


def test_guard_noop_for_text_stage_spec_config():
    # The text stage may carry its own n-gram config; only the Talker stage runs
    # the multi-frame loop, so the guard must stay a no-op there.
    text_req = _req(computed=100, prompt=100, spec=[0] * 15)
    sched = _make_scheduler(num_spec=15, stage="llm", waiting=[object()], running=[text_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert text_req.spec_token_ids == [0] * 15


def test_guard_arms_from_vllm_config_and_drops_drafts_for_a_running_chunk():
    # Real vLLM Scheduler instances expose no .speculative_config attribute
    # (and SchedulerConfig has no num_speculative_tokens), so the armed check
    # must read vllm_config.speculative_config -- otherwise the whole guard
    # silently stays off while the engine-side loop is armed.
    #
    # Steady-state decode sharing a step with a 7-token streaming chunk: both
    # spec_token_ids lists look innocent ([] and 7 placeholders), so the guard
    # must predict row widths from num_tokens - num_computed_tokens instead.
    from types import SimpleNamespace

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omni_talker_kstep_cache = None
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="tts"),
        speculative_config=SimpleNamespace(method="ngram", num_speculative_tokens=7),
    )
    assert sched._talker_kstep_armed() is True
    sched.num_spec_tokens = 7
    sched.max_model_len = 40960

    steady = _req(computed=100, prompt=100, spec=[0] * 7, total=101)  # 1 token -> 8 rows
    chunk = _req(computed=107, prompt=107, spec=[], total=114)  # 7-token chunk -> 7 rows
    sched.waiting = []
    sched.running = [steady, chunk]

    sched._drop_talker_drafts_if_prefill_pending()
    assert steady.spec_token_ids == []
    assert chunk.spec_token_ids == []


def test_guard_keeps_drafts_when_all_decodes_share_the_padded_width():
    # A first-step decode (no placeholders yet) and a steady-state decode both
    # schedule 1 token this step, so vLLM gives both the same 1+num_spec row
    # width: spans stay uniform and nothing may drop.
    steady = _req(computed=100, prompt=100, spec=[0] * 7, total=101)
    first_step = _req(computed=100, prompt=100, spec=[], total=101)
    sched = _make_scheduler(num_spec=7, waiting=[], running=[steady, first_step])

    sched._drop_talker_drafts_if_prefill_pending()
    assert steady.spec_token_ids == [0] * 7
    assert first_step.spec_token_ids == []


def test_talker_stop_token_ids_match_the_multi_frame_head():
    """Stage 1's stop id must be one the head that actually runs can emit.

    The default follows the head: the two-wide continue/stop row next to the
    NPU worker, and the codec EOS on every platform without that worker.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5 import pipeline as mcp_pipeline
    from vllm_omni.platforms.npu.worker import talker_multiframe

    assert talker_multiframe.STOP_TOKEN_ID == 1
    # The pipeline default cannot depend on the platform: the block that
    # collapses the head is a deploy-config decision, so the marker is added
    # there (test_minicpmo_talker_multi_frame_is_npu_scoped pins the merge).
    assert mcp_pipeline._talker_stop_token_ids() == [mcp_pipeline._CODEC_EOS_TOKEN_ID]


def test_multiframe_gate_matrix():
    from vllm_omni.platforms.npu.worker import talker_multiframe

    model = _contract_stub(armed=True)
    state = {"step": 5}

    # Uniform K-row decode: the loop engages with K=8.
    uniform = {
        "request_token_spans": [(0, 8), (8, 16)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state)},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, uniform) == 8

    # Mixed decode spans (an 8-row drafted decode plus a 1-row decode):
    # blocked, and flagged as a multi-token decode step.
    mixed_decode = {
        "request_token_spans": [(0, 8), (8, 9)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state)},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, mixed_decode) == 0
    assert talker_multiframe.is_multi_token_decode(model, mixed_decode) is True

    # Prefill rows mixed with a drafted decode: blocked the same way.
    mixed_prefill = {
        "request_token_spans": [(0, 515), (515, 523)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state), "_omni_is_prefill": True},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, mixed_prefill) == 0
    assert talker_multiframe.is_multi_token_decode(model, mixed_prefill) is True


def test_constant_drafts_fold_when_a_stop_truncates_a_row():
    """A request whose codec stop row fired mid-step accepted fewer tokens
    than the step ran; its next-step schedule is short the same way, so the
    batch folds and the whole step takes the single-frame path instead of
    scheduling a non-uniform span set that ``applies`` would refuse (and
    ``_model_forward`` would turn into a fatal error).

    The fold is decided by the caller: only a request that already ran the
    multi-frame loop -- tracked per-request in the runner's
    ``_kstep_drafted_reqs`` -- turns ``fold_short_rows`` on. With the flag
    off (the default) the same shape keeps drafting, which is the caller's
    way of saying the short row belongs to a request that has never
    multi-framed and must re-arm the K-step.
    """
    from vllm_omni.platforms.npu.worker import talker_multiframe

    # Five staggered codecs, K=6: one stopped at frame 3 (4 tokens), three
    # at frame 4 (5 tokens), one ran the step out (6 tokens). Every request
    # ran on a drafted step (5 drafts each), so the short rows are real
    # truncations.
    sampled = [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
    ]
    scheduled = [5, 5, 5, 5, 5]
    assert (
        talker_multiframe.constant_drafts(
            sampled, frames=6, num_reqs=5, scheduled_draft_counts=scheduled, fold_short_rows=True
        )
        == [[] for _ in range(5)]
    )
    # Default keeps the caller-decides contract: no fold without the flag.
    assert talker_multiframe.constant_drafts(sampled, frames=6, num_reqs=5, scheduled_draft_counts=scheduled) == [
        [talker_multiframe.CONTINUE_TOKEN_ID] * 5 for _ in range(5)
    ]


def test_constant_drafts_cold_start_short_row_still_drafts():
    """A short row from a request that never multi-framed must draft even
    with ``fold_short_rows`` on, or the K-step deadlocks on "no drafts
    emitted": the step after a fold schedules one token per request again,
    so every row looks short, and folding those too would starve the loop
    forever (the three-tier bench stall, 31/32 requests)."""
    from vllm_omni.platforms.npu.worker import talker_multiframe

    # Both rows sat on a draftless step (scheduled=0): one produced one
    # token, the other a full-width row. Neither is a stop truncation.
    sampled = [[5], [1, 2, 3, 4, 5, 6]]
    scheduled = [0, 0]
    drafts = talker_multiframe.constant_drafts(
        sampled, frames=6, num_reqs=2, scheduled_draft_counts=scheduled, fold_short_rows=True
    )
    assert drafts == [[talker_multiframe.CONTINUE_TOKEN_ID] * 5 for _ in range(2)]


def test_constant_drafts_fold_when_a_row_samples_nothing():
    from vllm_omni.platforms.npu.worker import talker_multiframe

    sampled = [[1, 2, 3, 4, 5, 6], [], [1, 2, 3, 4, 5, 6]]
    scheduled = [5, 5, 5]
    assert (
        talker_multiframe.constant_drafts(sampled, frames=6, num_reqs=3, scheduled_draft_counts=scheduled)
        == [[] for _ in range(3)]
    )


def test_constant_drafts_keep_drafts_when_every_row_is_full():
    """No stop anywhere: every request accepted the step's K tokens and the
    drafts arm the next K-frame step exactly as before the fold rule."""
    from vllm_omni.platforms.npu.worker import talker_multiframe

    sampled = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    scheduled = [5, 5]
    drafts = talker_multiframe.constant_drafts(sampled, frames=6, num_reqs=2, scheduled_draft_counts=scheduled)
    assert drafts == [[talker_multiframe.CONTINUE_TOKEN_ID] * 5 for _ in range(2)]


def test_constant_drafts_start_kstep_after_a_draftless_step():
    """A one-token row on a step the scheduler ran without drafts is the
    expected shape of a prefill result or of an ordinary single-frame step --
    not a truncation. These are exactly where the K-step span starts or
    resumes: the drafts must be emitted, or the loop never engages (the
    next step schedules one token again, produces another one-token row,
    and no drafts ever appear). Regression for the three-tier bench stall
    where 31/32 requests hung on this exact shape.
    """
    from vllm_omni.platforms.npu.worker import talker_multiframe

    # Prefill just completed for both requests: one token each, zero drafts.
    sampled = [[42], [43]]
    scheduled = [0, 0]
    drafts = talker_multiframe.constant_drafts(sampled, frames=8, num_reqs=2, scheduled_draft_counts=scheduled)
    assert drafts == [[talker_multiframe.CONTINUE_TOKEN_ID] * 7 for _ in range(2)]

    # Same shape after an intentional single-frame fallback step: the span
    # must resume on the next step.
    sampled = [[7]]
    drafts = talker_multiframe.constant_drafts(sampled, frames=8, num_reqs=1, scheduled_draft_counts=[0])
    assert drafts == [[talker_multiframe.CONTINUE_TOKEN_ID] * 7]


def test_constant_drafts_fold_a_mixed_drafted_and_undrafted_batch():
    """Some requests ran the K-step span while others sat out on a
    single-frame step: the rows cannot share one span (``applies`` refuses
    non-uniform schedules), so the batch folds rather than guess."""
    from vllm_omni.platforms.npu.worker import talker_multiframe

    sampled = [[1, 2, 3, 4, 5, 6, 7, 8], [9]]
    scheduled = [7, 0]
    drafts = talker_multiframe.constant_drafts(sampled, frames=8, num_reqs=2, scheduled_draft_counts=scheduled)
    assert drafts == [[], []]


def _contract_stub(*, armed: bool):
    """Minimal model that satisfies the multi-frame runner contract.

    The gate helpers verify the contract members whenever the flag is on, so a
    stub that only sets the flag is no longer a valid stand-in for the wrapper.
    """
    return SimpleNamespace(
        supports_multi_frame_decode=armed,
        batch_stop_logits=None,
        take_batch_stop_logits=lambda: None,
        set_batch_stop_logits=lambda logits: None,
        merge_frame_outputs=lambda frame_outputs, frame_stop_logits: frame_outputs,
    )


def _vocab_runner(*, supports_multi_frame: bool, vocab_size: int):
    return SimpleNamespace(
        model=_contract_stub(armed=supports_multi_frame),
        input_batch=SimpleNamespace(vocab_size=vocab_size),
    )


def test_stop_vocab_gate_reports_the_two_wide_stop_row_not_the_hidden_width():
    """``input_batch.vocab_size`` must be the stop row's width (2), not hidden.

    ``InputBatch.add_request`` keeps ``top_k = vocab_size`` as its "no top-k"
    sentinel and ``RejectionSampler.parse_output`` filters accepted tokens
    against it; both only hold at 0 or 2. The gate used to be handed
    ``text_hidden_states``, so it wrote the hidden width (768): that passes the
    ``parse_output`` filter by accident while taking ``top_k`` out of its
    sentinel range, and the two-wide stop row then never reaches the request's
    token list -- every request runs to ``max_tokens`` instead of stopping on
    EOS (the 910C scene where stage 1 reported ``finished_reason=length`` for
    all 34 requests).
    """
    from vllm_omni.platforms.npu.worker import talker_multiframe

    assert talker_multiframe.STOP_ROW_WIDTH == 2

    runner = _vocab_runner(supports_multi_frame=True, vocab_size=0)
    # Hidden-width rows on purpose: the width of this tensor is not the answer.
    talker_multiframe.ensure_stop_token_vocab(runner, torch.randn(4, 768))
    assert runner.input_batch.vocab_size == 2

    # Idempotent: arming again must not move it.
    talker_multiframe.ensure_stop_token_vocab(runner, torch.randn(4, 768))
    assert runner.input_batch.vocab_size == 2


def test_stop_vocab_gate_stays_off_without_multi_frame_or_rows():
    from vllm_omni.platforms.npu.worker import talker_multiframe

    # Single-frame models (K=1, stage 0/2) never take the branch that reads
    # vocab_size; the gate must leave them exactly as they were.
    single = _vocab_runner(supports_multi_frame=False, vocab_size=0)
    talker_multiframe.ensure_stop_token_vocab(single, torch.randn(4, 768))
    assert single.input_batch.vocab_size == 0

    # No rows at hand: None must not be read as "width unknown, arm anyway".
    armed = _vocab_runner(supports_multi_frame=True, vocab_size=0)
    talker_multiframe.ensure_stop_token_vocab(armed, None)
    assert armed.input_batch.vocab_size == 0


class _MinTokensLogitsProcessor:
    """The name is the contract: the implementation matches on ``type(proc).__name__``."""

    def __init__(self, min_toks):
        self.min_toks = min_toks


def test_kstep_min_tokens_neutralization_clears_the_censor_list():
    """The vLLM-level ``min_tokens`` mask list must be cleared under multi-frame decode.

    It masks the request's only stop signal (id 1 of the binary stop row) and
    unmasks only once ``len(output_token_ids) >= min_tokens`` -- a counter owned
    by vLLM's spec bookkeeping, so a request whose counter does not advance can
    never stop (observed: stage 1 all ``length``, zero ``stop``). The model-side
    guard already holds the codec EOS back with ``state.step < min_tokens``, so
    this layer is redundant.
    """
    from vllm_omni.platforms.npu.worker import talker_multiframe

    censor = _MinTokensLogitsProcessor({0: (50, [0, 0, 0], {1})})
    untouched = _MinTokensLogitsProcessor({0: (50, [0], {1})})
    untouched.__class__ = type("SomeOtherProcessor", (object,), {})  # not the target processor

    talker_multiframe.neutralize_kstep_min_tokens(SimpleNamespace(non_argmax_invariant=[untouched, censor]))
    assert censor.min_toks == {}
    assert untouched.min_toks == {0: (50, [0], {1})}

    # An empty list and odd inputs must not raise either.
    talker_multiframe.neutralize_kstep_min_tokens(SimpleNamespace(non_argmax_invariant=[]))
    talker_multiframe.neutralize_kstep_min_tokens(None)


def test_incomplete_prefill_chunk_skips_sampling():
    """The runner's eligibility flag must gate the K-step sampling branch.

    A request whose prompt spans several prefill chunks reports
    ``request_sample_eligible=False`` for the incomplete chunks; sampling
    there would advance codec history and RNG state, making the generated
    audio depend on how the prompt happened to be chunked (review on PR
    #7929). The flag only reaches the Talker while the outer wrapper
    forwards ``requires_request_sample_eligibility`` -- without that forward
    the runner never sends the flag and this branch degrades to the
    ``[True] * len(infos)`` fallback.
    """
    model = _make_talker(k_step_frames=8, scripted_samples=[42])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state

    out = model.make_omni_output(
        torch.randn(1, 8),
        model_intermediate_buffer=[{"request_id": "r1"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[False],
    )
    # No codec frame, no state advance, no history touch.
    assert out.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert state["step"] == 0
    assert "last_code" not in state
    assert model._request_codec_history.get("r1", []) == []

    # Eligible again (the chunking completed): sampling resumes.
    out2 = _frame_call(model, torch.randn(1, 8))
    assert state["last_code"] == 42
    assert out2.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [42]


def test_outer_wrapper_forwards_sampling_eligibility_flag():
    """The runner only sees the wrapper, so the flag must resolve through it.

    gpu/npu runners arm the ``request_sample_eligible`` transmission with
    ``getattr(self.model, "requires_request_sample_eligibility", False)``,
    and ``self.model`` is the stage's registered architecture -- the outer
    ``MiniCPMO45OmniForConditionalGeneration`` wrapper, never the inner
    Talker that declares the flag.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    wrapper = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(wrapper)
    # A wrapper without a Talker (stage 0 LLM) must not arm the transmission.
    assert wrapper.requires_request_sample_eligibility is False

    wrapper.talker = _make_talker(k_step_frames=8, scripted_samples=[])
    assert wrapper.requires_request_sample_eligibility is True


def test_request_sampling_params_pin_codec_knobs():
    """Per-request SamplingParams override the statically resolved knobs.

    Single-frame contract: temperature/seed/top-k/top-p/penalty overrides
    steer the codec stream exactly like they steer the vLLM sampler in the
    single-frame path. A request pinning temperature to 0 must also take
    the deterministic boundary sampler.
    """
    model = _make_talker(k_step_frames=8, scripted_samples=[42])
    # Deployment resolved a warm stochastic profile; the request pins its own.
    model._codec_temperature = 0.8
    model._codec_top_k = 100
    model._codec_top_p = 0.8
    model._codec_repetition_penalty = 1.05
    model._codec_seed = 42
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state

    model.make_omni_output(
        torch.randn(1, 8),
        model_intermediate_buffer=[{"request_id": "r1"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
        request_sampling_params=[
            SimpleNamespace(temperature=0.0, top_k=25, top_p=0.85, repetition_penalty=1.0, seed=7)
        ],
    )
    # temperature 0 -> the deterministic boundary sampler was taken.
    assert state["last_code"] == 42
    # The knobs are pinned for this request's samplers.
    assert state["codec_temperature"] == 0.0
    assert state["codec_top_k"] == 25
    assert state["codec_top_p"] == 0.85
    assert state["codec_repetition_penalty"] == 1.0
    gen = model._request_generator("r1", torch.device("cpu"))
    assert gen.initial_seed() == 7

    # A request without SamplingParams (dummy runs, CPU tests) keeps the
    # statically resolved deployment profile.
    model2 = _make_talker(k_step_frames=8, scripted_samples=[43])
    model2._codec_seed = 42
    model2._request_audio_states["r2"] = {"step": 0, "codes": torch.tensor([10, 11])}
    model2.make_omni_output(
        torch.randn(1, 8),
        model_intermediate_buffer=[{"request_id": "r2"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )
    gen2 = model2._request_generator("r2", torch.device("cpu"))
    assert gen2.initial_seed() == 42


def test_request_max_tokens_remaining_caps_kstep_frames():
    """The request's remaining output budget caps the K-step frame ceiling.

    The scheduler truncates the sampled ids at the request limit while the
    connector concatenates every emitted codec frame, so a request whose limit
    falls inside a K-frame step must stop emitting frames at the limit instead
    of running on to the stage's codec budget (review on PR #7929).
    """
    model = _make_talker(k_step_frames=8, scripted_samples=[42])
    model._codec_max_tokens = 4032
    state = {"step": 0, "max_tokens": 4032, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state

    model.make_omni_output(
        torch.randn(1, 8),
        model_intermediate_buffer=[{"request_id": "r1"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
        request_sampling_params=[
            SimpleNamespace(temperature=0.0, top_k=25, top_p=0.85, repetition_penalty=1.0, seed=7)
        ],
        request_max_tokens_remaining=[3],
    )
    # K=8 frames were available, but the request had only 3 tokens left.
    assert state["max_tokens"] == 3

    # A request without a limit keeps the stage-resolved codec budget.
    model2 = _make_talker(k_step_frames=8, scripted_samples=[43])
    model2._codec_max_tokens = 4032
    state2 = {"step": 0, "max_tokens": 4032, "codes": torch.tensor([10, 11])}
    model2._request_audio_states["r2"] = state2
    model2.make_omni_output(
        torch.randn(1, 8),
        model_intermediate_buffer=[{"request_id": "r2"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
        request_sampling_params=[
            SimpleNamespace(temperature=0.0, top_k=25, top_p=0.85, repetition_penalty=1.0, seed=7)
        ],
        request_max_tokens_remaining=[None],
    )
    assert state2["max_tokens"] == 4032


def test_reused_device_state_follows_tightened_max_tokens():
    """A reused codec device state picks up the caller's tightened ceiling.

    ``_merge_request_codec_params`` clamps ``state["max_tokens"]`` to the
    request's remaining output budget on later steps, but the device state is
    built once and reused. If it kept its creation-time budget, the codec loop
    would emit past the frame the engine stops accepting ids at -- the
    alignment the review asked to close (PR #7929).
    """
    model = _make_talker(k_step_frames=8, scripted_samples=[42])
    model._codec_max_tokens = 4032
    model._codec_repetition_penalty = 1.05
    model.head_code = nn.ModuleList([nn.Linear(8, _NUM_AUDIO_TOKENS, bias=False)])
    # _make_talker installs a scripted stub; this test needs the real boundary.
    real_greedy = type(model)._sample_audio_code_greedy
    model._sample_audio_code_greedy = real_greedy.__get__(model, type(model))
    state = {"step": 0, "max_tokens": 4032, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state
    hidden = torch.randn(1, 8)

    # Step 1: no request limit, so the device state starts at the stage budget.
    model._sample_audio_code_greedy(hidden, state["codes"], "r1", 0, 0, state["max_tokens"])
    assert int(model._request_codec_device_states["r1"].max_tokens.item()) == 4032

    # Step 2: the host ceiling was tightened to 3; the reused device state has
    # to follow it instead of staying at 4032.
    state["max_tokens"] = 3
    model._sample_audio_code_greedy(hidden, state["codes"], "r1", 1, 0, state["max_tokens"])
    assert int(model._request_codec_device_states["r1"].max_tokens.item()) == 3


def test_reused_device_state_follows_tightened_max_tokens_stochastic():
    """The stochastic boundary refreshes the reused device state the same way.

    ``_sample_audio_code`` reads the ceiling from the request state rather than
    from an argument, so it needs its own check that a reused device state does
    not keep the creation-time budget (PR #7929).
    """
    model = _make_talker(k_step_frames=8, scripted_samples=[42])
    model._codec_max_tokens = 4032
    model._codec_temperature = 0.8
    model._codec_repetition_penalty = 1.05
    model._codec_top_k = 0
    model._codec_top_p = 1.0
    model._codec_seed = 42
    model.head_code = nn.ModuleList([nn.Linear(8, _NUM_AUDIO_TOKENS, bias=False)])
    real_sample = type(model)._sample_audio_code
    model._sample_audio_code = real_sample.__get__(model, type(model))
    state = {"step": 0, "max_tokens": 4032, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state
    hidden = torch.randn(1, 8)

    model._sample_audio_code(hidden, state["codes"], "r1", 0)
    assert int(model._request_codec_device_states["r1"].max_tokens.item()) == 4032

    state["max_tokens"] = 5
    model._sample_audio_code(hidden, state["codes"], "r1", 1)
    assert int(model._request_codec_device_states["r1"].max_tokens.item()) == 5


def test_default_sampling_params_feed_codec_resolution():
    """``default_sampling_params`` sits between the YAML block and tts_config.

    It is the source the single-frame path's SamplingParams are built from,
    so a deployment tuning stage 1 through it must control the K-step codec
    sampler too (review on PR #7929). The explicit YAML block still wins key
    by key, and untouched keys fall through to the checkpoint config.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        resolve_codec_sampling_params,
    )

    tts_config = SimpleNamespace(
        seed=42,
        temperature=0.8,
        top_k=100,
        top_p=0.8,
        repetition_penalty=1.05,
        min_new_tokens=50,
        max_new_tokens=2048,
    )

    # No YAML block: the stage defaults drive the knobs the deployment set,
    # untouched keys keep the checkpoint values.
    resolved = resolve_codec_sampling_params(
        None, tts_config, deploy_defaults={"top_k": 25, "top_p": 0.85, "max_tokens": 4096}
    )
    assert resolved["top_k"] == 25
    assert resolved["top_p"] == 0.85
    assert resolved["max_tokens"] == 4096
    assert resolved["temperature"] == 0.8
    assert resolved["min_tokens"] == 50

    # The explicit YAML block still wins key by key.
    resolved2 = resolve_codec_sampling_params({"top_k": 9}, tts_config, deploy_defaults={"top_k": 25})
    assert resolved2["top_k"] == 9

    # No stage defaults either: back to the old chain (tts_config, then
    # module fallbacks).
    resolved3 = resolve_codec_sampling_params(None, tts_config)
    assert resolved3["top_k"] == 100
    assert resolved3["seed"] == 42


def test_request_min_tokens_reaches_codec_state():
    """The resolved request ``min_tokens`` reaches the K-step state, zero included.

    The NPU runner neutralizes vLLM's MinTokensLogitsProcessor, so the
    in-model codec sampler is the only min-length guard left (PR #7929
    review). The value merged here is the *resolved* request parameter:
    both entrypoints fold the deploy defaults in before the engine sees
    it (``serving_chat._apply_request_overrides`` and
    ``omni_base.resolve_sampling_params_list``), so the deployment
    minimum arrives as a positive 50, and an explicit caller 0 arrives
    as 0 -- floor disabled. vLLM 0.30.0's SamplingParams is a plain
    ``omit_defaults`` msgspec.Struct without ``model_fields_set``, so no
    provenance check survives here; tests use the real class.
    """
    from vllm import SamplingParams

    model = _make_talker(k_step_frames=8, scripted_samples=[42])
    model._codec_min_tokens = 50

    # The real class validates min_tokens <= max_tokens; the engine default
    # max_tokens is 16, so carry a realistic budget.
    base = dict(
        temperature=0.8,
        top_k=25,
        top_p=0.85,
        repetition_penalty=1.05,
        seed=None,
        max_tokens=2048,
    )

    # A positive request floor overrides the stage-resolved minimum.
    state = {"step": 0}
    model._merge_request_codec_params(state, SamplingParams(min_tokens=100, **base))
    assert state["min_tokens"] == 100

    # An explicit 0 pins 0 and turns the floor off, mirroring the
    # single-frame semantics (MinTokensLogitsProcessor skips 0).
    state_zero = {"step": 0}
    model._merge_request_codec_params(state_zero, SamplingParams(min_tokens=0, **base))
    assert state_zero["min_tokens"] == 0
    assert _codec_int_param(state_zero, "min_tokens", 50) == 0

    # Resolved deploy defaults keep the floor: this is what a request
    # that never mentions min_tokens resolves to online, and what the
    # offline entrypoint substitutes when the caller passes no params.
    state_default = {"step": 0}
    model._merge_request_codec_params(state_default, SamplingParams(min_tokens=50, **base))
    assert state_default["min_tokens"] == 50

    # No request params at all: the state stays unset and the K-step
    # loop keeps falling back to ``self._codec_min_tokens``.
    state_none = {"step": 0}
    model._merge_request_codec_params(state_none, None)
    assert "min_tokens" not in state_none

    # The min_new_tokens alias is honored too (older stubs may only set it).
    state_alias = {"step": 0}
    model._merge_request_codec_params(state_alias, SimpleNamespace(min_new_tokens=7, **base))
    assert state_alias["min_tokens"] == 7


def test_control_stop_ids_stay_out_of_codec_censor_set():
    """The scheduler's control-head ids never join the codec censor set.

    Under multi-frame decode the default NPU K=8 profile merges the request
    list to [1, 6561]: 1 finishes the request on the two-wide continue/stop
    control row, 6561 is the codec EOS. Inside the 6562-wide codec
    vocabulary 1 is an ordinary audio code -- censoring it would skew the
    below-floor distribution and truncate real speech once sampled. The
    single-frame path (no stop_token_ids: [1] block) never carries it.
    """
    model = _make_talker(k_step_frames=8, scripted_samples=[1, 42, _EOS_ID])
    model._codec_min_tokens = 50

    # The merge drops the control ids and keeps the codec-side stop ids.
    state = {"step": 0}
    model._merge_request_codec_params(state, SimpleNamespace(stop_token_ids=[1, 6561], min_tokens=50))
    assert state["codec_stop_token_ids"] == [6561]

    # The drop is gated on the multi-frame arm: a non-armed deployment keeps
    # the legacy verbatim copy (the control-row contract does not apply).
    model_legacy = _make_talker(k_step_frames=1, scripted_samples=[1])
    state_legacy = {"step": 0}
    model_legacy._merge_request_codec_params(state_legacy, SimpleNamespace(stop_token_ids=[1, 6561], min_tokens=50))
    assert state_legacy["codec_stop_token_ids"] == [1, 6561]

    # End to end across the sampling boundary: a codec 1 sampled past the
    # floor is emitted as ordinary audio and does not finish the request;
    # a later codec EOS still terminates it normally.
    state_stream = {"step": 0}
    model._request_audio_states["r1"] = state_stream
    params = SimpleNamespace(stop_token_ids=[1, 6561], min_tokens=50)

    def _frame():
        return model.make_omni_output(
            torch.randn(1, 8),
            model_intermediate_buffer=[{"request_id": "r1"}],
            request_token_spans=[(0, 1)],
            request_sample_eligible=[True],
            request_sampling_params=[params],
        )

    out0 = _frame()
    assert out0.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [1]
    assert not state_stream.get("finished")

    out1 = _frame()
    assert out1.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [42]
    assert not state_stream.get("finished")

    out2 = _frame()
    assert out2.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert state_stream["finished"] is True


def test_top_k_then_top_p_matches_single_frame_order():
    """K-step filtering follows the engine's order: top-k, then top-p.

    The single-frame path applies top-k first and then computes top-p over the
    top-k-filtered distribution, keeping at least one candidate
    (vllm/v1/sample/ops/topk_topp_sampler.py:392/:404/:415). Filtering top-p
    first -- or holding a floor of three candidates -- changes the sampling
    distribution even for identical logits and sampling parameters, so enabling
    K-step decoding would silently alter the single-frame contract
    (PR #7929 review).
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample import (
        make_device_state,
        prepare_codec_logits,
    )

    device_state = make_device_state(torch.zeros(0, dtype=torch.int32), step=0, max_tokens=100, finished=False)
    kwargs = dict(
        state=device_state,
        min_tokens=torch.tensor([0]),
        temperature=torch.tensor([1.0]),
        repetition_penalty=torch.tensor([1.0]),
        eos_token_id=_EOS_ID,
    )
    # The review's example: candidate weights [40, 30, 20, 10] with top_k=3 and
    # top_p=0.5. The engine keeps two candidates -- top-p runs over the
    # renormalized top-3 distribution [0.444, 0.333, 0.222], whose ascending
    # cumsum first exceeds 0.5 at the second candidate. Keeping three
    # candidates was the reported bug.
    weights = torch.tensor([40.0, 30.0, 20.0, 10.0])
    logits = torch.full((1, _NUM_AUDIO_TOKENS), -1e4)
    logits[0, 10:14] = weights.log()
    filtered = prepare_codec_logits(logits.clone(), top_k=3, top_p=0.5, **kwargs)
    assert int(torch.isfinite(filtered).sum()) == 2
    # An explicit top_k is honored exactly: top_k=1 keeps a single candidate.
    filtered_top_k_one = prepare_codec_logits(logits.clone(), top_k=1, top_p=1.0, **kwargs)
    assert int(torch.isfinite(filtered_top_k_one).sum()) == 1
    # top-p alone keeps at least one candidate, like the engine's `at least one`
    # row (topk_topp_sampler.py:415).
    filtered_top_p_only = prepare_codec_logits(logits.clone(), top_k=0, top_p=0.01, **kwargs)
    assert int(torch.isfinite(filtered_top_p_only).sum()) >= 1


def test_eos_window_mask_hides_codec_eos():
    """``eos_window_masked=True`` masks codec EOS regardless of the step.

    This is the turn-end drain path: duplex meta ``turn_end`` pins
    ``state["turn_end_drain"]`` (tts preprocess), the K-step loop forwards it
    as ``eos_window_masked``, and the sampler must hide EOS so the chunk can
    drain its remaining cadence frames instead of ending early.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample import (
        make_device_state,
        prepare_codec_logits,
    )

    logits = torch.full((1, _NUM_AUDIO_TOKENS), 1.0)
    logits[0, _EOS_ID] = 10.0
    device_state = make_device_state(torch.zeros(0, dtype=torch.int32), step=5, max_tokens=100, finished=False)
    kwargs = dict(
        state=device_state,
        min_tokens=torch.tensor([0]),
        temperature=torch.tensor([0.8]),
        repetition_penalty=torch.tensor([1.0]),
        eos_token_id=_EOS_ID,
        top_p=1.0,
        top_k=0,
    )
    # step >= min_tokens: EOS is eligible and keeps its dominant logit.
    unmasked = prepare_codec_logits(logits.clone(), eos_window_masked=False, **kwargs)
    assert unmasked[0, _EOS_ID] != float("-inf")
    # Drain window: EOS is forced out even though the step allows it.
    masked = prepare_codec_logits(logits.clone(), eos_window_masked=True, **kwargs)
    assert masked[0, _EOS_ID] == float("-inf")


_STOP_ID = 6561  # in-vocabulary for the codec head, not the codec EOS


def _real_boundary_talker(temperature: float):
    """A talker on the real (non-stub) greedy or stochastic boundary."""
    model = _make_talker(k_step_frames=8, scripted_samples=[])
    model._codec_temperature = temperature
    model._codec_repetition_penalty = 1.05
    model.head_code = nn.ModuleList([nn.Linear(8, _NUM_AUDIO_TOKENS, bias=False)])
    if temperature == 0.0:
        real_greedy = type(model)._sample_audio_code_greedy
        model._sample_audio_code_greedy = real_greedy.__get__(model, type(model))
    else:
        model._codec_top_k = 0
        model._codec_top_p = 1.0
        model._codec_seed = 42
        real_sample = type(model)._sample_audio_code
        model._sample_audio_code = real_sample.__get__(model, type(model))
    return model


def _bias_head_to(model, token_id: int):
    """Make the head's argmax land exactly on ``token_id``."""
    with torch.no_grad():
        model.head_code[0].weight.zero_()
        model.head_code[0].weight[token_id] = 1.0


def test_stop_id_finishes_request_through_device_state_greedy():
    """A codec stop id finishes the request via the device state alone.

    Review follow-up (PR #7929): make_omni_output used to recompute the
    eos/stop-id/limit transition host-side while codec_sample_result ran the
    same transition on the device. The device state is now the single source,
    so a stop-id frame finishes the request and drops out of the audio stream
    without the host ever recomputing the stop set.
    """
    model = _real_boundary_talker(0.0)
    state = {
        "step": 0,
        "max_tokens": 4032,
        "codes": torch.tensor([10, 11]),
        "codec_stop_token_ids": [_STOP_ID],
    }
    model._request_audio_states["r1"] = state
    _bias_head_to(model, _STOP_ID)

    out = _frame_call(model, torch.ones(1, 8))

    assert state["finished"] is True
    # The stop frame itself must not join the emitted audio stream.
    assert out.multimodal_outputs["codes"]["audio"][0].numel() == 0
    # K-step bookkeeping: a terminating frame records no confirmed codec id.
    assert "last_code" not in state
    assert model._request_codec_history.get("r1", []) == []
    # The device state is what decided this.
    assert bool(model._request_codec_device_states["r1"].finished.item()) is True


def test_normal_frame_then_stop_id_frame_device_routing():
    """A normal frame emits and records; the following stop frame terminates."""
    model = _real_boundary_talker(0.0)
    state = {
        "step": 0,
        "max_tokens": 4032,
        "codes": torch.tensor([10, 11]),
        "codec_stop_token_ids": [_STOP_ID],
    }
    model._request_audio_states["r1"] = state
    _bias_head_to(model, 42)
    out0 = _frame_call(model, torch.ones(1, 8))
    assert state["last_code"] == 42
    assert state["finished"] is False
    assert out0.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [42]

    _bias_head_to(model, _STOP_ID)
    out1 = _frame_call(model, torch.ones(1, 8))
    assert state["finished"] is True
    assert out1.multimodal_outputs["codes"]["audio"][0].numel() == 0
    # Only confirmed frames reach the history: the stop frame does not.
    assert model._request_codec_history["r1"] == [42]


def test_stop_id_finishes_request_stochastic_path():
    """The stochastic boundary routes stop ids through the same device state."""
    model = _real_boundary_talker(0.8)
    state = {
        "step": 0,
        "max_tokens": 4032,
        "codes": torch.tensor([10, 11]),
        "codec_stop_token_ids": [_STOP_ID],
    }
    model._request_audio_states["r1"] = state
    _bias_head_to(model, _STOP_ID)

    _frame_call(model, torch.ones(1, 8))

    assert state["finished"] is True
    assert bool(model._request_codec_device_states["r1"].finished.item()) is True


def test_ignore_eos_does_not_ignore_stop_ids():
    """``ignore_eos`` blanks the EOS id, not the request's stop ids.

    The engine treats the two independently (a sampled stop id finishes the
    request even when ``ignore_eos`` is set), and the device-side transition
    keeps that split: only the EOS comparison is blanked.
    """
    model = _real_boundary_talker(0.0)
    state = {
        "step": 0,
        "max_tokens": 4032,
        "codes": torch.tensor([10, 11]),
        "codec_stop_token_ids": [_STOP_ID],
        "codec_ignore_eos": True,
    }
    model._request_audio_states["r1"] = state
    _bias_head_to(model, _STOP_ID)

    _frame_call(model, torch.ones(1, 8))

    assert state["finished"] is True
    assert bool(model._request_codec_device_states["r1"].finished.item()) is True
