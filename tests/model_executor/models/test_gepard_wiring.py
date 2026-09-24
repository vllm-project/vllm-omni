# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for the seam between Gepard's talker and the AR runner.

``test_gepard_window.py`` pins the decode arithmetic *below* this layer, and
the e2e test exercises the layer *above* it but only along the path where the
stop head fires. Neither covers the wiring itself: whether anything calls the
final emit on a request's last step, and whether the payload it produces
survives the runner's sparse-audio routing.

These drive the real ``preprocess`` -> ``forward`` -> ``make_omni_output``
sequence through a stand-in for ``GPUARModelRunner.execute_model``, and use the
shared ``sparse_audio.resolve_sparse_mm_routing`` helper used by the runner to
decide what actually reaches the caller. Only the LM backbone and the frame
sampler are stubbed out; every method under test is the real one, so a test
here goes red when the engine
path breaks — which a test that calls the flush by hand cannot do.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.gepard_talker import (
    CHUNK_FRAMES,
    FIRST_CHUNK_FRAMES,
    GepardTalkerForConditionalGeneration,
    _ReqInfo,
)
from vllm_omni.model_executor.models.gepard.nanocodec import NanoCodec
from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids
from vllm_omni.worker import sparse_audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLES_PER_FRAME = 1024
NUM_HEADS = 32
SAMPLE_RATE = 22050
LOGGER_NAME = "vllm_omni.model_executor.models.gepard.gepard_talker"

# Each request tags its frames from a distinct base, so a sample that reaches
# the wrong request is visible in the value, not just the length.
_TAG_STRIDE = 100_000


class _TagCodec:
    """Frame carrying tag ``v`` -> samples ``v*SPF .. v*SPF+SPF-1``, wherever it sits.

    Shift-invariant, so "what the caller received" can be compared against one
    whole-history decode exactly. The tag rides channel 1: channel 0 is head0,
    which the talker keeps inside its real (small) vocab.
    """

    def __init__(self) -> None:
        self.decode_calls = 0

    def decode_from_codes(self, codes: torch.Tensor, codes_len: torch.Tensor):
        assert codes.shape[0] == 1, "streaming decode is per-request"
        assert codes.shape[1] == NUM_HEADS, f"expected {NUM_HEADS} code channels"
        self.decode_calls += 1
        tags = codes[0, 1, :].to(torch.int64)  # (T,)
        offsets = torch.arange(SAMPLES_PER_FRAME, dtype=torch.int64)
        audio = (tags.unsqueeze(1) * SAMPLES_PER_FRAME + offsets).reshape(1, -1)
        return audio.to(torch.float32), codes_len * SAMPLES_PER_FRAME


class _StubBackbone(nn.Module):
    """Stands in for the Qwen3.5 backbone; only shapes matter to this seam."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(int(input_ids.shape[0]), self.hidden)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = inputs_embeds if inputs_embeds is not None else input_ids
        return torch.zeros(int(src.shape[0]), self.hidden)


def _expected_samples(tag_base: int, num_frames: int) -> torch.Tensor:
    offsets = torch.arange(SAMPLES_PER_FRAME, dtype=torch.int64)
    tags = (torch.arange(num_frames, dtype=torch.int64) + tag_base).unsqueeze(1)
    return (tags * SAMPLES_PER_FRAME + offsets).reshape(-1).to(torch.float32)


class _MiniEngine:
    """Drives the talker the way ``GPUARModelRunner.execute_model`` does.

    Reproduces the three orderings the dropped-tail bug lived in:

    * ``on_requests_finished`` fires at the START of the step AFTER the one
      that finished the request (it reads ``scheduler_output.finished_req_ids``);
    * a finished request is not scheduled again, so it is absent from
      ``req_ids_output_copy`` and from that step's routing;
    * with nothing left to schedule there is no step at all, so no further
      ``forward`` runs;
    * a stop-token finish gets one more scheduled step, a budget finish does
      not -- under async scheduling only the latter is guarded against it.
    """

    def __init__(self, talker: GepardTalkerForConditionalGeneration, *, max_model_len: int) -> None:
        self.talker = talker
        self.max_model_len = max_model_len
        self.reqs: dict[str, dict] = {}
        self.order: list[str] = []
        self._finished_last_step: list[str] = []
        self.delivered: dict[str, list[torch.Tensor]] = {}
        self.committed_frames: dict[str, int] = {}
        self.steps = 0
        self._tags_handed_out = 0

    def add_request(
        self,
        req_id: str,
        *,
        max_tokens: int,
        stop_after: int | None = None,
        text_len: int = 3,
        seed: int | None = None,
    ) -> None:
        """Submit a request. An id already seen is a resubmit of that id.

        Tags come from a counter, so a resubmitted id gets frames the previous
        request could not have produced.
        """
        prompt_ids = build_gepard_prompt_ids(list(range(10, 10 + text_len)), config=self.talker.config)
        self._tags_handed_out += 1
        self.reqs[req_id] = {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_len": len(prompt_ids),
            "max_tokens": max_tokens,
            "stop_after": stop_after,
            "stop_fired": False,
            "seed": seed,
            "computed": 0,
            "generated": 0,
            "finished": False,
            "ghost_step_pending": False,
            "next_token": None,
            "tag_base": _TAG_STRIDE * self._tags_handed_out,
        }
        if req_id not in self.order:
            self.order.append(req_id)
        self.delivered[req_id] = []
        self.committed_frames[req_id] = 0

    def abort(self, req_id: str) -> None:
        """Kill a request the way an aborted client does: no STOP, no budget."""
        self.reqs[req_id]["finished"] = True
        self._finished_last_step.append(req_id)

    def _live(self) -> list[str]:
        return [r for r in self.order if not self.reqs[r]["finished"]]

    def step(self) -> bool:
        """Run one engine step. False when there was nothing to schedule."""
        live = self._live()
        if not live:
            return False
        talker = self.talker
        self.steps += 1

        # --- runner: notify the model of requests finished since last step ---
        if self._finished_last_step:
            talker.on_requests_finished(set(self._finished_last_step))
            self._finished_last_step = []

        # --- runner: per-request preprocess, in input_batch.req_ids order ---
        embeds_parts, ids_parts = [], []
        for req_id in live:
            r = self.reqs[req_id]
            is_prefill = r["computed"] < r["prompt_len"]
            if is_prefill:
                ids = r["prompt_ids"]
            else:
                ids = torch.tensor([r["next_token"] or 0], dtype=torch.long)
            span = int(ids.shape[0])
            out_ids, out_embeds, _ = talker.preprocess(
                input_ids=ids,
                input_embeds=None,
                request_id=req_id,
                _omni_prompt_len=r["prompt_len"],
                _omni_num_computed_tokens=r["computed"],
                _omni_is_prefill=is_prefill,
                _omni_max_tokens=r["max_tokens"],
                _omni_seed=r["seed"],
            )
            ids_parts.append(out_ids)
            embeds_parts.append(out_embeds)
            r["span"] = span

        inputs_embeds = torch.cat(embeds_parts, dim=0)
        input_ids = torch.cat(ids_parts, dim=0)
        positions = torch.arange(int(input_ids.shape[0]), dtype=torch.long)

        # --- runner: forward, then build the multimodal payload ---
        hidden = talker.forward(input_ids, positions, inputs_embeds=inputs_embeds)
        results = list(talker._results_queue)
        omni_out = talker.make_omni_output(hidden)
        mm = omni_out.multimodal_outputs
        # compute_logits drains this; it does not touch the audio path.
        talker._results_queue.clear()

        for req_id in live:
            state = talker._active_states.get(req_id)
            if state is not None:
                self.committed_frames[req_id] = state.frame_count

        # --- runner: sparse-audio routing decides what reaches the caller ---
        downstream, sparse_index, is_sparse = sparse_audio.resolve_sparse_mm_routing(
            engine_output_type="audio",
            req_ids_output_copy=list(live),
            downstream_req_ids=list(live),
            multimodal_outputs=mm,
        )
        assert is_sparse, "gepard always marks meta.sparse_audio"
        for rid in downstream:
            self.delivered[rid].append(mm["model_outputs"][sparse_index[rid]].reshape(-1))

        # --- runner/scheduler bookkeeping: advance and detect finishes ---
        stopped = {rid for rid, _head0, do_stop in results if do_stop}
        for req_id in live:
            r = self.reqs[req_id]
            r["computed"] += r["span"]
            r["generated"] = r["computed"] - r["prompt_len"] + 1
            r["next_token"] = 0
            length_cap = min(r["max_tokens"], self.max_model_len - r["prompt_len"])
            if r["ghost_step_pending"]:
                # This step WAS the extra one. The engine finished the request
                # on the stop token last step and discards whatever came back
                # here, whether or not the model chose to stop again.
                r["finished"] = True
                self._finished_last_step.append(req_id)
            elif req_id in stopped:
                # Ends on the stop token: one more step is already scheduled.
                r["ghost_step_pending"] = True
            elif r["generated"] >= length_cap:
                r["finished"] = True
                self._finished_last_step.append(req_id)
        return True

    def run(self, max_steps: int = 4000) -> None:
        for _ in range(max_steps):
            if not self.step():
                return
        raise AssertionError("engine did not drain")

    def audio_for(self, req_id: str) -> torch.Tensor:
        chunks = self.delivered[req_id]
        return torch.cat(chunks) if chunks else torch.empty(0)


def _make_talker(*, max_model_len: int = 4096) -> tuple[GepardTalkerForConditionalGeneration, _TagCodec]:
    """A real talker with only the audio-path collaborators populated.

    ``__init__`` needs a VllmConfig and builds the Qwen3.5 backbone; the seam
    under test needs none of that, so the backbone and the frame sampler are
    stubbed and everything else is the class's own code.
    """
    cfg = GepardConfig()
    # Fixed, not cfg.get_text_config().hidden_size: nothing in this seam reads
    # the real width, and a small one keeps the test off the backbone config.
    hidden = 64
    codec = NanoCodec(codec_id="test/fake-codec", sample_rate=SAMPLE_RATE)
    fake = _TagCodec()
    codec._codec = fake  # bypasses load(); NeMo is never imported
    assert codec.is_loaded

    talker = GepardTalkerForConditionalGeneration.__new__(GepardTalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.config = cfg
    talker.model = _StubBackbone(hidden)
    # requires_grad=False: preprocess writes the prefix into inputs_embeds
    # in place, and production runs that under inference_mode.
    talker.null_prefix = nn.Parameter(torch.zeros(cfg.num_speaker_prefix, hidden), requires_grad=False)
    talker._codec = codec
    talker._sample_rate = SAMPLE_RATE
    talker._max_model_len = max_model_len
    talker._active_states = {}
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    return talker, fake


def _install_scripted_sampling(talker: GepardTalkerForConditionalGeneration, engine: _MiniEngine) -> None:
    """Replace the 32-head sampler with a scripted one.

    Frames carry a per-request, per-step tag so the caller's audio is
    self-describing. STOP fires once, on the request's configured frame: on the
    extra step the engine then runs, re-firing is likely but not guaranteed,
    and the model may not assume it.
    """
    head0_vocab = talker.config.head0_vocab_size

    # Signature mirrors the real ``_sample_frame`` minus ``self``; the scripted
    # sampler ignores ``generators`` because seeding is decided in
    # ``_build_with_real_sampler``, which keeps the model's own sampler.
    def _sample_frame(rows: torch.Tensor, generators: list[torch.Generator | None] | None = None):
        n = int(rows.shape[0])
        head0 = torch.zeros(n, dtype=torch.long)
        heads = torch.zeros(n, NUM_HEADS - 1, dtype=torch.long)
        stop = torch.zeros(n, dtype=torch.bool)
        for i, (req_id, _span, _samples_frame, *_rest) in enumerate(talker._pending_requests):
            state = talker._active_states.get(req_id)
            frame_index = state.frame_count if state is not None else 0
            tag = engine.reqs[req_id]["tag_base"] + frame_index
            head0[i] = tag % head0_vocab
            heads[i] = tag
            r = engine.reqs[req_id]
            if r["stop_after"] is not None and frame_index >= r["stop_after"] and not r["stop_fired"]:
                r["stop_fired"] = True
                stop[i] = True
        return head0, heads, stop

    talker._sample_frame = _sample_frame
    talker._audio_frame_embed = lambda h0, h: torch.zeros(int(h0.shape[0]), talker.model.hidden)


def _build(*, max_model_len: int = 4096) -> tuple[GepardTalkerForConditionalGeneration, _TagCodec, _MiniEngine]:
    talker, codec = _make_talker(max_model_len=max_model_len)
    engine = _MiniEngine(talker, max_model_len=max_model_len)
    _install_scripted_sampling(talker, engine)
    return talker, codec, engine


def _build_with_real_sampler(*, max_model_len: int = 4096) -> tuple[GepardTalkerForConditionalGeneration, _MiniEngine]:
    """Same seam, but ``_sample_frame`` is the model's own 32-head sampler.

    The stub backbone hands every row the same hidden state, so the sampled
    codes — and therefore the audio the codec tags them into — vary with
    nothing but the Gumbel draws. That makes this the one place where "did the
    caller's seed reach the audio" is a decidable question.
    """
    talker, _codec = _make_talker(max_model_len=max_model_len)
    engine = _MiniEngine(talker, max_model_len=max_model_len)
    cfg = talker.config
    talker.vocab_sizes = list(cfg.audio_head_levels)
    talker.num_heads = cfg.num_audio_heads
    talker.head0_vocab = cfg.head0_vocab_size
    talker.stop_token = cfg.stop_token
    talker.stop_threshold = cfg.stop_threshold
    talker.temperature = cfg.temperature
    talker._greedy = False
    gather_idx, mask = GepardTalkerForConditionalGeneration._build_gather_mask(talker.vocab_sizes)
    talker._cb_gather_idx = gather_idx
    talker._cb_mask = mask
    # Fixed weights: two engine runs must differ only by their RNG.
    torch.manual_seed(0)
    talker.fused_codebook_head = nn.Linear(talker.model.hidden, sum(talker.vocab_sizes))
    talker.stop_head = nn.Linear(talker.model.hidden, 1)
    # Never stop: these requests end on their token budget, so the frame count
    # is fixed and only the sampled codes can differ.
    torch.nn.init.constant_(talker.stop_head.bias, -20.0)
    talker._audio_frame_embed = lambda h0, h: torch.zeros(int(h0.shape[0]), talker.model.hidden)
    return talker, engine


# --------------------------------------------------------------------------
# The tail must reach the caller, whichever way the request ends.
# --------------------------------------------------------------------------


def test_last_in_flight_request_delivers_every_committed_frame() -> None:
    """The only request runs out of budget: there is no later step to flush in.

    This is the case a deferred flush cannot serve at all — nothing runs after
    the final forward.
    """
    _talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=FIRST_CHUNK_FRAMES + CHUNK_FRAMES + 5)
    engine.run()

    frames = engine.committed_frames["req-0"]
    assert frames > 0
    torch.testing.assert_close(
        engine.audio_for("req-0"),
        _expected_samples(engine.reqs["req-0"]["tag_base"], frames),
    )


@pytest.mark.parametrize(
    "max_tokens",
    [
        1,
        FIRST_CHUNK_FRAMES,
        FIRST_CHUNK_FRAMES + 1,
        30,
        FIRST_CHUNK_FRAMES + CHUNK_FRAMES,
        100,
        150,
    ],
)
def test_budget_truncated_request_delivers_every_committed_frame(max_tokens: int) -> None:
    """Lowering max_tokens must shorten the clip, never silently cut it back
    to the last cadence boundary."""
    _talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=max_tokens)
    engine.run()

    frames = engine.committed_frames["req-0"]
    assert frames == max_tokens, "every sampled step commits one frame when STOP never fires"
    torch.testing.assert_close(
        engine.audio_for("req-0"),
        _expected_samples(engine.reqs["req-0"]["tag_base"], frames),
    )


def test_stop_head_path_still_delivers_every_committed_frame() -> None:
    """Regression guard: the path that already worked must keep working."""
    _talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=4000, stop_after=FIRST_CHUNK_FRAMES + 7)
    engine.run()

    frames = engine.committed_frames["req-0"]
    assert frames == FIRST_CHUNK_FRAMES + 7, "the STOP frame itself is not committed"
    torch.testing.assert_close(
        engine.audio_for("req-0"),
        _expected_samples(engine.reqs["req-0"]["tag_base"], frames),
    )


def test_the_step_after_a_stop_token_commits_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Async scheduling runs one more step on a request that already stopped.

    Committing its frame would strand it — nothing ships after the stop step —
    and would make the truncation warning fire on a request that delivered
    everything it produced, which is the one thing that warning must not do.
    """
    talker, _codec, engine = _build()
    stop_after = FIRST_CHUNK_FRAMES + 6
    engine.add_request("req-0", max_tokens=4000, stop_after=stop_after)

    target = logging.getLogger(LOGGER_NAME)
    target.addHandler(caplog.handler)
    prev = target.level
    target.setLevel(logging.WARNING)
    try:
        engine.run()
        # The cleanup that would warn runs on the step after the request
        # leaves, and here there is none. Drive it the way the runner would.
        talker.on_requests_finished({"req-0"})
        talker._flush_deferred_cleanup()
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(prev)

    assert engine.reqs["req-0"]["ghost_step_pending"], "the harness did not run the extra step"
    frames = engine.committed_frames["req-0"]
    assert frames == stop_after, f"the extra step committed a frame: {frames} > {stop_after}"
    torch.testing.assert_close(
        engine.audio_for("req-0"),
        _expected_samples(engine.reqs["req-0"]["tag_base"], frames),
    )
    assert not [m for m in caplog.messages if "undelivered frame" in m], (
        f"a fully delivered request was reported as truncated; saw {caplog.messages}"
    )


def test_a_resubmitted_request_id_keeps_its_new_state(caplog: pytest.LogCaptureFixture) -> None:
    """An aborted id can come back before its cleanup has run.

    The runner treats the two as distinct requests, so the pending free must
    not take the second one's state: without that, the state is popped at the
    end of the very forward that created it and every later decode falls back
    to the zero-embedding path — the request keeps producing frames, so only
    the audio is wrong.
    """
    _talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=4000)
    for _ in range(FIRST_CHUNK_FRAMES + 2):
        engine.step()
    engine.abort("req-0")

    # Same id, resubmitted before the model has been told about the abort.
    engine.add_request("req-0", max_tokens=FIRST_CHUNK_FRAMES + CHUNK_FRAMES + 5)

    target = logging.getLogger(LOGGER_NAME)
    target.addHandler(caplog.handler)
    prev = target.level
    target.setLevel(logging.WARNING)
    try:
        engine.run()
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(prev)

    frames = engine.committed_frames["req-0"]
    assert frames == engine.reqs["req-0"]["max_tokens"], "the second request lost frames"
    torch.testing.assert_close(
        engine.audio_for("req-0"),
        _expected_samples(engine.reqs["req-0"]["tag_base"], frames),
        msg="the second request did not receive its own audio",
    )
    # The abort still gets reported: the cleanup that would have said so is the
    # one this cancelled.
    assert any("resubmitted" in m for m in caplog.messages), (
        f"the aborted request's truncation went unreported; saw {caplog.messages}"
    )


def test_a_preempted_request_is_reported_rather_than_resumed(caplog: pytest.LogCaptureFixture) -> None:
    """Recompute preemption re-prefills a request that was mid-generation.

    This model cannot survive that, and the point of the test is that it says
    so. The recomputed ids are the prompt plus the head0 codes sampled so far,
    which go through the text embedding table, and each frame's other 31 codes
    are never in the token stream at all — there is nothing to stitch the two
    halves back together with, so generation restarts from frame 0.

    The harness re-prefills with the bare prompt, which understates the damage.
    What is under test is the report and the reset, not the corruption, which
    needs a real backbone to observe.
    """
    _talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=4000)
    for _ in range(FIRST_CHUNK_FRAMES + 2):
        engine.step()
    assert engine.committed_frames["req-0"] > 0

    # Preemption the way the scheduler does it: computed tokens back to 0, so
    # the next step re-prefills, and no id reaches on_requests_finished.
    engine.reqs["req-0"]["computed"] = 0

    target = logging.getLogger(LOGGER_NAME)
    target.addHandler(caplog.handler)
    prev = target.level
    target.setLevel(logging.WARNING)
    try:
        engine.step()
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(prev)

    assert any("preempted" in m for m in caplog.messages), (
        f"a preempted request restarted its audio silently; saw {caplog.messages}"
    )
    assert not any("resubmitted" in m for m in caplog.messages), (
        "a preemption was reported as an id resubmit; the two need different words"
    )
    assert engine.committed_frames["req-0"] == 1, "the re-prefill must start the frame history over"


def test_concurrent_requests_each_receive_their_own_full_audio() -> None:
    """One request finishes while another stays live.

    The earlier finisher is the case where a deferred flush *does* run but the
    payload is filtered out again, because its id has left the output batch.
    """
    _talker, _codec, engine = _build()
    engine.add_request("req-short", max_tokens=FIRST_CHUNK_FRAMES + 3)
    engine.add_request("req-long", max_tokens=FIRST_CHUNK_FRAMES + CHUNK_FRAMES + 11)
    engine.run()

    for req_id in ("req-short", "req-long"):
        frames = engine.committed_frames[req_id]
        assert frames == engine.reqs[req_id]["max_tokens"]
        torch.testing.assert_close(
            engine.audio_for(req_id),
            _expected_samples(engine.reqs[req_id]["tag_base"], frames),
            msg=f"{req_id} lost or gained samples",
        )


def test_codec_still_runs_once_per_chunk_not_once_per_frame() -> None:
    """Emitting on the final step must not degrade into per-frame decoding."""
    _talker, codec, engine = _build()
    num_frames = FIRST_CHUNK_FRAMES + 3 * CHUNK_FRAMES
    engine.add_request("req-0", max_tokens=num_frames)
    engine.run()

    assert codec.decode_calls == 4, f"expected one decode per chunk, got {codec.decode_calls}"


# --------------------------------------------------------------------------
# Deferred cleanup is cleanup: it must not try to produce output.
# --------------------------------------------------------------------------


def test_deferred_cleanup_never_queues_audio() -> None:
    """Anything queued from the cleanup hook is unroutable by construction.

    By the time it runs the id has left ``req_ids_output_copy``, so a payload
    under it is dropped; queueing there only risks leaking one request's audio
    into the next step's payload for a different request.
    """
    talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=FIRST_CHUNK_FRAMES + 4)
    engine.add_request("req-1", max_tokens=FIRST_CHUNK_FRAMES + CHUNK_FRAMES)
    while engine.step():
        if engine.reqs["req-0"]["finished"] and not engine.reqs["req-1"]["finished"]:
            break

    talker._audio_queue.clear()
    talker.on_requests_finished({"req-0"})
    talker._flush_deferred_cleanup()

    assert talker._audio_queue == [], "cleanup emitted audio that routing cannot deliver"
    assert "req-0" not in talker._active_states, "cleanup must still free the state"


def test_cleanup_warns_when_frames_are_left_undelivered(caplog: pytest.LogCaptureFixture) -> None:
    """An abort is the one case with no last step to emit on. Say so."""
    _talker, _codec, engine = _build()
    engine.add_request("req-0", max_tokens=4000)
    engine.add_request("req-1", max_tokens=4000)
    for _ in range(FIRST_CHUNK_FRAMES + 3):
        engine.step()
    engine.abort("req-0")

    target = logging.getLogger(LOGGER_NAME)
    target.addHandler(caplog.handler)
    prev = target.level
    target.setLevel(logging.WARNING)
    try:
        engine.step()  # cleanup for req-0 runs at the start of this step
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(prev)

    assert any("undelivered frame" in m for m in caplog.messages), (
        f"aborted request dropped audio without a warning; saw {caplog.messages}"
    )


# --------------------------------------------------------------------------
# A request's seed must reach the in-model sampler.
# --------------------------------------------------------------------------

_SEEDED_FRAMES = FIRST_CHUNK_FRAMES + 5


def _run_seeded(seed: int | None, *, neighbour: bool = False, neighbour_seed: int | None = None) -> torch.Tensor:
    """Audio delivered to ``req-a``, optionally sharing the batch with another."""
    _talker, engine = _build_with_real_sampler()
    engine.add_request("req-a", max_tokens=_SEEDED_FRAMES, seed=seed)
    if neighbour:
        engine.add_request("req-b", max_tokens=_SEEDED_FRAMES, seed=neighbour_seed)
    engine.run()
    return engine.audio_for("req-a")


def test_a_seeded_request_is_reproducible_within_one_process() -> None:
    """The bug this pins: the 32 heads are sampled in ``forward``, so a seed
    that only reaches vLLM's sampler leaves the audio at the mercy of the
    global RNG — two identical requests in one process then diverge."""
    torch.testing.assert_close(_run_seeded(1234), _run_seeded(1234))


def test_different_seeds_produce_different_audio() -> None:
    """Guards the opposite failure: a seed that is threaded but never used."""
    assert not torch.equal(_run_seeded(1234), _run_seeded(4321))


def test_a_seeded_request_is_unaffected_by_its_batch() -> None:
    """Each seeded request draws its own noise, so the noise it sees cannot
    depend on who else was scheduled alongside it.

    The stub backbone is what makes that checkable. A real one is a bf16
    batched matmul whose rows are not reproducible across batch shapes, so
    end-to-end batch invariance is not attainable there; this pins the half
    the model actually controls.
    """
    torch.testing.assert_close(_run_seeded(1234), _run_seeded(1234, neighbour=True, neighbour_seed=999))


def test_an_unseeded_neighbour_does_not_disturb_a_seeded_request() -> None:
    """The mixed case: one row draws from the global RNG, the other must not."""
    torch.testing.assert_close(_run_seeded(1234), _run_seeded(1234, neighbour=True))


# --------------------------------------------------------------------------
# Budget arithmetic, straight through the helper.
# --------------------------------------------------------------------------


def _info(**kwargs) -> _ReqInfo:
    base = {
        "request_id": "req-0",
        "is_prefill": False,
        "prompt_len": 20,
        "num_computed_tokens": 20,
        "max_tokens": 10,
        "seed": None,
    }
    base.update(kwargs)
    return _ReqInfo(**base)


def test_prefill_step_is_the_first_output_token() -> None:
    talker, _codec, _engine = _build()
    first = _info(is_prefill=True, num_computed_tokens=0, max_tokens=1)
    assert talker._is_last_output_token(first, span=20, samples_frame=True) is True
    more = _info(is_prefill=True, num_computed_tokens=0, max_tokens=2)
    assert talker._is_last_output_token(more, span=20, samples_frame=True) is False


def test_partial_prefill_chunk_does_not_consume_budget() -> None:
    """vLLM discards that row's token, so it must not count against max_tokens."""
    talker, _codec, _engine = _build()
    info = _info(is_prefill=True, num_computed_tokens=0, prompt_len=40, max_tokens=1)
    assert talker._is_last_output_token(info, span=10, samples_frame=False) is False


def test_budget_is_capped_by_max_model_len() -> None:
    """A max_tokens larger than the context window never gets reached."""
    talker, _codec, _engine = _build(max_model_len=30)
    # prompt 20 + 10 generated fills the window, whatever max_tokens says.
    at_cap = _info(num_computed_tokens=28, max_tokens=1000)
    assert talker._is_last_output_token(at_cap, span=1, samples_frame=True) is True
    below_cap = _info(num_computed_tokens=27, max_tokens=1000)
    assert talker._is_last_output_token(below_cap, span=1, samples_frame=True) is False


def test_missing_budget_is_not_treated_as_exhausted() -> None:
    """A runner that does not thread max_tokens must not trigger a fade."""
    talker, _codec, _engine = _build(max_model_len=0)
    assert talker._is_last_output_token(_info(max_tokens=None), span=1, samples_frame=True) is False
