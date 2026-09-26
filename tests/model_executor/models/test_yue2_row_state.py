# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Row-to-request lifecycle tests for the YuE2 model-owned sampler.

YuE2's sampler owns per-request state keyed by request id, because vLLM
compacts the persistent batch every step and a row index is only valid within
one step. Two invariants carry the song's correctness and are pinned here:

* a row only samples once its whole prompt is computed — including the
  prefix-cache case where the completing prefill row arrives with
  ``computed > 0`` and a partial token slice;
* the request ends exactly once, either on its phase's end token (clean) or on
  the frame budget (truncated), and never samples again afterwards.

The model class is instantiated with ``object.__new__`` so these tests run on
CPU with no vLLM engine, weight load or CUDA: the methods under test only
touch the per-request dicts set up here. ``_finish_request`` (the ODE+VAE
pass) is stubbed and its calls recorded.
"""

import pytest
import torch

from vllm_omni.model_executor.models.yue2.constants import (
    ABC_END,
    CODEC_OFFSET,
    KEY_MIN_TOKENS,
    KEY_PHASE,
    KEY_PREFIX_IDS,
    KEY_SEED,
    KEY_SKIP_SYNTHESIS,
    KEY_TEMPERATURE,
    MUSIC_END,
    SEMANTIC_SAMPLING,
    VOCAB_SIZE,
)
from vllm_omni.model_executor.models.yue2.yue2 import (
    Yue2ForCausalLM,
    _RequestState,
    _RowConstants,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_model() -> Yue2ForCausalLM:
    model = object.__new__(Yue2ForCausalLM)
    model._states = {}
    model._row_constants = {}
    model._step_rows = []
    model._decode_t0 = {}
    return model


def make_constants(request_id: str, **overrides) -> _RowConstants:
    defaults = dict(
        request_id=request_id,
        phase="semantic",
        temperature=0.0,  # greedy: assertions do not depend on RNG order
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        penalty_window=50,
        min_tokens=0,
        max_audio_frames=100,
        seed=831001,
        skip_synthesis=True,  # the real finish path is stubbed; keep it off
    )
    defaults.update(overrides)
    return _RowConstants(**defaults)


def make_state(request_id: str, prompt_len: int = 63, **overrides) -> _RequestState:
    return _RequestState(
        request_id=request_id,
        constants=make_constants(request_id, **overrides),
        prompt_tokens=prompt_len,
        prompt_len=prompt_len,
        prefix_ids=list(range(prompt_len)),
        generator=torch.Generator().manual_seed(1),
    )


def logits_for(*favorites: int, rows: int = 1) -> torch.Tensor:
    """[rows, vocab] logits where row r argmaxes favorites[r] (codec/end ids)."""
    logits = torch.full((rows, VOCAB_SIZE), -10.0)
    for row, token in enumerate(favorites):
        logits[row, token] = 10.0
    return logits


@pytest.fixture()
def finish_calls():
    calls: list[tuple[str, bool]] = []

    def recorder(state, *, hit_end: bool):
        calls.append((state.request_id, hit_end))

    return calls, recorder


class TestSampleRows:
    def test_decode_row_samples_and_appends_history(self, finish_calls):
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        state = make_state("r1")
        model._states["r1"] = state
        model._step_rows = [("r1", 63, 1)]  # full prompt computed, one new token
        out = model.sample(logits_for(CODEC_OFFSET + 5), None)
        assert out.sampled_token_ids[0, 0].item() == CODEC_OFFSET + 5
        assert state.history == [CODEC_OFFSET + 5]
        assert calls == []

    def test_mid_chunk_prefill_row_does_not_sample(self, finish_calls):
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        state = make_state("r1", prompt_len=63)
        model._states["r1"] = state
        model._step_rows = [("r1", 0, 32)]  # 0 + 32 < 63: chunk 1 of 2
        out = model.sample(logits_for(CODEC_OFFSET + 5), None)
        # vLLM discards this row's token; it must not touch the song state.
        assert out.sampled_token_ids[0, 0].item() == 0
        assert state.history == []
        assert calls == []

    def test_prefix_cache_completing_row_samples(self, finish_calls):
        """comp > 0 with the uncached tail still reaching the prompt end."""
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        model._states["r1"] = make_state("r1", prompt_len=63)
        model._step_rows = [("r1", 50, 13)]  # 50 cached + 13 tail == 63
        out = model.sample(logits_for(CODEC_OFFSET + 5), None)
        assert out.sampled_token_ids[0, 0].item() == CODEC_OFFSET + 5

    def test_end_token_finishes_once_and_is_not_history(self, finish_calls):
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        state = make_state("r1")
        state.history = [CODEC_OFFSET + 1, CODEC_OFFSET + 2]
        model._states["r1"] = state
        model._step_rows = [("r1", 63, 1)]
        out = model.sample(logits_for(MUSIC_END), None)
        assert out.sampled_token_ids[0, 0].item() == MUSIC_END
        assert state.finished and not state.truncated
        assert state.history == [CODEC_OFFSET + 1, CODEC_OFFSET + 2]  # end not kept
        assert calls == [("r1", True)]
        # The engine runs one more step before retiring the request: no
        # second finish, no phantom token.
        model._step_rows = [("r1", 64, 1)]
        out2 = model.sample(logits_for(MUSIC_END), None)
        assert calls == [("r1", True)]
        assert out2.sampled_token_ids[0, 0].item() == 0

    def test_abc_phase_stops_on_abc_end(self, finish_calls):
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        state = make_state("r1", phase="abc", min_tokens=0)
        model._states["r1"] = state
        model._step_rows = [("r1", 63, 1)]
        out = model.sample(logits_for(ABC_END), None)
        assert out.sampled_token_ids[0, 0].item() == ABC_END
        assert state.finished

    def test_frame_budget_truncates_semantic_only(self, finish_calls):
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        # semantic: history at budget-1, one more codec token crosses it
        state = make_state("r1", max_audio_frames=4)
        state.history = [CODEC_OFFSET + 1, CODEC_OFFSET + 2, CODEC_OFFSET + 3]
        model._states["r1"] = state
        model._step_rows = [("r1", 63, 1)]
        out = model.sample(logits_for(CODEC_OFFSET + 4), None)
        assert state.finished and state.truncated
        assert out.sampled_token_ids[0, 0].item() == MUSIC_END  # engine-level stop
        assert calls == [("r1", False)]
        assert len(state.history) == 4

    def test_abc_phase_ignores_frame_budget(self, finish_calls):
        """The budget is a semantic-phase concept: abc ids are text tokens."""
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        state = make_state("r1", phase="abc", max_audio_frames=2)
        state.history = [100, 200, 300]  # already past 2; abc keeps going
        model._states["r1"] = state
        model._step_rows = [("r1", 63, 1)]
        out = model.sample(logits_for(400), None)
        assert not state.finished
        assert out.sampled_token_ids[0, 0].item() == 400
        assert calls == []

    def test_two_requests_track_independent_songs(self, finish_calls):
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        model._states["a"] = make_state("a")
        model._states["b"] = make_state("b")
        model._step_rows = [("a", 63, 1), ("b", 63, 1)]
        out = model.sample(logits_for(CODEC_OFFSET + 5, CODEC_OFFSET + 7, rows=2), None)
        assert out.sampled_token_ids[0, 0].item() == CODEC_OFFSET + 5
        assert out.sampled_token_ids[1, 0].item() == CODEC_OFFSET + 7
        assert model._states["a"].history == [CODEC_OFFSET + 5]
        assert model._states["b"].history == [CODEC_OFFSET + 7]

    def test_unknown_request_row_is_left_at_zero(self, finish_calls):
        """A row whose state never got created (e.g. aborted) must not crash."""
        calls, recorder = finish_calls
        model = make_model()
        model._finish_request = recorder
        model._step_rows = [("ghost", 63, 1)]
        out = model.sample(logits_for(CODEC_OFFSET + 5), None)
        assert out.sampled_token_ids[0, 0].item() == 0
        assert calls == []

    def test_synthesis_failure_fails_only_that_request(self, finish_calls):
        """An NAR/VAE exception must not escape sample(): the runner calls
        model.sample() with no error handling, so one failing request would
        otherwise kill Stage-0 for every live request."""
        calls, _recorder = finish_calls
        model = make_model()
        model._last_mm = None
        model._audio_queue = []

        def boom(state, *, hit_end: bool):
            if state.request_id == "bad":
                raise torch.OutOfMemoryError("NAR OOM")
            calls.append((state.request_id, hit_end))

        model._finish_request = boom
        model._states["good"] = make_state("good")
        model._states["bad"] = make_state("bad")
        model._step_rows = [("good", 63, 1), ("bad", 63, 1)]
        out = model.sample(logits_for(MUSIC_END, MUSIC_END, rows=2), None)
        assert out.sampled_token_ids[0, 0].item() == MUSIC_END
        assert out.sampled_token_ids[1, 0].item() == MUSIC_END
        assert calls == [("good", True)]
        assert model._states["good"].finished and model._states["bad"].finished
        # The failed request ships an empty clip flagged as an error, so
        # serving answers 500 instead of a silent zero-length WAV.
        errors = {rid: err for rid, _audio, _trunc, err in model._audio_queue}
        assert errors == {"bad": True}

    def test_failed_audio_rides_make_omni_output_meta(self):
        """The error flag must survive the queue fallback path into the mm
        payload; the wire keeps ints (strings are dropped), so meta.error
        travels as 0/1 next to meta.truncated."""
        model = make_model()
        model._last_mm = None
        model._audio_queue = [("r1", torch.zeros((2, 0)), False, True)]
        out = model.make_omni_output(torch.zeros(1))
        meta = out.multimodal_outputs["meta"]
        assert meta["error"] == [1]
        assert meta["truncated"] == [0]
        assert model._audio_queue == []

    def test_empty_step_returns_neutral_tokens(self):
        model = make_model()
        model._step_rows = []
        out = model.sample(logits_for(CODEC_OFFSET + 5), None)
        assert out.sampled_token_ids.shape == (1, 1)
        # The runner's input_ids buffer is int32; long ids crash its scatter.
        assert out.sampled_token_ids.dtype == torch.int32


class TestCaptureConstants:
    def _model(self):
        model = make_model()
        return model

    def _kwargs(self, args):
        return {"sampling_extra_args": [args]}

    def test_prefix_ids_shipped_in_extra_args_win_over_the_scheduled_slice(self):
        model = self._model()
        full = list(range(63))
        model._step_rows = [("r1", 50, 13)]  # 50 cached, 13-token tail scheduled
        tail = torch.tensor([900 + i for i in range(13)])
        model._capture_constants(
            self._kwargs({KEY_PREFIX_IDS: full, KEY_PHASE: "semantic", KEY_SEED: 7}),
            tail,
        )
        state = model._states["r1"]
        assert state.prompt_len == 63
        assert state.prefix_ids == full  # NAR conditioning needs the WHOLE prefix
        assert state.constants.seed == 7

    def test_legacy_path_takes_the_scheduled_slice(self):
        model = self._model()
        model._step_rows = [("r1", 0, 3)]
        model._capture_constants(self._kwargs(None), torch.tensor([5, 6, 7]))
        state = model._states["r1"]
        assert state.prefix_ids == [5, 6, 7]
        assert state.prompt_len == 3

    def test_cache_hit_without_shipped_ids_creates_no_state(self):
        model = self._model()
        model._step_rows = [("r1", 50, 13)]
        model._capture_constants(self._kwargs({KEY_PHASE: "semantic"}), torch.zeros(13, dtype=torch.long))
        assert "r1" not in model._states
        assert "r1" not in model._row_constants

    def test_phase_presets_and_overrides(self):
        model = self._model()
        model._step_rows = [("abc", 0, 3), ("sem", 0, 3)]
        model._capture_constants(
            {"sampling_extra_args": [{KEY_PHASE: "abc"}, None]},
            torch.arange(6),
        )
        abc = model._states["abc"].constants
        sem = model._states["sem"].constants
        # abc preset (upstream yue2_generation_config): temperature 0.7, and
        # the abc phase produces tokens only — no synthesis.
        assert abc.temperature == 0.7 and abc.skip_synthesis is True
        assert abc.phase == "abc"
        # semantic preset: temperature 1.0, min_tokens 200, synthesis on.
        assert sem.temperature == 1.0 and sem.skip_synthesis is False
        assert sem.min_tokens == SEMANTIC_SAMPLING["min_tokens"]
        # explicit overrides beat presets
        model2 = self._model()
        model2._step_rows = [("r", 0, 3)]
        model2._capture_constants(
            self._kwargs({KEY_TEMPERATURE: 0.3, KEY_MIN_TOKENS: 5, KEY_SKIP_SYNTHESIS: False}),
            torch.zeros(3, dtype=torch.long),
        )
        c = model2._states["r"].constants
        assert (c.temperature, c.min_tokens, c.skip_synthesis) == (0.3, 5, False)

    def test_state_is_created_once_per_request(self):
        model = self._model()
        model._step_rows = [("r1", 0, 3)]
        args = {KEY_PREFIX_IDS: [1, 2, 3], KEY_PHASE: "semantic"}
        model._capture_constants(self._kwargs(args), torch.zeros(3, dtype=torch.long))
        first = model._states["r1"]
        # A later step (decode rows) must not reset the seeded generator or
        # the accumulated history by recreating the state.
        first.history.append(CODEC_OFFSET)
        model._step_rows = [("r1", 3, 1)]
        model._capture_constants(self._kwargs(args), torch.zeros(1, dtype=torch.long))
        assert model._states["r1"] is first
        assert model._states["r1"].history == [CODEC_OFFSET]

    def test_default_seed_matches_the_reference_preset(self):
        model = self._model()
        model._step_rows = [("r1", 0, 3)]
        model._capture_constants(self._kwargs({KEY_PHASE: "semantic"}), torch.zeros(3, dtype=torch.long))
        assert model._states["r1"].constants.seed == 831001

    def test_row_alignment_follows_step_rows_order(self):
        model = self._model()
        model._step_rows = [("a", 0, 2), ("b", 0, 3)]
        model._capture_constants(
            {
                "sampling_extra_args": [
                    {KEY_PREFIX_IDS: [1, 2], KEY_SEED: 1},
                    {KEY_PREFIX_IDS: [3, 4, 5], KEY_SEED: 2},
                ]
            },
            torch.zeros(5, dtype=torch.long),
        )
        assert model._states["a"].constants.seed == 1
        assert model._states["b"].constants.seed == 2
        assert model._states["a"].prefix_ids == [1, 2]
        assert model._states["b"].prefix_ids == [3, 4, 5]
