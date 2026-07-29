# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dry-run of LongCat-Next's talker_mtp.

talker_mtp only ever executes on GPU inside a TP group, so nothing in the
unit suite used to reach it -- a GPU run died on its very first line
(``self.config.audio_config.vq_config``, where audio_config is a plain dict on
vllm-omni's shim config). These tests stub the TP group, audio_head and
embeddings so the whole body runs on CPU, which is where that class of
runtime-only AttributeError/shape bug is cheap to catch.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.longcat_next import modeling_longcat_next as mln
from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    AUDIOTEXT_PAD_TOKEN_ID,
    AUDIOTEXT_START_TOKEN_ID,
)

M = mln.LongcatNextForCausalLM

# Real values from the checkpoint's config.json.
CODEBOOK_SIZES = [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024]
AUDIO_OFFSET = 131125
HIDDEN = 8


class _FakeTPGroup:
    rank = 0

    @staticmethod
    def broadcast(tensor, src=0):  # single-rank: nothing to exchange
        return tensor


class _StubModel(nn.Module):
    """Real nn.Module, not a SimpleNamespace.

    talker_mtp and its helpers use real nn.Module machinery
    (register_buffer, nn.Embedding) -- a plain lambda stub for
    _ensure_replicated_audio_code_embedding previously masked a real bug
    (register_buffer raising KeyError because __init__ pre-declared the
    buffer's name as a plain `= None` attribute). Being a genuine nn.Module
    here is what makes that class of bug reachable from a CPU test.
    """

    _dbg_step = staticmethod(M._dbg_step)
    _sample_audio_code = M._sample_audio_code
    _ensure_replicated_audio_code_embedding = M._ensure_replicated_audio_code_embedding
    _ensure_audio_code_embed_module = M._ensure_audio_code_embed_module
    talker_mtp = M.talker_mtp

    def __init__(self, audio_head_fn, embed_tokens_fn, offsets):
        super().__init__()
        self._audio_gen: dict = {}
        self._audio_debug = False
        self._dbg_sampled = 0
        self._dbg_kept = 0
        self._dbg_emitted = 0
        self.audio_codebook_sizes = list(CODEBOOK_SIZES)
        self.register_buffer("audio_offset_vals", offsets, persistent=False)
        self.audio_head = audio_head_fn
        self.dtype = torch.float32
        self.max_gen = 750
        self.model = SimpleNamespace(embed_tokens=embed_tokens_fn)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(mln, "get_tp_group", lambda: _FakeTPGroup())

    offsets = torch.cumsum(
        torch.tensor([AUDIO_OFFSET] + CODEBOOK_SIZES[:-1], dtype=torch.long), dim=0
    )

    def audio_head(hidden, visual_tokens, visual_emb_layers, level):
        """Mimics the checkpoint's CasualDepthTransformerHead.forward.

        Deliberately faithful on the two things that actually broke a GPU run:
        visual_emb_layers is `.to()`-d and *called* (so None raises), and it is
        called once per level with globally-offset ids, so the indices must be
        in range for the embedding table it was handed.
        """
        assert visual_emb_layers is not None, "audio_head needs an embedding module"
        visual_emb_layers = visual_emb_layers.to(hidden.device)
        assert visual_tokens.shape[-1] == len(CODEBOOK_SIZES)
        stacked = torch.stack(
            [visual_emb_layers(visual_tokens[..., i]) for i in range(len(CODEBOOK_SIZES) - 1)],
            dim=1,
        )
        cumsum = torch.cumsum(stacked, dim=1)
        hidden_states = torch.concat([hidden.reshape(-1, 1, HIDDEN), cumsum], dim=1)
        assert hidden_states.size(1) == len(CODEBOOK_SIZES)
        return torch.randn(1, CODEBOOK_SIZES[level])

    def embed_tokens(tok):
        return torch.zeros(int(tok.reshape(-1).shape[0]), HIDDEN)

    return _StubModel(audio_head, embed_tokens, offsets)


def _state(**over):
    base = {
        "gen_step": 5,
        "audio_start": True,
        "text_end": False,
        "terminal": False,
        "delay": 0,
        "ext_id": AUDIOTEXT_PAD_TOKEN_ID,
    }
    base.update(over)
    return base


def _call(model, state, text_token=2620):
    model._audio_gen["r0"] = state
    return M.talker_mtp(
        model,
        input_ids=torch.tensor([text_token], dtype=torch.long),
        inputs_embeds=torch.zeros(1, HIDDEN),
        last_talker_hidden=torch.zeros(1, HIDDEN),
        text_step=torch.zeros(1, HIDDEN),
        req_ids=["r0"],
    )


def test_ensure_replicated_embedding_is_idempotent_across_calls(model):
    """Regression: nn.Module.register_buffer raises KeyError if the name is
    already a plain instance attribute -- even set to None. __init__ used to
    pre-declare `self._replicated_audio_code_embedding = None`, so the very
    first real call crashed with `KeyError: attribute '...' already exists`
    (only reached once a GPU run finally got past every earlier bug). Calling
    this twice, and via both of its production call sites
    (_ensure_audio_code_embed_module, and the direct call in the frame_kept
    branch), must not raise.
    """
    device = torch.device("cpu")

    first = model._ensure_replicated_audio_code_embedding(device)
    second = model._ensure_replicated_audio_code_embedding(device)
    assert first is second

    # A full talker_mtp call (audio_start=True) exercises both call sites
    # in one pass: _ensure_audio_code_embed_module (hoisted, every rank)
    # and the frame_kept branch's direct call.
    _call(model, _state())


def test_talker_mtp_runs_across_multiple_decode_steps(model):
    """talker_mtp is called once per decode step on the SAME model instance
    for the life of a request; nothing about repeated calls (buffer/module
    caching, per-row state mutation) should break on the second call."""
    state = _state(gen_step=5)
    for _ in range(3):
        _, codes = _call(model, state)
        assert codes.shape == (1, len(CODEBOOK_SIZES))


def test_talker_mtp_runs_and_returns_per_step_codes(model):
    """The whole body executes and returns this step's codes (not None, and
    not a running accumulation)."""
    embeds, codes = _call(model, _state())

    assert embeds.shape == (1, HIDDEN)
    assert codes is not None
    assert codes.shape == (1, len(CODEBOOK_SIZES))
    assert (codes >= 0).all(), "an audio_start frame should be kept, not -1"


def test_frame_discarded_before_audio_start(model):
    """delay=0 means step 0 is discarded; talker_mtp marks it all -1."""
    _, codes = _call(model, _state(gen_step=0, audio_start=False))

    assert (codes == -1).all()


def test_terminal_row_yields_no_codes(model):
    _, codes = _call(model, _state(terminal=True))

    assert (codes == -1).all()


def test_max_gen_cap_marks_terminal(model):
    """The reference force-ends a chunk at max_gen; without this the model can
    generate until the request's outer token budget runs out."""
    state = _state(gen_step=750)
    _call(model, state)

    assert state["terminal"] is True


def test_codes_stay_in_range_for_every_level(model):
    """Each level's sampled code must be a valid index into that level's
    codebook, since the embedding lookup offsets by level."""
    _, codes = _call(model, _state())

    for level, size in enumerate(CODEBOOK_SIZES):
        assert 0 <= int(codes[0, level]) < size


def test_text_end_and_ext_start_paths_execute(model):
    """Exercise the two masking branches (ext_id==audiotext_start, text_end)
    so neither can regress into a runtime error."""
    _, codes = _call(
        model, _state(text_end=True, ext_id=AUDIOTEXT_START_TOKEN_ID), text_token=AUDIOTEXT_PAD_TOKEN_ID
    )

    assert codes.shape == (1, len(CODEBOOK_SIZES))


def test_embedding_table_is_built_on_non_zero_ranks(model, monkeypatch):
    """Regression for a TP deadlock.

    The audio-code table is built from ``self.model.embed_tokens``, a
    VocabParallelEmbedding whose forward all-reduces across the TP group.
    Materialising it inside the ``rank == 0`` sampling block left ranks
    1..N-1 out of that collective and hung the group (RPC TimeoutError, a
    c10d::Work stack, and zero mtp log lines). Every rank must reach it, so a
    non-zero rank must still end up with the table built.
    """

    class _Rank1(_FakeTPGroup):
        rank = 1

    monkeypatch.setattr(mln, "get_tp_group", lambda: _Rank1())

    _call(model, _state())

    holder = model.__dict__.get("_audio_embed_holder", {})
    assert holder.get("module") is not None, (
        "non-zero rank skipped the embedding build -> it is inside a rank-0 "
        "guard and the embed_tokens all-reduce will deadlock"
    )


def test_empty_batch_short_circuits(model):
    embeds, codes = M.talker_mtp(
        model,
        input_ids=torch.zeros(0, dtype=torch.long),
        inputs_embeds=torch.zeros(0, HIDDEN),
        last_talker_hidden=torch.zeros(0, HIDDEN),
        text_step=torch.zeros(0, HIDDEN),
        req_ids=[],
    )

    assert codes is None
    assert embeds.shape == (0, HIDDEN)


# ------------------------------------------------------------------ #
# _sample_audio_code
# ------------------------------------------------------------------ #


class TestSampleAudioCode:
    """_sample_audio_code(self, logits, do_sample, temperature, top_k, top_p).

    A self-contained static method (no model state required), callable on any
    instance that carries the method.
    """

    VOCAB = 8192  # level-0 codebook width

    @pytest.fixture
    def logits(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        return torch.randn(self.VOCAB, generator=rng)

    def _call(self, model, logits, **kw):
        return M._sample_audio_code(model, logits, **kw)

    # -- argmax path ------------------------------------------------------- #

    def test_argmax_when_do_sample_false(self, model, logits):
        code = self._call(model, logits, do_sample=False)
        assert code.ndim == 0
        assert 0 <= int(code) < self.VOCAB

    def test_argmax_is_deterministic(self, model, logits):
        c1 = int(self._call(model, logits.clone(), do_sample=False))
        c2 = int(self._call(model, logits.clone(), do_sample=False))
        assert c1 == c2

    def test_argmax_picks_highest_logit(self, model):
        logits = torch.zeros(self.VOCAB)
        logits[42] = 10.0
        code = int(self._call(model, logits, do_sample=False))
        assert code == 42

    # -- greedy with temperature=0 ---------------------------------------- #

    def test_temperature_zero_falls_to_argmax(self, model, logits):
        """temperature=0 and do_sample=True should short-circuit to argmax."""
        code = self._call(model, logits, do_sample=True, temperature=0)
        argmax_code = int(self._call(model, logits, do_sample=False))
        assert int(code) == argmax_code

    # -- top-k truncation -------------------------------------------------- #

    def test_top_k_filters_outside_topk(self, model):
        logits = torch.arange(100, dtype=torch.float)
        for k in range(1, 10):
            code = int(self._call(model, logits, do_sample=False, top_k=k, top_p=1.0))
            assert code <= k - 1, f"top_k={k} allowed code {code}"

    def test_top_k_clamps_to_vocab_size(self, model, logits):
        """When top_k > vocab, it should be clamped to vocab size, not error."""
        code = self._call(model, logits, do_sample=False, top_k=self.VOCAB * 2)
        assert 0 <= int(code) < self.VOCAB

    # -- top-p truncation -------------------------------------------------- #

    def test_top_p_does_not_raise(self, model, logits):
        code = self._call(model, logits, do_sample=False, top_p=0.5)
        assert 0 <= int(code) < self.VOCAB

    # -- combined top-k + top-p -------------------------------------------- #

    def test_top_k_top_p_produces_valid_code(self, model, logits):
        code = self._call(
            model, logits, do_sample=True, temperature=1.0, top_k=50, top_p=0.9
        )
        assert 0 <= int(code) < self.VOCAB

    # -- multinomial returns valid range ----------------------------------- #

    def test_multinomial_samples_within_range(self, model, logits):
        codes = [
            int(self._call(model, logits, do_sample=True, temperature=0.8, top_k=0, top_p=1.0))
            for _ in range(100)
        ]
        assert all(0 <= c < self.VOCAB for c in codes)
        assert len(set(codes)) > 1, "multinomial should not be deterministic"

    # -- temperature scaling changes distribution -------------------------- #

    def test_higher_temperature_increases_diversity(self, model):
        rng = torch.Generator()
        rng.manual_seed(0)
        logits = torch.full((self.VOCAB,), 0.0, generator=rng)
        logits[:10] = 1.0  # only first 10 have non-negligible prob
        cold = [
            int(self._call(model, logits.clone(), do_sample=True, temperature=0.1, top_k=0, top_p=1.0))
            for _ in range(50)
        ]
        hot = [
            int(self._call(model, logits.clone(), do_sample=True, temperature=5.0, top_k=0, top_p=1.0))
            for _ in range(50)
        ]
        cold_unique = len(set(cold))
        hot_unique = len(set(hot))
        assert hot_unique >= cold_unique, (
            f"higher temperature should spread mass wider "
            f"(cold={cold_unique} unique vs hot={hot_unique})"
        )


# ------------------------------------------------------------------ #
# compute_logits
# ------------------------------------------------------------------ #


class _LogitsModel:
    """Minimal model stub for compute_logits logic.

    __init__ signature must match what the test calls (no args for fixture).
    """

    def __init__(self):
        self._audio_gen: dict = {}
        self._eos_id = 2
        self.logits_processor = _FakeLogitsProcessor()
        self.lm_head = None


class _FakeLogitsProcessor:
    """Stub for vllm's LogitsProcessor.__call__(lm_head, hidden).

    Returns hidden_states reshaped to [n, vocab] as logits.
    """

    def __call__(self, lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
        vocab = 131125
        if hidden_states.dim() == 2 and hidden_states.shape[-1] == 1:
            return hidden_states.expand(-1, vocab)
        return hidden_states.new_zeros(hidden_states.shape[0], vocab)


_AUDIOGEN_START_TOKEN_ID = 131123
_AUDIOTEXT_PAD_TOKEN_ID = 131122


@pytest.fixture
def logits_model():
    return _LogitsModel()


def _add_audio_state(m, req_id="r0", **over):
    state = {"terminal": False, "text_end": False, "gen_step": 5}
    state.update(over)
    m._audio_gen[req_id] = state
    return state


class TestComputeLogits:
    """compute_logits(self, hidden_states) -> Tensor | None.

    Suppresses EOS during audio generation; forces all logits to -inf (except
    AUDIOTEXT_PAD) once text_end fires.
    """

    VOCAB = 131125

    def test_returns_logits_when_no_audio_gen(self, logits_model):
        hidden = torch.zeros(1, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out is not None
        assert out.shape == (1, self.VOCAB)

    def test_returns_none_when_logits_processor_returns_none(self, logits_model):
        logits_model.logits_processor = lambda *a: None
        out = M.compute_logits(logits_model, torch.zeros(1, 4096))
        assert out is None

    def test_suppresses_eos_for_active_request(self, logits_model):
        _add_audio_state(logits_model, "r0", terminal=False)
        hidden = torch.zeros(1, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out[0, logits_model._eos_id] == float("-inf")

    def test_does_not_suppress_eos_for_terminal_request(self, logits_model):
        _add_audio_state(logits_model, "r0", terminal=True)
        hidden = torch.zeros(1, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out[0, logits_model._eos_id] != float("-inf")

    def test_does_not_suppress_eos_for_request_not_in_audio_gen(self, logits_model):
        hidden = torch.zeros(1, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out[0, logits_model._eos_id] != float("-inf")

    def test_forces_text_end_row_to_pad_only(self, logits_model):
        _add_audio_state(logits_model, "r0", text_end=True)
        hidden = torch.zeros(1, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out[0, _AUDIOTEXT_PAD_TOKEN_ID] == 0.0
        assert (out[0, :] == float("-inf")).sum() >= self.VOCAB - 2, (
            "text_end should force all entries to -inf except the pad token"
        )

    def test_eos_suppression_respects_eos_id_bound(self, logits_model):
        logits_model._eos_id = self.VOCAB + 10  # out of range
        _add_audio_state(logits_model, "r0")
        hidden = torch.zeros(1, 4096)
        # Should not IndexError
        out = M.compute_logits(logits_model, hidden)
        assert out is not None

    def test_row_count_mismatch_skips_suppression(self, logits_model):
        """num_logits != len(_audio_gen) — prefill or multi-batch skip."""
        _add_audio_state(logits_model, "r0")
        hidden = torch.zeros(3, 4096)  # 3 logits rows but only 1 audio request
        out = M.compute_logits(logits_model, hidden)
        # EOS should NOT be suppressed because guard returned early
        assert out[0, logits_model._eos_id] != float("-inf")

    def test_multiple_requests_all_get_suppression(self, logits_model):
        _add_audio_state(logits_model, "r0", terminal=False)
        _add_audio_state(logits_model, "r1", terminal=False)
        hidden = torch.zeros(2, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out[0, logits_model._eos_id] == float("-inf")
        assert out[1, logits_model._eos_id] == float("-inf")

    def test_mixed_terminal_and_active(self, logits_model):
        _add_audio_state(logits_model, "r0", terminal=True)
        _add_audio_state(logits_model, "r1", terminal=False)
        hidden = torch.zeros(2, 4096)
        out = M.compute_logits(logits_model, hidden)
        assert out[0, logits_model._eos_id] != float("-inf")  # terminal: untouched
        assert out[1, logits_model._eos_id] == float("-inf")  # active: suppressed
