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

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from vllm_omni.model_executor.models.longcat_next import modeling_longcat_next as mln
from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    AUDIOTEXT_PAD_TOKEN_ID,
    AUDIOTEXT_START_TOKEN_ID,
    IMG_NEWLINE_TOKEN_ID,
    IMG_PAD_TOKEN_ID,
)

M = mln.LongcatNextForCausalLM

# Real values from the checkpoint's config.json.
CODEBOOK_SIZES = [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024]
AUDIO_OFFSET = 131125
# Visual: every level is 16384-wide (config.json's visual_config.vq_config).
VISUAL_CODEBOOK_SIZES = [16384] * 8
VISUAL_OFFSET = 150581
HIDDEN = 8

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
    _sample_depth_head = M._sample_depth_head
    _ensure_replicated_audio_code_embedding = M._ensure_replicated_audio_code_embedding
    _ensure_audio_code_embed_module = M._ensure_audio_code_embed_module
    _ensure_replicated_visual_code_embedding = M._ensure_replicated_visual_code_embedding
    _ensure_visual_code_embed_module = M._ensure_visual_code_embed_module
    _code_embeddings = M._code_embeddings
    talker_mtp = M.talker_mtp

    def __init__(self, audio_head_fn, visual_head_fn, embed_tokens_fn, offsets, visual_offsets):
        super().__init__()
        self._audio_gen: dict = {}
        self._visual_gen: dict = {}
        self._audio_debug = False
        self._dbg_sampled = 0
        self._dbg_kept = 0
        self._dbg_emitted = 0
        self.audio_codebook_sizes = list(CODEBOOK_SIZES)
        self.visual_codebook_sizes = list(VISUAL_CODEBOOK_SIZES)
        self.register_buffer("audio_offset_vals", offsets, persistent=False)
        self.register_buffer("visual_offset_vals", visual_offsets, persistent=False)
        self.audio_head = audio_head_fn
        self.visual_head = visual_head_fn
        self.dtype = torch.float32
        self.max_gen = 750
        self.model = SimpleNamespace(embed_tokens=embed_tokens_fn)
        # visual_embedding_layer: the real DecoderLayer bridge just refines
        # an existing embedding tensor (pre_layernorm + MLP residual) --
        # identity is a faithful-enough stand-in for shape/crash testing,
        # matching this suite's existing philosophy (structural correctness
        # on CPU, exact numerics reserved for GPU runs against real weights).
        self.visual_tokenizer = SimpleNamespace(visual_embedding_layer=lambda x: x)


def _depth_head_fn(codebook_sizes):
    def head(hidden, tokens, emb_layers, level):
        """Mimics the checkpoint's CasualDepthTransformerHead.forward.

        Deliberately faithful on the two things that actually broke a GPU run:
        emb_layers is `.to()`-d and *called* (so None raises), and it is
        called once per level with globally-offset ids, so the indices must be
        in range for the embedding table it was handed. Shared between
        audio_head and visual_head stubs since it's the same checkpoint class.
        """
        assert emb_layers is not None, "depth head needs an embedding module"
        emb_layers = emb_layers.to(hidden.device)
        assert tokens.shape[-1] == len(codebook_sizes)
        stacked = torch.stack(
            [emb_layers(tokens[..., i]) for i in range(len(codebook_sizes) - 1)],
            dim=1,
        )
        cumsum = torch.cumsum(stacked, dim=1)
        hidden_states = torch.concat([hidden.reshape(-1, 1, HIDDEN), cumsum], dim=1)
        assert hidden_states.size(1) == len(codebook_sizes)
        return torch.randn(1, codebook_sizes[level])

    return head


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(mln, "get_tp_group", lambda: _FakeTPGroup())

    offsets = torch.cumsum(
        torch.tensor([AUDIO_OFFSET] + CODEBOOK_SIZES[:-1], dtype=torch.long), dim=0
    )
    visual_offsets = torch.cumsum(
        torch.tensor([VISUAL_OFFSET] + VISUAL_CODEBOOK_SIZES[:-1], dtype=torch.long), dim=0
    )

    def embed_tokens(tok):
        return torch.zeros(int(tok.reshape(-1).shape[0]), HIDDEN)

    return _StubModel(
        _depth_head_fn(CODEBOOK_SIZES), _depth_head_fn(VISUAL_CODEBOOK_SIZES),
        embed_tokens, offsets, visual_offsets,
    )


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


def test_talker_mtp_none_sampling_params_sample_not_greedy(model):
    """The runner passes explicit None sampling keys when subtalker_sampling_params
    is unset; talker_mtp must coalesce them to the checkpoint defaults. Otherwise
    do_sample=None collapses to greedy argmax (the single-repeated-code run)."""
    # level-0 logits peaked on two near-equal codes: argmax is always 0, but
    # top_k/top_p sampling (audio defaults) must spread draws over {0, 1}.
    def rigged_head(hidden, tokens, emb_layers, level):
        if level == 0:
            logits = torch.full((1, CODEBOOK_SIZES[0]), -100.0)
            logits[0, 0] = 0.3
            logits[0, 1] = -0.3
            return logits
        return torch.randn(1, CODEBOOK_SIZES[level])

    model.audio_head = rigged_head
    model._audio_gen["r0"] = _state()
    seen = set()
    for _ in range(60):
        _, codes = M.talker_mtp(
            model,
            input_ids=torch.tensor([2620], dtype=torch.long),
            inputs_embeds=torch.zeros(1, HIDDEN),
            last_talker_hidden=torch.zeros(1, HIDDEN),
            text_step=torch.zeros(1, HIDDEN),
            req_ids=["r0"],
            do_sample=None,
            temperature=None,
            top_k=None,
            top_p=None,
        )
        seen.add(int(codes[0, 0]))
    assert 0 in seen and 1 in seen, (
        f"None sampling params must sample both top codes, got {sorted(seen)}"
    )


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
# talker_mtp -- visual (image-gen) dispatch branch
# ------------------------------------------------------------------ #


def _visual_state(**over):
    base = {"gen_step": 3, "token_w": 4, "ext_id": IMG_PAD_TOKEN_ID, "terminal": False}
    base.update(over)
    return base


def _call_visual(model, state, text_token=None):
    if text_token is None:
        text_token = state.get("ext_id", IMG_PAD_TOKEN_ID)
    model._visual_gen["r0"] = state
    return M.talker_mtp(
        model,
        input_ids=torch.tensor([text_token], dtype=torch.long),
        inputs_embeds=torch.zeros(1, HIDDEN),
        last_talker_hidden=torch.zeros(1, HIDDEN),
        text_step=torch.zeros(1, HIDDEN),
        req_ids=["r0"],
    )


def test_visual_mtp_runs_and_returns_per_step_codes(model):
    """A real-pixel step (ext_id=IMG_PAD, not a row boundary) samples and
    keeps a code, mirroring test_talker_mtp_runs_and_returns_per_step_codes
    but for the visual dispatch branch."""
    embeds, codes = _call_visual(model, _visual_state())

    assert embeds.shape == (1, HIDDEN)
    assert codes is not None
    assert codes.shape == (1, len(VISUAL_CODEBOOK_SIZES))
    assert (codes >= 0).all(), "a real-pixel step should be kept, not -1"


def test_visual_mtp_discards_row_boundary_frame(model):
    """A row-boundary step (ext_id=IMG_NEWLINE) never carries a real pixel,
    regardless of what visual_head sampled -- mirrors the reference's
    output_processor.py discarding tmp_multi_ids at row boundaries."""
    _, codes = _call_visual(model, _visual_state(ext_id=IMG_NEWLINE_TOKEN_ID))

    assert (codes == -1).all()


def test_visual_mtp_terminal_row_yields_no_codes(model):
    _, codes = _call_visual(model, _visual_state(terminal=True))

    assert (codes == -1).all()


def test_visual_mtp_codes_stay_in_range_for_every_level(model):
    """Each level's sampled code must be a valid index into that level's
    codebook (all 16384-wide for visual), since the embedding lookup
    (_code_embeddings + visual_embedding_layer) offsets by level."""
    _, codes = _call_visual(model, _visual_state())

    for level, size in enumerate(VISUAL_CODEBOOK_SIZES):
        assert 0 <= int(codes[0, level]) < size


def test_visual_mtp_masks_end_sentinel(model):
    """The level-0 end-of-image sentinel class (codebook_sizes[0]=16384) is
    masked for the visual head (reference output_processor.py:312), so the
    image can never self-terminate -- the grid bound is the sole terminator.
    Rig the head to peak at the sentinel and confirm the code is forced away
    and no terminal fires."""

    def rigged_head(hidden, tokens, emb_layers, level):
        if level == 0:
            logits = torch.full((1, VISUAL_CODEBOOK_SIZES[0] + 1), -100.0)
            logits[0, VISUAL_CODEBOOK_SIZES[0]] = 100.0  # sentinel: masked
            logits[0, 0] = 50.0                          # next-best survivor
            return logits
        return torch.zeros(1, VISUAL_CODEBOOK_SIZES[level])

    model.visual_head = rigged_head
    state = _visual_state()
    _, codes = _call_visual(model, state, text_token=IMG_PAD_TOKEN_ID)

    assert int(codes[0, 0]) != VISUAL_CODEBOOK_SIZES[0], "sentinel must be masked"
    assert int(codes[0, 0]) == 0
    assert state["terminal"] is False


def test_advance_visual_gen_grid_bound_terminates(model):
    """The image must stop deterministically once the token_h x token_w grid is
    complete (token_h*(token_w+1) steps), not overrun. The final (spurious)
    trailing row-boundary newline becomes IMAGE_END + terminal so the visible
    stream closes the image."""
    model.config = SimpleNamespace(hidden_size=HIDDEN)
    M._advance_visual_gen(
        model, "r0", last_token=IMG_START_TOKEN_ID,
        device=torch.device("cpu"), dtype=torch.float32,
        token_w=2, token_h=2,
    )
    state = model._visual_gen["r0"]
    assert state["token_w"] == 2 and state["token_h"] == 2

    exts = []
    for _ in range(6):
        M._advance_visual_gen(
            model, "r0", last_token=IMG_PAD_TOKEN_ID,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        exts.append(state["ext_id"])

    assert exts == [
        IMG_PAD_TOKEN_ID, IMG_PAD_TOKEN_ID, IMG_NEWLINE_TOKEN_ID,
        IMG_PAD_TOKEN_ID, IMG_PAD_TOKEN_ID, IMG_END_TOKEN_ID,
    ]
    assert state["terminal"] is True


def test_talker_mtp_audio_repetition_penalty_changes_greedy_code(model):
    """Audio repetition penalty (default 1.3) is plumbed end-to-end: per-request
    past_codes accumulate across kept frames and the level-0 argmax flips away
    from a code once it enters the history (before the fix it repeated forever)."""
    def rigged_head(hidden, tokens, emb_layers, level):
        if level == 0:
            logits = torch.full((1, CODEBOOK_SIZES[0]), -100.0)
            logits[0, 0] = 0.5
            logits[0, 1] = 0.45
            return logits
        return torch.zeros(1, CODEBOOK_SIZES[level])

    model.audio_head = rigged_head
    state = _state()
    model._audio_gen["r0"] = state

    kwargs = dict(
        input_ids=torch.tensor([2620], dtype=torch.long),
        inputs_embeds=torch.zeros(1, HIDDEN),
        last_talker_hidden=torch.zeros(1, HIDDEN),
        text_step=torch.zeros(1, HIDDEN),
        req_ids=["r0"],
        do_sample=False,
        repetition_penalty=1.3,
    )
    _, codes1 = M.talker_mtp(model, **kwargs)
    assert int(codes1[0, 0]) == 0

    _, codes2 = M.talker_mtp(model, **kwargs)
    assert int(codes2[0, 0]) == 1, "repeated code must be penalised"
    assert len(state["past_codes"]) == 2


def test_visual_mtp_defaults_to_no_rep_penalty(model):
    """Visual repetition_penalty defaults to 1.0, so with a head peaking at one
    code the greedy choice repeats (matching the reference's image-case 1.0)."""
    def rigged_head(hidden, tokens, emb_layers, level):
        if level == 0:
            logits = torch.full((1, VISUAL_CODEBOOK_SIZES[0]), -100.0)
            logits[0, 0] = 0.5
            logits[0, 1] = 0.45
            return logits
        return torch.zeros(1, VISUAL_CODEBOOK_SIZES[level])

    model.visual_head = rigged_head
    state = _visual_state()
    model._visual_gen["r0"] = state

    kwargs = dict(
        input_ids=torch.tensor([IMG_PAD_TOKEN_ID], dtype=torch.long),
        inputs_embeds=torch.zeros(1, HIDDEN),
        last_talker_hidden=torch.zeros(1, HIDDEN),
        text_step=torch.zeros(1, HIDDEN),
        req_ids=["r0"],
        do_sample=False,
    )
    _, codes1 = M.talker_mtp(model, **kwargs)
    _, codes2 = M.talker_mtp(model, **kwargs)
    assert int(codes1[0, 0]) == 0
    assert int(codes2[0, 0]) == 0, "penalty 1.0 must not flip the argmax"


def test_visual_mtp_runs_across_multiple_decode_steps(model):
    """Same rationale as the audio equivalent: talker_mtp is called once per
    decode step on the same model instance for the life of a request."""
    state = _visual_state()
    for _ in range(3):
        _, codes = _call_visual(model, state)
        assert codes.shape == (1, len(VISUAL_CODEBOOK_SIZES))


def test_audio_and_visual_gen_are_mutually_exclusive_per_row(model):
    """A request in _visual_gen must not also be treated as an audio
    request (and vice versa) -- talker_mtp checks audio_state first, so
    this pins that a visual-only request never takes the audio branch."""
    model._visual_gen["r0"] = _visual_state()
    assert model._audio_gen.get("r0") is None
    _, codes = M.talker_mtp(
        model,
        input_ids=torch.tensor([IMG_PAD_TOKEN_ID], dtype=torch.long),
        inputs_embeds=torch.zeros(1, HIDDEN),
        last_talker_hidden=torch.zeros(1, HIDDEN),
        text_step=torch.zeros(1, HIDDEN),
        req_ids=["r0"],
    )
    assert codes.shape == (1, len(VISUAL_CODEBOOK_SIZES))


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

    # -- None coalescing (runner's subtalker_sampling_params unset) -------- #

    def test_none_sampling_params_coalesce_to_defaults_and_sample(self, model, logits):
        """Regression: the runner passes explicit None for every sampling key
        when subtalker_sampling_params is unset (gpu_model_runner.py:1835-1901).
        `if do_sample and temperature > 0` short-circuits on None into greedy
        argmax (the single-repeated-code collapse), and a truthy do_sample
        with temperature=None would raise TypeError. All-None must resolve to
        the audio defaults (temperature=0.5/top_k=5/top_p=0.85) and sample."""
        codes = [
            int(self._call(model, logits, do_sample=None, temperature=None, top_k=None, top_p=None))
            for _ in range(50)
        ]
        assert all(0 <= c < self.VOCAB for c in codes)
        assert len(set(codes)) > 1, "all-None params must sample, not greedy argmax"

    # -- repetition penalty (reference output_processor.py:369-397) --------- #

    def test_repetition_penalty_flips_greedy_argmax(self, model):
        """A repeated code's logit is divided by the penalty (score > 0), so
        once the argmax enters the past it stops dominating: with penalty=1.3
        and past=[7], code 7's 0.5 -> 0.5/1.3=0.385 while unpenalized code 9
        (0.45) wins. This is the mechanism that breaks a monotonically
        repeating level-0 code (the 7196x1786 image run; audio rep=1.3)."""
        logits = torch.full((self.VOCAB,), -100.0)
        logits[7] = 0.5
        logits[9] = 0.45

        assert int(self._call(model, logits, do_sample=False)) == 7
        assert int(self._call(
            model, logits, do_sample=False, repetition_penalty=1.3,
            past_codes=torch.tensor([7]),
        )) == 9

    def test_repetition_penalty_1p0_is_noop(self, model, logits):
        """penalty == 1.0 (visual default) must not change the greedy choice
        regardless of history."""
        logits = torch.full((self.VOCAB,), -100.0)
        logits[7] = 0.5
        logits[9] = 0.45
        code = int(self._call(
            model, logits, do_sample=False, repetition_penalty=1.0,
            past_codes=torch.tensor([7]),
        ))
        assert code == 7

    def test_repetition_penalty_empty_history_is_noop(self, model, logits):
        code = int(self._call(
            model, logits, do_sample=False, repetition_penalty=1.3,
            past_codes=torch.tensor([]),
        ))
        argmax = int(self._call(model, logits, do_sample=False))
        assert code == argmax

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
        self._visual_gen: dict = {}
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
