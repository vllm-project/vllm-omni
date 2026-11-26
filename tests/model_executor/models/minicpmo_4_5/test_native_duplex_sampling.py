# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 duplex sampling: batched rows against a row-by-row reference, the deferred path against both."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = minicpmo_4_5_omni.MiniCPMO45OmniForConditionalGeneration
CHUNK_EOS, LISTEN, TTS_PAD, VOCAB = 5, 6, 10, 40
_NAMES = ("unit", "chunk_eos", "listen", "tts_bos", "turn_eos", "chunk_tts_eos", "tts_pad")
TOKEN_IDS = {f"{name}_token_id": token for name, token in zip(_NAMES, (1, CHUNK_EOS, LISTEN, 7, 8, 9, TTS_PAD))}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_chunk_boundary_probability_preserves_softmax_shift_invariance(dtype):
    # A shared logit offset must not change the decision to stop a speech chunk.
    # Materializing logsumexp in BF16 loses precision before subtracting it.
    logits = torch.zeros(5, 3, dtype=dtype)
    logits += torch.tensor([0, 16, 32, 64, 128], dtype=dtype)[:, None]
    probabilities = Model._duplex_boundary_chunk_eos_probs(logits, 1)
    expected = torch.full((5,), 1 / 3, dtype=dtype)
    torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
    # A draw between the correct probability and the rounded logsumexp result
    # must keep generating for every row, irrespective of the common offset.
    assert not (torch.full((5,), 0.35) < probabilities).any()


class _CharTokenizer:
    """Token t decodes to (t % 3) + 1 copies of one CJK character; special ids decode to ""."""

    clean_up_tokenization_spaces, bad_token_ids, all_special_ids = False, list[int](), sorted(TOKEN_IDS.values())

    def __len__(self):
        return VOCAB

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if t in self.all_special_ids else chr(0x4E00 + t) * (t % 3 + 1) for t in map(int, ids))

    def batch_decode(self, batch, skip_special_tokens=False):
        return [self.decode(ids) for ids in batch]


def _setup(
    seed: int,
    *,
    all_greedy: bool = False,
    max_chars: int = 6,
    turns_open: bool = True,
    payloads: dict[int, dict[str, object]] | None = None,
):
    """Six rows: sampled, listen in an open turn, speak-length cap, boundary chunk_eos, greedy, penalized."""
    torch.manual_seed(seed)
    steps = [torch.randn(6, VOCAB) * 2 for _ in range(2)]
    for logits in steps:
        logits[3, CHUNK_EOS] += 50.0
        logits[1, LISTEN] += 40.0
    states = {row: SimpleNamespace(generated_tokens=[], current_turn_ended=True) for row in range(6)}
    states[1].current_turn_ended, states[5].generated_tokens = not turns_open, [11, 11, 12]
    model = Model.__new__(Model)
    model.model_stage, model.max_new_speak_tokens_per_chunk, model.max_speak_chars_per_chunk = "llm", 5, max_chars
    model._minicpmo45_native_duplex_token_ids_cache = dict(TOKEN_IDS)
    model._minicpmo45_tokenizer_cache, model._minicpmo45_duplex_state_for_row = _CharTokenizer(), states.get
    model._minicpmo45_duplex_payload_for_row = (lambda row: None) if payloads is None else payloads.get
    model._minicpmo45_duplex_row_request_max_tokens = lambda row: None
    md = SimpleNamespace(all_greedy=all_greedy, top_p=torch.tensor([0.9, 1.0, 0.8, 0.8, 0.8, 0.85]))
    md.generators = {row: torch.Generator().manual_seed(100 + row) for row in range(6)}
    md.output_token_ids = [[11, 12], [2], [1, 2, 3, 4], [2, 3], [7, 2], [20, 21, 22]]
    md.temperature, md.top_k = torch.tensor([0.8, 1.1, 0.7, 0.7, 0.0, 0.9]), torch.tensor([10, 0, 100, 100, 100, 20])
    return model, states, md, steps


def _reference(logits, md, states, row_params, payloads=None) -> list[int]:
    """Row by row: speak-length cap, boundary draw, then the stage-2 draw (no cut, no listen rewrite).

    Rows with a ``listen_prob_scale`` / ``listen_top_k`` payload get the released
    ``StreamDecoder.decode`` listen treatment: the listen logit scales after the
    repetition penalty, and a listen ranking within ``listen_top_k`` is returned
    outright. The batched sampler draws every stage-2 row and overrides the
    early return, so this oracle consumes the same uniform either way.
    """
    out = []
    for row, (temperature, top_k, top_p) in enumerate(row_params):
        generator, recent = md.generators[row], md.output_token_ids[row]
        if len(recent) >= 4 or torch.rand((), generator=generator) < torch.softmax(logits[row], -1)[CHUNK_EOS]:
            out.append(CHUNK_EOS)
            continue
        row_logits = logits[row : row + 1].clone()
        row_logits[0, [TTS_PAD, CHUNK_EOS]] = float("-inf")
        for token in set((states[row].generated_tokens or recent)[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :]):
            row_logits[0, token] /= 1.05
        payload = payloads.get(row) if payloads is not None else None
        listen_scale = float(payload.get("listen_prob_scale", 1.0)) if isinstance(payload, dict) else 1.0
        listen_top_k = payload.get("listen_top_k") if isinstance(payload, dict) else None
        if listen_scale != 1.0:
            row_logits[0, LISTEN] *= listen_scale
        early = listen_top_k is not None and int((row_logits[0] > row_logits[0, LISTEN]).sum()) < listen_top_k
        if temperature <= 0:
            out.append(LISTEN if early else int(row_logits.argmax()))
            continue
        if early:
            # The released decoder returns before the final draw, so an early
            # listen consumes no stage-2 random number.
            out.append(LISTEN)
            continue
        scaled = row_logits / temperature
        candidates = Model._duplex_top_k_top_p_candidates(scaled, top_k=top_k, top_p=top_p)
        # The candidate-space distribution is the dense top-k / top-p filter's.
        dense = torch.softmax(Model._top_k_top_p_filter(scaled, top_k=int(top_k), top_p=top_p), -1)
        torch.testing.assert_close(_expand(scaled, candidates), dense)
        uniform = torch.rand((), generator=generator)
        out.append(int(Model._duplex_draw_candidates(scaled, candidates, uniform.view(1))))
    return out


def _expand(logits, candidates) -> torch.Tensor:
    """The candidates as a dense distribution: the tie slot spread over its lowest-index tied tokens."""
    probs, ids, kth, ties = candidates
    dense = torch.zeros(logits.shape, dtype=probs.dtype).scatter_(1, ids[:, :-1], probs[:, :-1])
    tied = logits == kth
    kept = tied & (tied.cumsum(dim=-1) <= ties)
    return dense + kept * (probs[:, -1:] / ties.clamp(min=1))


def _run(model, md, steps) -> tuple[list[list[int]], list[bool]]:
    """Two steps through the Stage-0 sampler; also whether each left a deferred commit."""
    tokens, deferred = [], []
    for logits in steps:
        out = model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
        tokens.append(out.sampled_token_ids.squeeze(-1).tolist())
        deferred.append(getattr(model, "_minicpmo45_duplex_pending_samples", None) is not None)
        model._commit_minicpmo45_duplex_pending_samples()
        for row, token in enumerate(tokens[-1]):
            md.output_token_ids[row].append(token)
    return tokens, deferred


@pytest.mark.parametrize("seed", range(4))
def test_batched_rows_match_the_row_by_row_reference(seed):
    model, states, md, steps = _setup(seed, max_chars=10**6, turns_open=False)
    _, ref_states, ref_md, _ = _setup(seed)
    params = model._minicpmo45_duplex_row_params(md, 6)
    kwargs = dict(row_idxs=list(range(6)), token_ids=TOKEN_IDS, row_params=params)
    rows = model._sample_minicpmo45_native_duplex_rows(steps[0], md, **kwargs)
    assert rows == _reference(steps[0], ref_md, ref_states, params)
    assert all(torch.equal(md.generators[r].get_state(), ref_md.generators[r].get_state()) for r in range(6))


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_deferred_rows_match_the_synchronous_rows(seed, all_greedy, monkeypatch):
    sync_model, sync_states, sync_md, steps = _setup(seed, all_greedy=all_greedy)
    monkeypatch.setattr(sync_model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    expected, _ = _run(sync_model, sync_md, steps)
    model, states, md, _ = _setup(seed, all_greedy=all_greedy)
    assert _run(model, md, steps) == (expected, [True, True])
    assert {row: vars(s) for row, s in states.items()} == {row: vars(s) for row, s in sync_states.items()}
    assert all(torch.equal(md.generators[r].get_state(), sync_md.generators[r].get_state()) for r in range(6))


def _tied_logits(device: str) -> torch.Tensor:
    """Rows whose k-th logit is tied well past k, with the nucleus cutting inside and before the tie."""
    logits = torch.full((4, 64), -3.0, device=device)
    logits[0, [3, 9, 17, 30, 41, 50]] = 1.0  # six tied for k=2
    logits[1, 7], logits[1, [2, 12, 22, 32, 42]] = 4.0, 1.0  # a leader, then five tied for k=3
    logits[2] = torch.randn(64, generator=torch.Generator().manual_seed(0)).to(device)
    logits[3, ::4] = 0.5  # sixteen tied for k=4
    return logits


@pytest.mark.parametrize("top_p", [1.0, 0.9, 0.5, 0.2])
@pytest.mark.parametrize("top_k", [2, 3, 4, 0])
def test_candidate_draws_follow_the_dense_distribution(top_k, top_p):
    # A midpoint grid of uniforms integrates the inverse CDF exactly: each token
    # is drawn in proportion to its dense probability, ties included.
    logits, n = _tied_logits("cpu"), 4096
    expected = torch.softmax(Model._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p), dim=-1)
    rows = logits.repeat(n, 1)
    grid = ((torch.arange(n, dtype=torch.float32) + 0.5) / n).repeat_interleave(logits.shape[0])
    draws = Model._duplex_draw_candidates(
        rows, Model._duplex_top_k_top_p_candidates(rows, top_k=top_k, top_p=top_p), grid
    )
    counts = torch.zeros_like(expected).index_put_(
        (torch.arange(rows.shape[0]) % logits.shape[0], draws), torch.ones(rows.shape[0]), accumulate=True
    )
    torch.testing.assert_close(counts / n, expected, atol=2.0 / n, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_p", [0.2, 0.25, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("n", [4, 10, 20, 64, 100])
def test_tied_nucleus_boundaries_match_the_dense_filter(n, top_p, dtype):
    # Uniform ties put every top_p on (or next to) a candidate boundary.
    logits = torch.full((3, n + 8), -30.0)
    logits[0, :n], logits[1, 3 : 3 + n], logits[2, :2], logits[2, 4 : 4 + n] = 20.0, 0.5, 2.0, 1.0
    logits = logits.to(dtype)
    # The candidates compute in float32; bf16 widens exactly, so ties survive.
    expected = torch.softmax(Model._top_k_top_p_filter(logits.float(), top_k=3, top_p=top_p), dim=-1)
    actual = _expand(logits, Model._duplex_top_k_top_p_candidates(logits, top_k=3, top_p=top_p))
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected)


class _NoHostReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.is_nonzero.default):
            raise AssertionError(f"host read: {func}")
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_candidate_sampling_reads_nothing_back_to_the_host(device):
    logits = _tied_logits(device)
    with _NoHostReads():
        for top_k, top_p in [(2, 0.5), (4, 1.0), (0, 0.8)]:
            candidates = Model._duplex_top_k_top_p_candidates(logits, top_k=top_k, top_p=top_p)
            Model._duplex_draw_candidates(logits, candidates, torch.rand(logits.shape[0], device=device))


def _character_cap_case(monkeypatch, *, deferred, all_greedy=True):
    model, states, md, _ = _setup(0, all_greedy=all_greedy, max_chars=28, turns_open=False)
    model.max_new_speak_tokens_per_chunk = 64
    md.top_k[0], md.top_p[0] = 1, 1.0
    if not deferred:
        monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)

    def sample(token, alternative=None):
        logits = torch.full((1, VOCAB), -100.0)
        logits[0, token] = 100.0
        if alternative is not None:
            logits[0, alternative] = 99.0
        output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
        model._commit_minicpmo45_duplex_pending_samples()
        return output.sampled_token_ids.item()

    return model, states[0], md, sample


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("prefix", [[20] * 9, [20] * 8 + [19]])
def test_character_target_keeps_crossing_token_then_caps_without_sampling(monkeypatch, deferred, all_greedy, prefix):
    model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred, all_greedy=all_greedy)
    # Token 20 has 3 characters, token 19 has 2: 27 + 2 crosses; 26 + 2 reaches 28.
    md.output_token_ids[0] = prefix.copy()
    assert sample(19) == 19
    assert state.generated_tokens == [19]
    md.output_token_ids[0].append(19)
    before = md.generators[0].get_state().clone()

    def must_not_sample(*args, **kwargs):
        pytest.fail("An exhausted character budget must end before any sampler draw")

    monkeypatch.setattr(model, "_minicpmo45_duplex_boundary_hits", must_not_sample)
    monkeypatch.setattr(model, "_minicpmo45_duplex_stage2_draws", must_not_sample)
    assert sample(19) == CHUNK_EOS
    assert state.generated_tokens == [19]
    assert torch.equal(md.generators[0].get_state(), before)


@pytest.mark.parametrize("deferred", [False, True])
def test_character_target_waits_for_complete_utf8_prefix(monkeypatch, deferred):
    model, _, md, sample = _character_cap_case(monkeypatch, deferred=deferred)

    class PartialTokenizer(_CharTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return "甲" * 27 + "\ufffd" if list(ids) == [20] else "甲" * 28

    model._minicpmo45_tokenizer_cache = PartialTokenizer()
    md.output_token_ids[0] = [20]
    assert sample(19) == 19  # Complete the bytes rather than drop their token.
    md.output_token_ids[0].append(19)
    assert sample(19) == CHUNK_EOS


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_character_budget_precedes_future_special_draw_only_when_exhausted(monkeypatch, deferred, exhausted):
    _, _, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    md.output_token_ids[0] = [20] * (10 if exhausted else 9)
    assert sample(TOKEN_IDS["turn_eos_token_id"]) == (CHUNK_EOS if exhausted else TOKEN_IDS["turn_eos_token_id"])


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("hard_cap", [False, True])
def test_explicit_force_listen_bypasses_only_soft_character_target(monkeypatch, deferred, hard_cap):
    model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    model._minicpmo45_duplex_payload_for_row = lambda row: {"force_listen": True}
    model.max_new_speak_tokens_per_chunk = 5 if hard_cap else 64
    md.output_token_ids[0] = [20] * 10
    assert sample(LISTEN) == (CHUNK_EOS if hard_cap else LISTEN)
    assert state.generated_tokens == []


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("all_greedy", [False, True])
def test_sampled_eos_is_preserved_without_punctuation(monkeypatch, deferred, all_greedy):
    """Neither an unpunctuated answer nor an unfinished word authorizes resampling EOS."""
    model, state, md, _ = _character_cap_case(monkeypatch, deferred=deferred, all_greedy=all_greedy)
    model.max_speak_chars_per_chunk = 0
    state.current_turn_ended = False
    prefix, candidate = [27], 26
    for token in prefix:
        model._record_minicpmo45_duplex_generation_token(0, token)
    md.output_token_ids[0] = list(prefix)
    eos = TOKEN_IDS["turn_eos_token_id"]
    logits = torch.full((1, VOCAB), -100.0)
    logits[0, eos], logits[0, candidate] = 27.5, 25.0
    if deferred:
        with _NoHostReads():
            output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
    else:
        output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
    model._commit_minicpmo45_duplex_pending_samples()
    assert output.sampled_token_ids.item() == eos
    assert state.generated_tokens == [*prefix, eos]
    assert state.current_turn_ended


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("previous,candidate", [(23, 23), (27, 26)], ids=["repetition", "continuation"])
def test_model_selected_repetition_and_continuation_are_preserved(monkeypatch, deferred, previous, candidate):
    model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    model.max_speak_chars_per_chunk = 0
    state.current_turn_ended = False
    model._record_minicpmo45_duplex_generation_token(0, previous)
    md.output_token_ids[0] = []
    assert sample(candidate) == candidate
    assert state.generated_tokens == [previous, candidate]
    assert not state.current_turn_ended


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("all_greedy", [False, True])
def test_only_finite_eos_is_preserved(monkeypatch, deferred, all_greedy):
    model, state, md, _ = _character_cap_case(monkeypatch, deferred=deferred, all_greedy=all_greedy)
    model.max_speak_chars_per_chunk = 0
    state.current_turn_ended = False
    model._record_minicpmo45_duplex_generation_token(0, 23)
    md.output_token_ids[0] = []
    eos = TOKEN_IDS["turn_eos_token_id"]
    logits = torch.full((1, VOCAB), -torch.inf)
    logits[0, eos] = 1.0
    output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
    model._commit_minicpmo45_duplex_pending_samples()
    assert output.sampled_token_ids.item() == eos
    assert state.generated_tokens == [23, eos]
    assert state.current_turn_ended


def test_deferred_text_is_committed_before_the_next_turn_eos(monkeypatch):
    """Consecutive deferred samples retain text then EOS without an explicit commit in between."""
    model, state, md, _ = _character_cap_case(monkeypatch, deferred=True, all_greedy=True)
    model.max_speak_chars_per_chunk = 0
    state.current_turn_ended = False
    eos = TOKEN_IDS["turn_eos_token_id"]

    def draw(token):
        logits = torch.full((1, VOCAB), -100.0)
        logits[0, token] = 100.0
        output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
        return output.sampled_token_ids.item()

    md.output_token_ids[0] = []
    assert draw(21) == 21
    assert state.generated_tokens == []
    md.output_token_ids[0] = [21]
    assert draw(eos) == eos
    assert state.generated_tokens == [21]
    model._commit_minicpmo45_duplex_pending_samples()
    assert state.generated_tokens == [21, eos]
    assert state.current_turn_ended is True


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("window_enabled", [False, True])
def test_discarded_async_text_cannot_suppress_next_unit_eos(monkeypatch, deferred, window_enabled):
    from tests.model_executor.models.minicpmo_4_5.helpers import stage0_vision_runtime

    runtime = stage0_vision_runtime()
    for name, value in TOKEN_IDS.items():
        setattr(runtime, name, value)
    state = _MiniCPMO45Stage0SessionState(session_id="accepted-span", window_enabled=window_enabled)
    first = runtime._stage_prefill_embeddings_only(state, None, video_frames=[object()], seq=1)
    assert first["success"]
    model, _, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model.max_speak_chars_per_chunk = 0
    md.output_token_ids[0] = []
    assert sample(23) == 23  # Accepted text.
    md.output_token_ids[0] = [23]
    assert sample(CHUNK_EOS) == CHUNK_EOS
    assert sample(23) == 23  # Late lookahead text; scheduler stopped at the boundary.
    second = runtime._stage_prefill_embeddings_only(
        state,
        None,
        video_frames=[object()],
        seq=2,
        stage0_window={"completed_token_ids": [23], "completed_terminator_token_id": CHUNK_EOS},
    )
    assert second["success"]
    assert second["input_token_ids"][:3] == [CHUNK_EOS, runtime.unit_end_token_id, runtime.unit_token_id]
    md.output_token_ids[0] = []
    eos = TOKEN_IDS["turn_eos_token_id"]
    assert sample(eos, 29) == eos
    assert state.current_turn_ended


def _policy_model_with_state(state):
    """A bare LLM-stage model whose session ``s0`` resolves to ``state``."""
    model = Model.__new__(Model)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = dict(TOKEN_IDS)
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={"s0": state})
    return model


def _policy_row(payload):
    return SimpleNamespace(
        row_idx=0,
        request_id="r0",
        seq=0,
        session_id="s0",
        payload=payload,
        temperature=None,
        top_k=None,
        top_p=None,
        max_tokens=None,
    )


def test_response_create_unit_releases_turn_ended_latch():
    """A response.create unit opens the assistant turn despite the latch.

    A truncated turn leaves ``current_turn_ended`` set; without the
    ``open_assistant_turn`` marker the forced response's silent rows are
    masked to listen-only and the listen -> tts_bos swap stays blocked, so
    the forced response can only listen.
    """
    state = _MiniCPMO45Stage0SessionState(session_id="s0")
    state.current_turn_ended = True
    state.pending_speech_context = False
    model = _policy_model_with_state(state)
    logits = torch.zeros(1, VOCAB)
    model.prepare_duplex_sampling(logits, None, rows=(_policy_row({"open_assistant_turn": True}),))
    # The latch is released and the response is open: the silent row is not
    # masked to listen-only and the listen -> tts_bos swap can fire.
    assert state.current_turn_ended is False
    assert state.pending_speech_response_open is True
    assert bool(torch.isfinite(logits[0]).all())
    # Opening a response permits generation; it must also permit the model to listen.
    assert model._finalize_minicpmo45_native_duplex_sample(0, LISTEN, TOKEN_IDS) == LISTEN
    model._record_minicpmo45_duplex_terminator(0, LISTEN, TOKEN_IDS)
    assert state.current_turn_ended is True
    assert state.pending_speech_response_open is False


def test_latched_silent_row_still_masked_without_marker():
    """Without the marker a latched silent row keeps the listen-only mask."""
    state = _MiniCPMO45Stage0SessionState(session_id="s0")
    state.current_turn_ended = True
    state.pending_speech_context = False
    model = _policy_model_with_state(state)
    logits = torch.zeros(1, VOCAB)
    model.prepare_duplex_sampling(logits, None, rows=(_policy_row({}),))
    assert state.current_turn_ended is True
    masked = torch.isneginf(logits[0])
    assert int(masked.sum()) == VOCAB - 1
    assert logits[0][LISTEN] == 0.0


def test_marker_does_not_touch_already_open_turn():
    """An unlatched turn keeps its state: the marker only releases the latch."""
    state = _MiniCPMO45Stage0SessionState(session_id="s0")
    state.current_turn_ended = False
    state.pending_speech_response_open = False
    model = _policy_model_with_state(state)
    logits = torch.zeros(1, VOCAB)
    model.prepare_duplex_sampling(logits, None, rows=(_policy_row({"open_assistant_turn": True}),))
    assert state.current_turn_ended is False
    assert state.pending_speech_response_open is False
    assert model._finalize_minicpmo45_native_duplex_sample(0, LISTEN, TOKEN_IDS) == TOKEN_IDS["tts_bos_token_id"]


@pytest.mark.parametrize("initially_ended", [True, False])
def test_response_create_marker_is_consumed_once_per_unit(initially_ended):
    state = _MiniCPMO45Stage0SessionState(session_id="s0", current_turn_ended=initially_ended)
    model = _policy_model_with_state(state)
    row = _policy_row({"open_assistant_turn": True})
    model.prepare_duplex_sampling(torch.zeros(1, VOCAB), None, rows=(row,))
    model._record_minicpmo45_duplex_terminator(0, TOKEN_IDS["turn_eos_token_id"], TOKEN_IDS)
    logits = torch.zeros(1, VOCAB)
    model.prepare_duplex_sampling(logits, None, rows=(row,))
    assert state.current_turn_ended is True
    assert torch.isneginf(logits[0, :LISTEN]).all()
    assert model._finalize_minicpmo45_native_duplex_sample(0, LISTEN, TOKEN_IDS) == LISTEN
    row.seq = 1
    model.prepare_duplex_sampling(torch.zeros(1, VOCAB), None, rows=(row,))
    assert state.current_turn_ended is False


# ---- listen_prob_scale / listen_top_k (the released StreamDecoder knobs) ----


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0, 4.0])
@pytest.mark.parametrize("seed", range(4))
def test_listen_prob_scale_matches_the_row_by_row_reference(seed, scale):
    # Rows 0/2/5 carry the knob; rows 1/3/4 sample untouched beside them.
    payloads = {row: {"listen_prob_scale": scale} for row in (0, 2, 5)}
    model, _, md, steps = _setup(seed, max_chars=10**6, turns_open=False, payloads=payloads)
    _, ref_states, ref_md, _ = _setup(seed)
    params = model._minicpmo45_duplex_row_params(md, 6)
    kwargs = dict(row_idxs=list(range(6)), token_ids=TOKEN_IDS, row_params=params)
    rows = model._sample_minicpmo45_native_duplex_rows(steps[0], md, **kwargs)
    assert rows == _reference(steps[0], ref_md, ref_states, params, payloads=payloads)
    assert all(torch.equal(md.generators[r].get_state(), ref_md.generators[r].get_state()) for r in range(6))


@pytest.mark.parametrize("top_k", [1, 2, 3, 6])
@pytest.mark.parametrize("seed", range(4))
def test_listen_top_k_early_return_matches_the_row_by_row_reference(seed, top_k):
    payloads = {row: {"listen_top_k": top_k} for row in range(6)}
    model, _, md, steps = _setup(seed, max_chars=10**6, turns_open=False, payloads=payloads)
    _, ref_states, ref_md, _ = _setup(seed)
    params = model._minicpmo45_duplex_row_params(md, 6)
    kwargs = dict(row_idxs=list(range(6)), token_ids=TOKEN_IDS, row_params=params)
    rows = model._sample_minicpmo45_native_duplex_rows(steps[0], md, **kwargs)
    assert rows == _reference(steps[0], ref_md, ref_states, params, payloads=payloads)
    assert all(torch.equal(md.generators[r].get_state(), ref_md.generators[r].get_state()) for r in range(6))


@pytest.mark.parametrize("seed", range(4))
def test_neutral_listen_knobs_leave_the_draw_bit_identical(seed):
    base_model, _, base_md, steps = _setup(seed, max_chars=10**6, turns_open=False)
    params = base_model._minicpmo45_duplex_row_params(base_md, 6)
    kwargs = dict(row_idxs=list(range(6)), token_ids=TOKEN_IDS, row_params=params)
    expected = base_model._sample_minicpmo45_native_duplex_rows(steps[0], base_md, **kwargs)
    for neutral in ({row: {} for row in range(6)}, {row: {"listen_prob_scale": 1.0} for row in range(6)}):
        model, _, md, _ = _setup(seed, max_chars=10**6, turns_open=False, payloads=neutral)
        fresh_params = model._minicpmo45_duplex_row_params(md, 6)
        assert model._sample_minicpmo45_native_duplex_rows(steps[0], md, **kwargs) == expected
        assert (
            all(torch.equal(md.generators[r].get_state(), base_md.generators[r].get_state()) for r in range(6))
            and fresh_params == params
        )


def test_listen_top_k_early_return_stays_out_of_the_repetition_history():
    payloads = {0: {"listen_top_k": 2}, 1: {"listen_top_k": 2}}
    model, states, md, _ = _setup(3, all_greedy=True, max_chars=10**6, turns_open=False, payloads=payloads)
    logits = torch.full((6, VOCAB), -10.0)
    logits[0, LISTEN], logits[0, 17] = 5.0, 4.0  # listen leads: rank 0, within 2 -> early
    logits[1, 18], logits[1, 19], logits[1, LISTEN] = 9.0, 8.0, -5.0  # rank 2, outside 2
    logits[4, 21] = 7.0  # no knob on this row
    params = model._minicpmo45_duplex_row_params(md, 6)
    rows = model._sample_minicpmo45_native_duplex_rows(
        logits, md, row_idxs=[0, 1, 4], token_ids=TOKEN_IDS, row_params=params
    )
    assert rows[0] == LISTEN
    # The released decoder's early return never reaches its generated_tokens.
    assert LISTEN not in states[0].generated_tokens
    assert rows[1] == 18 and 18 in states[1].generated_tokens
    assert rows[2] == 21 and 21 in states[4].generated_tokens


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_deferred_listen_knobs_match_the_synchronous_rows(seed, all_greedy, monkeypatch):
    payloads = {
        0: {"listen_prob_scale": 0.5},
        1: {"listen_top_k": 2},
        3: {"listen_prob_scale": 3.0, "listen_top_k": 4},
    }
    sync_model, sync_states, sync_md, steps = _setup(seed, all_greedy=all_greedy, payloads=payloads)
    monkeypatch.setattr(sync_model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    expected, _ = _run(sync_model, sync_md, steps)
    model, states, md, _ = _setup(seed, all_greedy=all_greedy, payloads=payloads)
    assert _run(model, md, steps) == (expected, [True, True])
    assert {row: vars(s) for row, s in states.items()} == {row: vars(s) for row, s in sync_states.items()}
    assert all(torch.equal(md.generators[r].get_state(), sync_md.generators[r].get_state()) for r in range(6))


def test_listen_top_k_beyond_the_sampler_range_is_clamped_not_raised():
    """A session may pass any positive integer; ranks never exceed the vocabulary.

    The stage-2 sampler stores the bound in a torch.long tensor, so an
    int64-overflowing value must be clamped there rather than raise during
    decoding. Every bound above vocab + 1 is behavior-identical to vocab + 1,
    because a rank counts strictly greater logits over the whole vocabulary.
    """
    huge = 2**70
    payloads = {0: {"listen_top_k": huge}, 1: {"listen_top_k": VOCAB + 1}}
    model, _, md, steps = _setup(7, all_greedy=True, max_chars=10**6, turns_open=False, payloads=payloads)
    params = model._minicpmo45_duplex_row_params(md, 6)
    kwargs = dict(row_idxs=[0, 1], token_ids=TOKEN_IDS, row_params=params)
    rows = model._sample_minicpmo45_native_duplex_rows(steps[0], md, **kwargs)
    # Both bounds cover every possible rank, so both rows early-return listen.
    assert rows[0] == LISTEN
    assert rows[1] == LISTEN


@pytest.mark.parametrize("seed", range(4))
def test_early_listen_leaves_the_reference_rng_for_the_next_stochastic_step(seed):
    """An early listen must not advance the row's generator past the reference.

    The released StreamDecoder returns when the listen rank passes, before the
    stage-2 draw, so the next stochastic step samples from the generator state
    the reference leaves. Step 0 early-listens on the boosted row; step 1 draws
    stochastically from wherever that state is. The deferred path is covered by
    ``test_deferred_listen_knobs_match_the_synchronous_rows``, which checks
    deferred-vs-sync generator-state equality with the same ``listen_top_k``
    payloads on a two-step run.
    """
    payloads = {row: {"listen_top_k": 3} for row in range(6)}
    model, model_states, md, steps = _setup(seed, max_chars=10**6, turns_open=False, payloads=payloads)
    _, ref_states, ref_md, _ = _setup(seed)
    params = model._minicpmo45_duplex_row_params(md, 6)
    kwargs = dict(row_idxs=list(range(6)), token_ids=TOKEN_IDS, row_params=params)

    # Step 0: outputs and generator states match the reference.
    rows0 = model._sample_minicpmo45_native_duplex_rows(steps[0], md, **kwargs)
    assert rows0 == _reference(steps[0], ref_md, ref_states, params, payloads=payloads)
    assert rows0[1] == LISTEN  # the boosted row early-listens
    assert all(torch.equal(md.generators[r].get_state(), ref_md.generators[r].get_state()) for r in range(6))

    # Step 1: both sides run the same fresh logits. Sync the repetition
    # history so the early classification agrees on both sides.
    second = torch.randn(6, VOCAB) * 2
    for row, token in enumerate(rows0):
        md.output_token_ids[row].append(token)
        ref_md.output_token_ids[row].append(token)
    for row in range(6):
        ref_states[row].generated_tokens = list(model_states[row].generated_tokens)
    model._sample_minicpmo45_native_duplex_rows(second, md, **kwargs)
    _reference(second, ref_md, ref_states, params, payloads=payloads)
    # The generator states still match after the stochastic step.
    assert all(torch.equal(md.generators[r].get_state(), ref_md.generators[r].get_state()) for r in range(6))
