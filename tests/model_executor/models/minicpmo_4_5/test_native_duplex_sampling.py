# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 duplex sampling: batched rows against a row-by-row reference, the deferred path against both."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import diagnostics
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = minicpmo_4_5_omni.MiniCPMO45OmniForConditionalGeneration
CHUNK_EOS, LISTEN, TTS_PAD, VOCAB = 5, 6, 10, 40
_NAMES = ("unit", "chunk_eos", "listen", "tts_bos", "turn_eos", "chunk_tts_eos", "tts_pad")
TOKEN_IDS = {f"{name}_token_id": token for name, token in zip(_NAMES, (1, CHUNK_EOS, LISTEN, 7, 8, 9, TTS_PAD))}


@pytest.fixture
def trace_records(monkeypatch):
    records = []
    monkeypatch.setenv("VLLM_OMNI_MINICPMO_DUPLEX_TRACE_START_UNIT", "0")
    diagnostics.trace_start_unit.cache_clear()
    # Exercise serialization too: masked logits must be JSON null, not -Infinity.
    monkeypatch.setattr(diagnostics.logger, "info", lambda _, data: records.append(json.loads(data)))
    yield records
    diagnostics.trace_start_unit.cache_clear()


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("all_greedy", [False, True])
def test_diagnostics_preserve_tokens_state_and_rng(monkeypatch, trace_records, deferred, all_greedy):
    baseline, baseline_states, baseline_md, steps = _setup(2, all_greedy=all_greedy)
    if not deferred:
        monkeypatch.setattr(baseline, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    with monkeypatch.context() as disabled:
        disabled.setattr(minicpmo_4_5_omni, "trace_start_unit", lambda: None)
        expected = _run(baseline, baseline_md, steps)
    assert trace_records == []

    model, states, md, _ = _setup(2, all_greedy=all_greedy)
    if not deferred:
        monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    assert _run(model, md, steps) == expected
    assert {row: vars(s) for row, s in states.items()} == {row: vars(s) for row, s in baseline_states.items()}
    assert all(torch.equal(md.generators[r].get_state(), baseline_md.generators[r].get_state()) for r in range(6))
    first = {record["row"]: record for record in trace_records[:6]}
    assert first[2]["reason"] == "chunk_cap"
    assert first[3]["reason"] == "chunk_boundary"
    assert first[2]["stage2_token_id"] is first[3]["stage2_token_id"] is None
    assert "stage2_logits" not in first[2] and "stage2_logits" not in first[3]
    assert first[1]["reason"] == "listen_to_tts_bos"
    assert first[1]["stage2_token_id"] == LISTEN
    assert first[1]["stage2_logits"]["control_scores"]["chunk_eos_token_id"] is None


@pytest.mark.parametrize("deferred", [False, True])
def test_diagnostics_explain_held_and_no_progress_eos(monkeypatch, trace_records, deferred):
    model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    state.session_id, state.prepared_append_identity, state.audio_chunk_idx = "trace-eos", (2, 160), 160
    model.max_speak_chars_per_chunk = 0
    md.output_token_ids[0] = []
    sample(21)
    md.output_token_ids[0] = [21]
    assert sample(TOKEN_IDS["turn_eos_token_id"], TOKEN_IDS["tts_bos_token_id"]) == TOKEN_IDS["tts_bos_token_id"]
    held = trace_records[-1]
    assert held["session_id"] == "trace-eos" and held["append_identity"] == [2, 160]
    assert held["reason"] == "hold_turn_eos"
    assert held["before"]["turn_end_hold_pending"] is False
    assert held["after"]["turn_end_hold_pending"] is True
    assert held["repetition_token_ids"] == [21]
    assert held["stage2_logits"]["top_ids"][0] == TOKEN_IDS["turn_eos_token_id"]
    md.output_token_ids[0] = []
    assert sample(TOKEN_IDS["turn_eos_token_id"]) == TOKEN_IDS["turn_eos_token_id"]
    accepted = trace_records[-1]
    assert accepted["reason"] == "accept_turn_eos"
    assert accepted["before"]["turn_end_hold_pending"] is True
    assert accepted["after"]["current_turn_ended"] is True


@pytest.mark.parametrize("all_greedy", [False, True])
def test_diagnostics_match_across_sync_and_deferred_commit(monkeypatch, trace_records, all_greedy):
    sync, _, md, steps = _setup(3, all_greedy=all_greedy)
    monkeypatch.setattr(sync, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    _run(sync, md, steps)
    expected = list(trace_records)
    trace_records.clear()
    model, _, md, _ = _setup(3, all_greedy=all_greedy)
    _run(model, md, steps)
    assert trace_records == expected


def test_diagnostics_filter_rows_and_defer_host_reads(monkeypatch, trace_records):
    model, states, md, steps = _setup(0, all_greedy=True, max_chars=0)
    monkeypatch.setenv("VLLM_OMNI_MINICPMO_DUPLEX_TRACE_START_UNIT", "140")
    diagnostics.trace_start_unit.cache_clear()
    for row, state in states.items():
        state.audio_chunk_idx, state.session_id = (140 if row == 5 else 139), f"s{row}"
    states[5].generated_tokens = [TOKEN_IDS["turn_eos_token_id"]]
    steps[0][5, TOKEN_IDS["turn_eos_token_id"]] = 3.0
    with _NoHostReads():
        model._sample_minicpmo45_native_duplex_stage0(steps[0], md, duplex_rows=list(range(6)))
    assert trace_records == []
    model._commit_minicpmo45_duplex_pending_samples()
    assert len(trace_records) == 1
    record = trace_records[0]
    assert record["row"] == 5 and record["session_id"] == "s5"
    assert record["model_logits"]["control_scores"]["turn_eos_token_id"] == 3.0
    assert record["stage2_logits"]["control_scores"]["turn_eos_token_id"] == pytest.approx(3.0 / 1.05)


def test_disabled_diagnostics_do_not_capture_logits(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO_DUPLEX_TRACE_START_UNIT", raising=False)
    diagnostics.trace_start_unit.cache_clear()

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled diagnostics must not collect tensors or format logs")

    monkeypatch.setattr(diagnostics.SamplingTrace, "capture_logits", forbidden)
    monkeypatch.setattr(diagnostics, "emit_trace", forbidden)
    model, _, md, steps = _setup(0)
    _run(model, md, steps)
    diagnostics.trace_start_unit.cache_clear()


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


def _setup(seed: int, *, all_greedy: bool = False, max_chars: int = 6, turns_open: bool = True):
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
    model._minicpmo45_duplex_payload_for_row = model._minicpmo45_duplex_row_request_max_tokens = lambda row: None
    md = SimpleNamespace(all_greedy=all_greedy, top_p=torch.tensor([0.9, 1.0, 0.8, 0.8, 0.8, 0.85]))
    md.generators = {row: torch.Generator().manual_seed(100 + row) for row in range(6)}
    md.output_token_ids = [[11, 12], [2], [1, 2, 3, 4], [2, 3], [7, 2], [20, 21, 22]]
    md.temperature, md.top_k = torch.tensor([0.8, 1.1, 0.7, 0.7, 0.0, 0.9]), torch.tensor([10, 0, 100, 100, 100, 20])
    return model, states, md, steps


def _reference(logits, md, states, row_params) -> list[int]:
    """Row by row: speak-length cap, boundary draw, then the stage-2 draw (no cut, no listen rewrite)."""
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
        if temperature <= 0:
            out.append(int(row_logits.argmax()))
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


class _SentenceTokenizer(_CharTokenizer):
    """Ids ending in 22 decode as a finished sentence; any other text is cut after 「正」."""

    def decode(self, ids, skip_special_tokens=False):
        text_ids = [token for token in map(int, ids) if token not in self.all_special_ids]
        if not text_ids:
            return ""
        if text_ids[-1] == 22:
            return "正方形中间有一个数字十一。"
        return "现在画面中出现的是一个绿色的正方形，正"


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("all_greedy", [False, True])
def test_premature_eos_uses_current_logits_without_inserting_chunk_boundary(monkeypatch, deferred, all_greedy):
    model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred, all_greedy=all_greedy)
    model.max_speak_chars_per_chunk = 0
    model._minicpmo45_tokenizer_cache = _SentenceTokenizer()
    model._record_minicpmo45_duplex_generation_token(0, 21)
    state.current_turn_ended = False
    turn_eos = TOKEN_IDS["turn_eos_token_id"]
    state.generated_tokens.extend([turn_eos, 22, 23])
    md.output_token_ids[0] = [21]
    logits = torch.full((1, VOCAB), -100.0)
    # Scores from the local reproduction at unit 160: the continuation is
    # plausible here, but is lost after injecting a new audiovisual unit.
    logits[0, turn_eos], logits[0, 22], logits[0, 23] = 27.7665, 27.2101, 26.7928
    output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
    model._commit_minicpmo45_duplex_pending_samples()
    assert output.sampled_token_ids.item() == 22
    assert state.generated_tokens[-1] == 22
    assert state.held_turn_ends == 1
    md.output_token_ids[0].append(22)
    assert sample(turn_eos) == turn_eos
    assert state.current_turn_ended is True


@pytest.mark.parametrize("deferred", [False, True])
def test_unused_continuation_candidate_does_not_advance_rng(monkeypatch, deferred):
    snapshots = []
    for unfinished in (False, True):
        model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred, all_greedy=False)
        model.max_speak_chars_per_chunk = 0
        state.current_turn_ended = False
        state.open_turn_tail = "现在画面中出现的是" if unfinished else ""
        md.output_token_ids[0] = []
        assert sample(23) == 23
        snapshots.append(md.generators[0].get_state())
    assert torch.equal(*snapshots)


@pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (3, 1.0), (3, 0.8)])
@pytest.mark.parametrize("eos_id", [1, 3])
def test_continuation_draws_follow_conditional_distribution(top_k, top_p, eos_id):
    logits = torch.tensor([[0.0, 2.0, 1.0, 1.0, 1.0, -1.0]])
    n = 20000
    rows = logits.expand(n, -1)
    uniforms = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    original = Model._duplex_top_k_top_p_candidates(rows, top_k=top_k, top_p=top_p)
    chosen = Model._duplex_draw_candidates(rows, original, uniforms) == eos_id
    retry_uniforms = Model._duplex_continuation_uniforms(rows, original, uniforms, eos_id)
    retry_logits = rows.clone()
    retry_logits[:, eos_id] = -torch.inf
    candidates = Model._duplex_top_k_top_p_candidates(retry_logits, top_k=top_k, top_p=top_p)
    draws = Model._duplex_draw_candidates(retry_logits, candidates, retry_uniforms)[chosen]
    actual = torch.bincount(draws, minlength=logits.shape[-1]).float() / draws.numel()
    expected = torch.softmax(Model._top_k_top_p_filter(retry_logits[0], top_k=top_k, top_p=top_p), dim=-1)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0)


@pytest.mark.parametrize("deferred", [False, True])
def test_turn_eos_before_sentence_end_selects_same_position_continuation(monkeypatch, deferred):
    """Issue 9828: <|turn_eos|> after 「正」 must not end the turn.

    Select a continuation before injecting an artificial chunk boundary.
    """
    model, state, md, sample = _character_cap_case(monkeypatch, deferred=deferred, all_greedy=True)
    model.max_speak_chars_per_chunk = 0
    state.current_turn_ended = False
    turn_eos = TOKEN_IDS["turn_eos_token_id"]
    model._minicpmo45_tokenizer_cache = _SentenceTokenizer()
    model._record_minicpmo45_duplex_generation_token(0, 21)
    assert state.open_turn_tail.endswith("正")
    md.output_token_ids[0] = [21]

    assert sample(turn_eos, 23) == 23
    assert state.current_turn_ended is False
    assert state.held_turn_ends == 1
    assert state.generated_tokens == [21, 23]
    assert state.open_turn_tail.endswith("正")

    # A normal model-selected chunk boundary can still precede more text.
    md.output_token_ids[0].append(23)
    assert sample(CHUNK_EOS) == CHUNK_EOS
    md.output_token_ids[0] = []
    assert sample(23) == 23
    md.output_token_ids[0] = [23]
    assert sample(turn_eos, 23) == 23
    assert state.current_turn_ended is False
    assert state.held_turn_ends == 2

    model._record_minicpmo45_duplex_generation_token(0, 22)
    md.output_token_ids[0] = [CHUNK_EOS, 22]
    assert sample(turn_eos) == turn_eos
    assert state.current_turn_ended is True
    assert state.held_turn_ends == 0
    assert state.open_turn_tail == ""


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("control_token", [None, TOKEN_IDS["tts_bos_token_id"], 30])
def test_turn_eos_without_text_progress_ends_the_turn(monkeypatch, deferred, control_token):
    """An unpunctuated answer gets one continuation attempt, not 16 empty chunks."""
    model, _, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    # Like <|spk_bos|>: a tokenizer special ID outside the duplex field map,
    # permitted by the sampler but omitted from decoded text.
    model._minicpmo45_tokenizer_cache.all_special_ids = [*TOKEN_IDS.values(), 30]
    state = _MiniCPMO45Stage0SessionState(session_id="no-progress", current_turn_ended=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model.max_speak_chars_per_chunk = 0
    turn_eos = TOKEN_IDS["turn_eos_token_id"]
    md.output_token_ids[0] = []
    assert sample(21) == 21  # No sentence-ending punctuation.
    md.output_token_ids[0] = [21]
    assert sample(turn_eos, TOKEN_IDS["tts_bos_token_id"]) == TOKEN_IDS["tts_bos_token_id"]

    md.output_token_ids[0] = []
    if control_token is not None:
        assert sample(control_token) == control_token
        md.output_token_ids[0] = [control_token]
    assert sample(turn_eos) == turn_eos
    assert state.current_turn_ended is True
    assert state.held_turn_ends == 0
    assert state.open_turn_tail == ""

    # A later answer gets its own continuation attempt.
    md.output_token_ids[0] = []
    assert sample(21) == 21
    md.output_token_ids[0] = [21]
    assert sample(turn_eos, TOKEN_IDS["tts_bos_token_id"]) == TOKEN_IDS["tts_bos_token_id"]
    assert state.current_turn_ended is False
    assert state.held_turn_ends == 1


@pytest.mark.parametrize("deferred", [False, True])
def test_turn_end_hold_cap_survives_text_progress(monkeypatch, deferred):
    """Progress permits retries but cannot reset the total per-turn hold limit."""
    model, _, md, sample = _character_cap_case(monkeypatch, deferred=deferred)
    state = _MiniCPMO45Stage0SessionState(session_id="hold-cap", current_turn_ended=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model.max_speak_chars_per_chunk = 0
    turn_eos = TOKEN_IDS["turn_eos_token_id"]
    for _ in range(MiniCPMO45DuplexPolicy.MAX_HELD_UNFINISHED_TURN_ENDS):
        md.output_token_ids[0] = []
        assert sample(21) == 21
        md.output_token_ids[0] = [21]
        assert sample(turn_eos, 23) == 23
        assert state.current_turn_ended is False
    md.output_token_ids[0] = []
    assert sample(21) == 21
    md.output_token_ids[0] = [21]
    assert sample(turn_eos) == turn_eos
    assert state.current_turn_ended is True


def test_deferred_sentence_ender_is_committed_before_the_next_turn_eos(monkeypatch):
    """The next stage-0 step commits the pending text token before it holds.

    ``。`` and ``<|turn_eos|>`` are consecutive deferred samples, with no
    explicit commit between them. The turn end is accepted.
    """
    model, state, md, _sample = _character_cap_case(monkeypatch, deferred=True, all_greedy=True)
    model.max_speak_chars_per_chunk = 0
    state.current_turn_ended = False
    model._minicpmo45_tokenizer_cache = _SentenceTokenizer()
    turn_eos = TOKEN_IDS["turn_eos_token_id"]

    def draw(token):
        logits = torch.full((1, VOCAB), -100.0)
        logits[0, token] = 100.0
        output = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=[0])
        return output.sampled_token_ids.item()

    md.output_token_ids[0] = []
    assert draw(21) == 21
    model._commit_minicpmo45_duplex_pending_samples()
    assert state.open_turn_tail.endswith("正")

    md.output_token_ids[0] = [21]
    assert draw(22) == 22
    assert state.open_turn_tail.endswith("正")  # 22 is still pending

    md.output_token_ids[0] = [21, 22]
    assert draw(turn_eos) == turn_eos
    model._commit_minicpmo45_duplex_pending_samples()
    assert state.open_turn_tail == ""
    assert state.current_turn_ended is True
