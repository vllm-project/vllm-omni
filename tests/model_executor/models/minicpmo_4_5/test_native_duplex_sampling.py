# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 duplex sampling: batched rows against a row-by-row reference, the deferred path against both."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = minicpmo_4_5_omni.MiniCPMO45OmniForConditionalGeneration
CHUNK_EOS, LISTEN, TTS_PAD, VOCAB = 5, 6, 10, 40
_NAMES = ("unit", "chunk_eos", "listen", "tts_bos", "turn_eos", "chunk_tts_eos", "tts_pad")
TOKEN_IDS = {f"{name}_token_id": token for name, token in zip(_NAMES, (1, CHUNK_EOS, LISTEN, 7, 8, 9, TTS_PAD))}


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
