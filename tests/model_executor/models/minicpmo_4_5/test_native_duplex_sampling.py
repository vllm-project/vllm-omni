# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 duplex sampling: batched rows against a row-by-row reference, the deferred path against both."""

from types import SimpleNamespace

import pytest
import torch

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
        probs, ids = Model._duplex_top_k_top_p_candidates(row_logits / temperature, top_k=top_k, top_p=top_p)
        # The candidate-space distribution is the dense top-k / top-p filter's.
        dense = torch.softmax(Model._top_k_top_p_filter(row_logits / temperature, top_k=int(top_k), top_p=top_p), -1)
        torch.testing.assert_close(torch.zeros_like(dense).scatter_(1, ids, probs), dense)
        out.append(int(ids[0, torch.multinomial(probs[0], 1, generator=generator)]))
    return out


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_p", [1.0, 0.8, 0.5])
def test_duplex_candidates_preserves_tied_logits_and_top_p(dtype, top_p):
    logits = torch.tensor(
        [
            [4.0, 3.0, 3.0, 1.0],
            [5.0, 2.0, 1.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=dtype,
    )
    for top_k in [1, 2, 3]:
        filtered = Model._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        expected = torch.softmax(filtered, dim=-1)
        probs, ids = Model._duplex_top_k_top_p_candidates(logits, top_k=top_k, top_p=top_p)
        actual = torch.zeros_like(expected).scatter_(1, ids, probs)
        torch.testing.assert_close(actual, expected)

