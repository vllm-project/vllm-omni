# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiMo-Audio per-request speech-code routing.

Guards the cross-request audio-corruption class (same class as RFC #4316
Issue 1 for MOSS-TTS): the fused thinker+talker emits one speech-code row per
request, so ``codes["audio"]`` must be a batch-ordered **list** for the
generic splitter (``to_payload_element``) to route ``element[idx]`` to
request ``idx``. A batched ``(num_reqs, 1, 8, 4)`` tensor is only split
correctly by the token-offset path when every request has exactly one
scheduled token; in a mixed prefill+decode step the tensor's first dim
matches neither token length, the splitter falls through to ``clone()``, and
every request receives the whole batch — leaking codes across requests and
crashing the async-chunk ragged-list conversion
(``expected sequence of length 36 at dim 1 (got 144)``).

CPU-only and weight-free: ``forward`` is driven on a bare model with
``generate_codes`` stubbed to return row-indexed codes, so assertions on the
payload prove which request got which row.
"""

from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")

from vllm_omni.utils.mm_outputs import to_payload_element  # noqa: E402

NUM_CODEBOOKS = 8
GROUP_SIZE = 4


@functools.lru_cache(maxsize=1)
def _model_cls():
    """Defer the import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.mimo_audio.mimo_audio import (
        MiMoAudioForConditionalGeneration,
    )

    return MiMoAudioForConditionalGeneration


def _row_indexed_codes(num_reqs: int) -> torch.Tensor:
    """(num_reqs, 1, 8, 4) tensor where request r's codes all hold r."""
    codes = torch.empty((num_reqs, 1, NUM_CODEBOOKS, GROUP_SIZE), dtype=torch.long)
    for r in range(num_reqs):
        codes[r] = r
    return codes


def _make_bare_model(next_speech_tokens, num_tokens: int):
    """A model whose only behaviour is the emit logic under test."""
    cls = _model_cls()
    model = cls.__new__(cls)
    model.model_stage = "fused_thinker_talker"
    hidden = torch.zeros((num_tokens, 16), dtype=torch.float32)
    model.generate_codes = lambda **kwargs: (next_speech_tokens, hidden)
    return model


def _emit(next_speech_tokens, num_tokens: int):
    model = _make_bare_model(next_speech_tokens, num_tokens)
    result = model.forward(
        input_ids=torch.zeros((num_tokens,), dtype=torch.long),
        positions=torch.zeros((num_tokens,), dtype=torch.long),
    )
    return result.multimodal_outputs["codes"]["audio"]


class TestPerRequestCodeRouting:
    def test_emit_is_batch_aligned_list(self):
        audio = _emit(_row_indexed_codes(3), num_tokens=3)

        assert isinstance(audio, list)
        assert len(audio) == 3  # one entry per request, batch-aligned
        for r, codes in enumerate(audio):
            assert codes.shape == (1, 1, NUM_CODEBOOKS, GROUP_SIZE)
            assert torch.all(codes == r)

    def test_no_codes_step_emits_none(self):
        audio = _emit(None, num_tokens=2)
        assert audio is None

    def test_mixed_prefill_decode_routes_by_request_index(self):
        """req 1 prefills 7 tokens while reqs 0 and 2 decode: the splitter
        must hand each request its own row, not the whole batch."""
        audio = _emit(_row_indexed_codes(3), num_tokens=9)

        seq_len = 9  # 1 (decode) + 7 (prefill) + 1 (decode) scheduled tokens
        spans = [(0, 1), (1, 8), (8, 9)]
        for idx, (start, end) in enumerate(spans):
            routed = to_payload_element(
                audio,
                idx,
                start,
                end,
                seq_len=seq_len,
                scheduled_seq_len=seq_len,
            )
            assert routed.shape == (1, 1, NUM_CODEBOOKS, GROUP_SIZE)
            assert torch.all(routed == idx), f"req {idx} received another request's codes"

    def test_pure_decode_routes_by_request_index(self):
        """One token per request (the only case the old tensor emit handled)."""
        audio = _emit(_row_indexed_codes(4), num_tokens=4)

        for idx in range(4):
            routed = to_payload_element(audio, idx, idx, idx + 1, seq_len=4, scheduled_seq_len=4)
            assert torch.all(routed == idx)

    def test_routed_codes_do_not_alias_the_emit(self):
        """The splitter clones list elements; mutating the routed payload must
        not corrupt another request's view of the emit."""
        audio = _emit(_row_indexed_codes(2), num_tokens=5)
        routed = to_payload_element(audio, 0, 0, 4, seq_len=5, scheduled_seq_len=5)
        routed.fill_(99)
        assert torch.all(audio[0] == 0)
