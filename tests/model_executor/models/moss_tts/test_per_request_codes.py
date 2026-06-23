# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MOSS-TTS per-request code routing in the talker.

Guards the cross-request audio-corruption class (RFC #4316 Issue 1, bug report
#4355): under continuous batching every Stage-1 request must decode its *own*
codes, not request 0's. The talker emits ``codes["audio"]`` as a batch-ordered
**list** so the generic splitter (``to_payload_element``) routes
``element[idx]`` to request ``idx``; the list must stay index-aligned with the
batch (placeholder for skipped requests) or a shorter list silently falls back
to entry 0 — the original corruption. The hidden-row mapping must also use the
runner's real per-request token spans, not an equal split, or mixed
prefill+decode steps sample each request's codes from the wrong rows.

CPU-only and weight-free: ``make_omni_output`` is driven directly on a bare
talker with ``_sample_audio_codes`` mocked to echo the hidden row it was given,
so an assertion on the output value proves which row each request sampled from.
"""

from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")

N_VQ = 4


@functools.lru_cache(maxsize=1)
def _delay_talker_cls():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    return MossTTSDelayTalkerForGeneration


def _make_bare_delay_talker():
    """A talker whose only behaviour is the routing/span logic under test.

    ``_sample_audio_codes`` is replaced with a stub returning an (N_VQ,) tensor
    filled with the value of the hidden row it received. With a ``hidden`` where
    row r holds the scalar r, the emitted codes reveal which row each request
    sampled.
    """
    cls = _delay_talker_cls()
    talker = cls.__new__(cls)
    talker._batch_state = None
    talker.n_vq = N_VQ

    def _stub_sample(last_h, state):
        row_value = int(last_h.reshape(-1)[0].item())
        return torch.full((N_VQ,), row_value, dtype=torch.long)

    talker._sample_audio_codes = _stub_sample
    return talker


def _row_indexed_hidden(num_rows: int) -> torch.Tensor:
    """(num_rows, 1) tensor where row r holds the scalar r."""
    return torch.arange(num_rows, dtype=torch.float32).reshape(num_rows, 1)


def _audio_payload(result):
    return result.multimodal_outputs["codes"]["audio"]


class TestPerRequestRouting:
    def test_mixed_prefill_decode_routes_by_span(self):
        """req 0 prefills 3 rows, req 1 decodes 1 row in the same step.

        With real spans, req 0 samples its last row (2) and req 1 its last row
        (3); the payload is a batch-aligned list, element[idx] per request.
        """
        talker = _make_bare_delay_talker()
        hidden = _row_indexed_hidden(4)
        info_dicts = [{}, {}]
        spans = [(0, 3), (3, 4)]

        result = talker.make_omni_output(
            hidden,
            runtime_additional_information=info_dicts,
            request_token_spans=spans,
        )
        audio = _audio_payload(result)

        assert isinstance(audio, list)
        assert len(audio) == 2  # one entry per request, batch-aligned
        assert audio[0].shape == (1, N_VQ)
        assert torch.all(audio[0] == 2)  # req 0: last row of span [0,3)
        assert torch.all(audio[1] == 3)  # req 1: last row of span [3,4)
        # No cross-request leakage.
        assert not torch.equal(audio[0], audio[1])

    def test_no_request_samples_outside_its_span(self):
        talker = _make_bare_delay_talker()
        spans = [(0, 2), (2, 5), (5, 6)]
        hidden = _row_indexed_hidden(6)
        info_dicts = [{}, {}, {}]

        result = talker.make_omni_output(
            hidden,
            runtime_additional_information=info_dicts,
            request_token_spans=spans,
        )
        audio = _audio_payload(result)

        assert len(audio) == 3
        sampled = []
        for i, (start, end) in enumerate(spans):
            value = int(audio[i].reshape(-1)[0].item())
            assert start <= value < end, f"req {i} sampled row {value} outside [{start},{end})"
            sampled.append(value)
        assert len(set(sampled)) == len(sampled)  # distinct rows, no sharing

    def test_skipped_request_keeps_list_aligned(self):
        """An empty-span request gets an empty placeholder at its own index, so
        later requests are not shifted onto request 0's codes."""
        talker = _make_bare_delay_talker()
        # req 1 has an empty span (start == end) and must not displace req 2.
        spans = [(0, 1), (1, 1), (1, 2)]
        hidden = _row_indexed_hidden(2)
        info_dicts = [{}, {}, {}]

        result = talker.make_omni_output(
            hidden,
            runtime_additional_information=info_dicts,
            request_token_spans=spans,
        )
        audio = _audio_payload(result)

        assert len(audio) == 3
        assert torch.all(audio[0] == 0)  # req 0: row 0
        assert audio[1].numel() == 0  # req 1: empty placeholder, aligned
        assert torch.all(audio[2] == 1)  # req 2: row 1, not shifted

    def test_accumulates_into_per_request_history(self):
        talker = _make_bare_delay_talker()
        hidden = _row_indexed_hidden(2)
        prior = torch.full((1, N_VQ), 7, dtype=torch.long)
        info_dicts = [{"audio_codes": {"accumulated": prior}}, {}]
        spans = [(0, 1), (1, 2)]

        result = talker.make_omni_output(
            hidden,
            runtime_additional_information=info_dicts,
            request_token_spans=spans,
        )
        audio = _audio_payload(result)

        assert audio[0].shape == (2, N_VQ)
        assert torch.all(audio[0][0] == 7)  # prior history kept
        assert torch.all(audio[0][1] == 0)  # this step's row 0 appended
        assert torch.all(audio[1] == 1)

    def test_runs_without_spans_backward_compat(self):
        """Missing request_token_spans must not crash (equal-split fallback)
        and still yields a batch-aligned list."""
        talker = _make_bare_delay_talker()
        hidden = _row_indexed_hidden(2)
        info_dicts = [{}, {}]

        result = talker.make_omni_output(hidden, runtime_additional_information=info_dicts)
        audio = _audio_payload(result)
        assert isinstance(audio, list)
        assert len(audio) == 2

    def test_no_codes_returns_empty_payload(self):
        """Zero hidden rows → no codes → empty multimodal payload (no list)."""
        talker = _make_bare_delay_talker()
        hidden = torch.zeros((0, 1), dtype=torch.float32)
        result = talker.make_omni_output(hidden, runtime_additional_information=[{}], request_token_spans=[(0, 0)])
        assert result.multimodal_outputs == {}
