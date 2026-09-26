# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only checks for the YuE2 prompt/token contract and sampling arithmetic.

Run: pytest tests/model_executor/models/test_yue2_contract.py
"""

import pytest
import torch

from vllm_omni.model_executor.models.yue2.constants import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    EOD,
    MUSIC_END,
    MUSIC_START,
)
from vllm_omni.model_executor.models.yue2.prompt import (
    abc_ids_from_generated,
    abc_prefix_ids,
    semantic_frames,
    semantic_prefix_ids,
)
from vllm_omni.model_executor.models.yue2.sampling import distribution, sample_row

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode(text):
    # Stand-in for the tiktoken BPE: any deterministic id under EOD works for
    # structural checks.
    return [EOD - 1 - (i % 97) for i, _ in enumerate(text)]


def test_abc_prefix_shape():
    ids = abc_prefix_ids(_encode, "style", "lyrics", "full")
    assert ids[0] == EOD
    assert ids[-1] == ABC_START
    assert ABC_START not in ids[:-1]


def test_semantic_prefix_off_mode_skips_abc():
    ids = semantic_prefix_ids(_encode, "s", "l", "off")
    assert ids[-3:] == [ABC_START, ABC_END, MUSIC_START]


def test_semantic_prefix_embeds_abc():
    ids = semantic_prefix_ids(_encode, "s", "l", "full", abc_ids=[5, 6, 7])
    assert ids[-4:] == [5, 6, 7, ABC_END, MUSIC_START][-4:]


def test_semantic_prefix_rejects_out_of_vocab_abc():
    with pytest.raises(ValueError):
        semantic_prefix_ids(_encode, "s", "l", "full", abc_ids=[EOD + 5])


def test_abc_ids_roundtrip():
    generated = [ABC_START, 10, 20, ABC_END]
    assert abc_ids_from_generated(generated) == [10, 20]
    assert abc_ids_from_generated([10, 20]) == [10, 20]


def test_semantic_frames_offset():
    assert semantic_frames([CODEC_OFFSET + 3, CODEC_OFFSET + 4, MUSIC_END]) == [3, 4]


def test_distribution_masks_by_phase():
    logits = torch.zeros(1, 184704)
    scores = distribution(
        logits,
        temperature=1.0,
        top_p=1.0,
        top_k=40000,
        repetition_penalty=1.0,
        penalty_window=50,
        history=[],
        step=0,
        min_tokens=0,
        phase="semantic",
    )
    assert scores.dim() == 2
    finite = torch.isfinite(scores).sum()
    assert int(finite) == 32768 + 1
    assert scores[0, MUSIC_END].isfinite()
    assert not scores[0, EOD].isfinite()


def test_distribution_min_tokens_blocks_end():
    logits = torch.zeros(184704)
    scores = distribution(
        logits,
        temperature=1.0,
        top_p=1.0,
        top_k=1000,
        repetition_penalty=1.0,
        penalty_window=50,
        history=[],
        step=0,
        min_tokens=200,
        phase="semantic",
    )
    assert scores.dim() == 1
    assert not scores[MUSIC_END].isfinite()


def test_sample_row_is_seed_deterministic():
    logits = torch.full((184704,), -10.0)
    logits[CODEC_OFFSET + 5] = 5.0
    scores = distribution(
        logits,
        temperature=1.0,
        top_p=0.95,
        top_k=100,
        repetition_penalty=1.0,
        penalty_window=50,
        history=[],
        step=0,
        min_tokens=0,
        phase="semantic",
    )
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert sample_row(scores, g1) == sample_row(scores, g2)


def test_window_penalty_pushes_down_repeats():
    from vllm_omni.model_executor.models.yue2.sampling import window_penalty

    # Upstream arithmetic: positive logits are divided by penalty**freq,
    # negative logits multiplied; zero logits stay zero. id 0 seen twice →
    # freq 2 → alpha 4 → 1/4; untouched ids keep their logit. The old
    # assertion (out[0] < 0 == out[1]) chained-compared its way into
    # demanding a sign change on a zero logit.
    logits = torch.ones(3)
    out = window_penalty(logits, [0, 0], 2.0)
    assert out[0] == pytest.approx(0.25)
    assert out[1] == out[2] == pytest.approx(1.0)
    out_neg = window_penalty(-logits, [0, 0], 2.0)
    assert out_neg[0] == pytest.approx(-4.0)
    assert out_neg[1] == out_neg[2] == pytest.approx(-1.0)
    assert window_penalty(torch.zeros(3), [0, 0], 2.0).eq(0).all()
