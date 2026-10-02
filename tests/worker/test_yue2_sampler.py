# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2's batched sampler draws exactly what the per-row reference draws.

The model samples all rows of a preset as one batch, over the phase's own
columns only, and writes the multinomial out as ``argmax(p / q)`` with
``q ~ Exp(1)`` so the draw stays on the device. The reference below is the
upstream per-row arithmetic over the full vocabulary with
``torch.multinomial``. Same seeds must give the same tokens, and each
request's generator must advance by exactly the same amount per step, or a
request's song changes with the batch it happens to share.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.yue2.yue2 import (
    ABC_END,
    ABC_SAMPLING,
    CODEC_OFFSET,
    CODEC_SIZE,
    EOD,
    MUSIC_END,
    PHASE_VOCAB,
    SEMANTIC_SAMPLING,
    VOCAB_SIZE,
    sample_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _reference_scores(logits: torch.Tensor, preset: dict, history: list[int], phase: str) -> torch.Tensor:
    """Upstream per-row scores over the full vocabulary (float32)."""
    scores = logits.float()  # a new tensor: the logits are bf16
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[:EOD] = 0
    else:
        allowed[CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[end] = 0
    scores = scores + allowed
    if len(history) < preset["min_tokens"]:
        scores[end] = -torch.inf
    recent = history[-preset["penalty_window"] :]
    if preset["repetition_penalty"] != 1.0 and recent:
        freq = torch.zeros_like(scores)
        freq.scatter_add_(-1, torch.tensor(recent), torch.ones(len(recent)))
        alpha = preset["repetition_penalty"] ** freq
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)
    if preset["temperature"] == 0:
        return scores
    scores = scores / preset["temperature"]
    threshold = scores.topk(preset["top_k"]).values[-1]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    values, indices = scores.sort(descending=True, stable=True)
    probabilities = values.softmax(-1)
    removed = probabilities.cumsum(-1) - probabilities > preset["top_p"]
    removed[:1] = False
    values = values.masked_fill(removed, -torch.inf)
    return values.scatter(-1, indices, values)


def _reference_draw(scores: torch.Tensor, generator: torch.Generator, greedy: bool) -> int:
    if greedy:
        return int(scores.argmax())
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator))


def _batched_draw(logits, preset, histories, phase, generators) -> list[int]:
    vocab = PHASE_VOCAB[phase]
    window = preset["penalty_window"]
    recent = [[vocab.col(t) for t in history[-window:]] for history in histories]
    width = max(map(len, recent))
    window_cols = torch.tensor([r + [vocab.num_cols] * (width - len(r)) for r in recent]) if width else None
    block = [len(history) < preset["min_tokens"] for history in histories]
    ids, bad = sample_rows(
        logits,
        vocab,
        temperature=preset["temperature"],
        top_p=preset["top_p"],
        top_k=preset["top_k"],
        repetition_penalty=preset["repetition_penalty"],
        window_cols=window_cols,
        block_end=torch.tensor(block) if any(block) else None,
        generators=generators,
    )
    assert not bad.any()
    return ids.tolist()


@pytest.mark.parametrize(
    ("phase", "rows", "min_tokens", "temperature"),
    [
        ("semantic", 1, None, None),
        ("semantic", 3, None, None),
        ("semantic", 3, 0, None),  # the end token may be drawn
        ("semantic", 2, None, 0.0),  # greedy
        ("abc", 2, None, None),
        ("abc", 2, 0, None),
    ],
)
def test_batched_draws_match_per_row_multinomial(phase, rows, min_tokens, temperature) -> None:
    preset = dict(ABC_SAMPLING if phase == "abc" else SEMANTIC_SAMPLING)
    if min_tokens is not None:
        preset["min_tokens"] = min_tokens
    if temperature is not None:
        preset["temperature"] = temperature
    end = ABC_END if phase == "abc" else MUSIC_END
    reference = [torch.Generator().manual_seed(1000 + row) for row in range(rows)]
    batched = [torch.Generator().manual_seed(1000 + row) for row in range(rows)]
    histories: list[list[int]] = [[] for _ in range(rows)]
    logits_generator = torch.Generator().manual_seed(7)
    for _ in range(40):
        logits = (torch.randn((rows, VOCAB_SIZE), generator=logits_generator) * 3).to(torch.bfloat16)
        # Peaky over a few ids, like a real LM, so the penalty and top-p matter.
        if phase == "semantic":
            logits[:, CODEC_OFFSET : CODEC_OFFSET + 300] += 6
        else:
            logits[:, :300] += 6
        logits[:, end] += 4
        expected = [
            _reference_draw(
                _reference_scores(logits[row], preset, histories[row], phase), reference[row], temperature == 0
            )
            for row in range(rows)
        ]
        assert _batched_draw(logits, preset, histories, phase, batched) == expected
        for row in range(rows):
            # A greedy request never reads its stream (its temperature is
            # fixed for life), so only sampling rows must advance in step.
            if temperature != 0:
                assert torch.equal(reference[row].get_state(), batched[row].get_state())
            if expected[row] != end:
                histories[row].append(expected[row])


def test_a_row_draws_the_same_alone_or_in_a_batch() -> None:
    """A request's song must not depend on which requests share its step."""
    preset = SEMANTIC_SAMPLING
    logits = torch.randn((3, VOCAB_SIZE), generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    histories = [[CODEC_OFFSET + i for i in range(60)], [], [CODEC_OFFSET + 5] * 10]
    together = _batched_draw(
        logits, preset, histories, "semantic", [torch.Generator().manual_seed(s) for s in (1, 2, 3)]
    )
    alone = [
        _batched_draw(
            logits[row : row + 1], preset, histories[row : row + 1], "semantic", [torch.Generator().manual_seed(s)]
        )[0]
        for row, s in enumerate((1, 2, 3))
    ]
    assert together == alone


@pytest.mark.parametrize("phase", ["abc", "semantic"])
def test_phase_vocab_columns_round_trip(phase) -> None:
    vocab = PHASE_VOCAB[phase]
    end = ABC_END if phase == "abc" else MUSIC_END
    ids = [end, 0, EOD - 1] if phase == "abc" else [end, CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE - 1]
    cols = torch.tensor([vocab.col(i) for i in ids])
    assert vocab.col(end) == vocab.end_col
    assert cols.max() < vocab.num_cols
    assert vocab.ids(cols).tolist() == ids
    logits = torch.randn((2, VOCAB_SIZE))
    assert torch.equal(vocab.take(logits)[:, cols], logits[:, ids])
