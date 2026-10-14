# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 frame sampling (CUDA): the top-k / top-p filter of every draw, and the
repetition-aware (RAS) main-token draw.

The request parameters (top_k=1) make the first draw the arg-max token. The
re-draw uses fixed parameters (temperature 1.0, top_k 30, top_p 0.95), so over
many identical rows it spreads across the near-tied candidates it keeps. A row
whose first draw repeats its RAS window must therefore often return another
token; a row without a repeat always keeps the arg-max token.
"""

import math

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.fish_omni.sampling import EMPTY_RAS_SLOT, RAS_WINDOW_FRAMES, FusedFrameSampler

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "H100"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

EOS = 5
SEMANTIC_BEGIN = 100
CODEBOOK_SIZE = 128
ROWS = 2000
MAX_TOP_K = 32
CANDIDATES = torch.tensor([EOS, *range(SEMANTIC_BEGIN, SEMANTIC_BEGIN + CODEBOOK_SIZE)])
# Near-tied candidates: the arg-max semantic token, end of speech, and two more
# semantic tokens; top_p 0.95 drops the last of them.
ARGMAX, OTHER_A, OTHER_B = SEMANTIC_BEGIN + 1, SEMANTIC_BEGIN + 2, SEMANTIC_BEGIN + 3


def _rows(value, dtype=torch.float32, rows=ROWS):
    return torch.full((rows,), value, dtype=dtype, device="cuda")


def _sampler(rows=ROWS):
    sampler = FusedFrameSampler(max_top_k=MAX_TOP_K, num_codebooks=10, max_rows=rows, device=torch.device("cuda"))
    torch.manual_seed(0)
    return sampler, sampler.noise(rows, torch.device("cuda"))


def _frequencies(draws, width):
    return torch.bincount(draws, minlength=width).float() / draws.numel()


def test_top_p_uses_full_vocabulary_mass():
    """probs [0.4, 0.3, 0.2, 0.1, ~0...], top_k 2, top_p 0.9: tokens 0 and 1 both stay."""
    sampler, noise = _sampler()
    logits = torch.full((ROWS, 64), -30.0, device="cuda")
    logits[:, :4] = torch.tensor([math.log(p) for p in (0.4, 0.3, 0.2, 0.1)], device="cuda")
    draws = sampler.code(logits, _rows(1.0), _rows(2, torch.long), _rows(0.9), noise, 0)
    expected = torch.tensor([0.4 / 0.7, 0.3 / 0.7, 0.0, 0.0], device="cuda")
    assert torch.allclose(_frequencies(draws, 64)[:4], expected, atol=0.03)


def test_greedy_rows_in_a_mixed_batch_take_the_argmax():
    """A greedy request (temperature 0, or vLLM's -1) takes the argmax whatever shares its
    batch; sampled rows next to it still spread over the near-tied tokens."""
    sampler, noise = _sampler()
    logits = torch.full((ROWS, 64), -30.0, device="cuda")
    logits[:, :3] = torch.tensor([0.5, 1.0, 0.5], device="cuda")
    temperature = _rows(1.0)
    temperature[0::4] = 0.0
    temperature[2::4] = -1.0
    draws = sampler.code(logits, temperature, _rows(1 << 20, torch.long), _rows(1.0), noise, 0)
    assert (draws[0::2] == 1).all()
    assert set(draws[1::2].tolist()) == {0, 1, 2}


def test_top_p_keeps_the_most_likely_token():
    sampler, noise = _sampler()
    logits = torch.zeros((ROWS, 64), device="cuda")
    logits[:, 0] = 2.0
    draws = sampler.code(logits, _rows(1.0), _rows(1 << 20, torch.long), _rows(0.01), noise, 0)
    assert (draws == 0).all()


def _logits(top, second):
    """Candidate logits (columns follow CANDIDATES)."""
    logits = torch.full((ROWS, CANDIDATES.numel()), -30.0)
    column = {int(token): index for index, token in enumerate(CANDIDATES)}
    logits[:, column[OTHER_A]] = 0.0
    logits[:, column[OTHER_B]] = 0.0
    logits[:, column[second]] = 0.01
    logits[:, column[top]] = 0.02
    return logits.cuda()


def _window(history):
    """The RAS window the model keeps: the last RAS_WINDOW_FRAMES main tokens, left-padded."""
    recent = history[-RAS_WINDOW_FRAMES:]
    row = [EMPTY_RAS_SLOT] * (RAS_WINDOW_FRAMES - len(recent)) + recent
    return torch.tensor(row, device="cuda").expand(ROWS, -1)


def _sample(logits, history):
    sampler, noise = _sampler()
    return sampler.main_tokens(
        logits,
        CANDIDATES.cuda(),
        _window(history),
        _rows(1.0),
        _rows(1, torch.long),
        _rows(1.0),
        noise,
        SEMANTIC_BEGIN,
        CODEBOOK_SIZE,
    )


def test_non_repeated_token_keeps_first_draw():
    tokens, code_zero = _sample(_logits(ARGMAX, OTHER_A), history=[OTHER_A, OTHER_B])
    assert bool((tokens == ARGMAX).all())
    assert bool((code_zero == ARGMAX - SEMANTIC_BEGIN).all())


def test_repeated_semantic_token_is_redrawn():
    tokens, _ = _sample(_logits(ARGMAX, OTHER_A), history=[OTHER_B, ARGMAX])
    kept = (tokens == ARGMAX).float().mean().item()
    assert 0.3 < kept < 0.7
    assert set(tokens.tolist()) <= {ARGMAX, OTHER_A, OTHER_B}


def test_redraw_may_end_speech():
    tokens, _ = _sample(_logits(ARGMAX, EOS), history=[ARGMAX])
    assert bool((tokens == EOS).any())


def test_end_of_speech_is_never_redrawn():
    tokens, _ = _sample(_logits(EOS, OTHER_A), history=[EOS, EOS])
    assert bool((tokens == EOS).all())


def test_repeat_outside_window_keeps_first_draw():
    tokens, _ = _sample(_logits(ARGMAX, OTHER_A), history=[ARGMAX] + [OTHER_B] * RAS_WINDOW_FRAMES)
    assert bool((tokens == ARGMAX).all())
