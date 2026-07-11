# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard against out-of-vocabulary stop ids in min-tokens masking (#4962).

vLLM's input processor folds the stage tokenizer's EOS id into
``SamplingParams.all_stop_token_ids``. For AR stages whose lm_head is
narrower than the tokenizer vocabulary (codec talkers such as Qwen3-TTS:
3072 logits vs text EOS 151645), any ``min_tokens >= 1`` makes
``MinTokensLogitsProcessor`` write ``-inf`` at an out-of-range index and
the engine dies with a CUDA device-side assert.
"""

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessors,
    MinTokensLogitsProcessor,
)

from vllm_omni.worker.sampling_utils import sanitize_min_tokens_stop_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TALKER_VOCAB = 3072
CODEC_EOS = 2150
TEXT_EOS = 151645


def _make_min_tokens_proc(stop_token_ids: list[int], eos_token_id: int | None) -> MinTokensLogitsProcessor:
    """Build a MinTokensLogitsProcessor with one active request, mirroring
    the production path (engine folds the tokenizer EOS via
    ``update_from_generation_config``)."""
    params = SamplingParams(min_tokens=2, stop_token_ids=stop_token_ids)
    if eos_token_id is not None:
        params.update_from_generation_config({}, eos_token_id)
    proc = MinTokensLogitsProcessor(None, device=torch.device("cpu"), is_pin_memory=False)
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, None, [])],
            moved=[],
        )
    )
    return proc


def test_oob_stop_id_crashes_without_guard():
    """Document the failure mode: the folded text EOS is out of range for a
    narrow codec head and index_put_ raises (device-side assert on CUDA)."""
    proc = _make_min_tokens_proc([CODEC_EOS], eos_token_id=TEXT_EOS)
    logits = torch.zeros(1, TALKER_VOCAB)
    with pytest.raises((IndexError, RuntimeError)):
        proc.apply(logits)


def test_guard_filters_oob_and_keeps_in_range_mask():
    proc = _make_min_tokens_proc([CODEC_EOS], eos_token_id=TEXT_EOS)
    sanitize_min_tokens_stop_ids(LogitsProcessors([proc]), TALKER_VOCAB)

    logits = torch.zeros(1, TALKER_VOCAB)
    out = proc.apply(logits)

    assert out[0, CODEC_EOS] == -float("inf")
    masked = (out == -float("inf")).nonzero()
    assert masked.tolist() == [[0, CODEC_EOS]]


def test_guard_noop_when_all_stop_ids_in_range():
    """Full-width heads (e.g. CosyVoice3 talker) must be untouched: no
    tensor rebuild, mask unchanged."""
    proc = _make_min_tokens_proc([CODEC_EOS], eos_token_id=None)
    slice_before = proc.logits_slice

    sanitize_min_tokens_stop_ids(LogitsProcessors([proc]), TALKER_VOCAB)

    assert proc.logits_slice is slice_before
    out = proc.apply(torch.zeros(1, TALKER_VOCAB))
    assert out[0, CODEC_EOS] == -float("inf")


def test_guard_sanitizes_params_only_once():
    """The stop-id set is shared with the request's SamplingParams, so the
    second call finds nothing to drop (no repeated rebuilds/warnings)."""
    proc = _make_min_tokens_proc([CODEC_EOS], eos_token_id=TEXT_EOS)
    sanitize_min_tokens_stop_ids(LogitsProcessors([proc]), TALKER_VOCAB)
    slice_after_first = proc.logits_slice

    sanitize_min_tokens_stop_ids(LogitsProcessors([proc]), TALKER_VOCAB)

    assert proc.logits_slice is slice_after_first


def test_guard_handles_multiple_requests():
    params_oob = SamplingParams(min_tokens=4, stop_token_ids=[CODEC_EOS])
    params_oob.update_from_generation_config({}, TEXT_EOS)
    params_ok = SamplingParams(min_tokens=4, stop_token_ids=[7])
    proc = MinTokensLogitsProcessor(None, device=torch.device("cpu"), is_pin_memory=False)
    proc.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[(0, params_oob, None, []), (1, params_ok, None, [])],
            moved=[],
        )
    )

    sanitize_min_tokens_stop_ids(LogitsProcessors([proc]), TALKER_VOCAB)
    out = proc.apply(torch.zeros(2, TALKER_VOCAB))

    assert out[0, CODEC_EOS] == -float("inf")
    assert out[1, 7] == -float("inf")
    assert int((out == -float("inf")).sum()) == 2
