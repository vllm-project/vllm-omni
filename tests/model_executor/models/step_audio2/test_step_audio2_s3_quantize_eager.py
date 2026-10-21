# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``_quantize_s3_graph`` must stay bit-identical to s3tokenizer's short-audio branch (eager fallback off-NPU).

The eager path inlines the else-branch of ``S3TokenizerV2.quantize`` so short
audio skips the implicit D2H sync that ``long_audio_mask.any()`` forces. This
test pins the inlined copy to the library on CPU with random weights, across
varied lengths, batch shapes, and the long-audio boundary.
"""

import threading

import pytest
import torch

from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    _S3_MAX_FRAMES,
    StepAudio2Token2WavCore,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

s3tokenizer = pytest.importorskip("s3tokenizer")


def _core_with(model) -> StepAudio2Token2WavCore:
    # The eager path only touches audio_tokenizer and device; skip the full
    # model bootstrap (yaml, campplus, vocoder weights). Core subclasses
    # nn.Module, so its base bookkeeping must be initialized before a child
    # module can be assigned. Seed the private fields directly so reads go
    # through the real `audio_tokenizer` property instead of relying on the
    # nn.Module.__getattr__ fallback after a failed `_ensure_models_loaded`.
    core = StepAudio2Token2WavCore.__new__(StepAudio2Token2WavCore)
    torch.nn.Module.__init__(core)
    core._models_loaded = True
    core._audio_tokenizer = model
    core.device = torch.device("cpu")
    # Graph bookkeeping normally declared in __init__ (skipped above).
    core._s3_graphs = {}
    core._s3_failed = set()
    core._s3_graph_lock = threading.Lock()
    core._s3_graph_done = None
    core._s3_replay_calls = 0
    core._s3_eager_calls = 0
    return core


def _batch(lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    max_len = max(lengths)
    mels = torch.randn(len(lengths), 128, max_len, generator=g)
    mels_lens = torch.tensor(lengths, dtype=torch.long)
    return mels, mels_lens


@pytest.mark.parametrize(
    "lengths",
    [[50], [50, 120, 200], [100, 100], [1, 2999], [3000]],
)
def test_short_audio_eager_matches_library(lengths) -> None:
    torch.manual_seed(0)
    model = s3tokenizer.model_v2.S3TokenizerV2("speech_tokenizer_v2_25hz").eval()
    core = _core_with(model)

    mels, mels_lens = _batch(lengths)
    with torch.inference_mode():
        lib_code, lib_len = model.quantize(mels, mels_lens)
        eager_code, eager_len = core._quantize_s3_graph(mels, mels_lens)

    assert torch.equal(lib_code, eager_code)
    assert torch.equal(lib_len, eager_len)


def test_max_frames_boundary_matches_library() -> None:
    # Both the library and the host-side check use strict ">", so exactly
    # _S3_MAX_FRAMES frames must take the short path on both sides.
    torch.manual_seed(1)
    model = s3tokenizer.model_v2.S3TokenizerV2("speech_tokenizer_v2_25hz").eval()
    core = _core_with(model)

    mels, mels_lens = _batch([_S3_MAX_FRAMES], seed=1)
    assert int(mels_lens.max()) <= _S3_MAX_FRAMES

    with torch.inference_mode():
        lib_code, lib_len = model.quantize(mels, mels_lens)
        eager_code, eager_len = core._quantize_s3_graph(mels, mels_lens)

    assert torch.equal(lib_code, eager_code)
    assert torch.equal(lib_len, eager_len)
