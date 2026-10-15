# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched reference speech-token extraction returns each caller its own tokens."""

import threading
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _s3(calls):
    def padding(mels):
        lens = torch.tensor([m.shape[-1] for m in mels])
        out = torch.zeros(len(mels), 1, int(lens.max()))
        for i, m in enumerate(mels):
            out[i, :, : m.shape[-1]] = m
        return out, lens

    def quantize(mels, lens):
        calls.append(int(mels.shape[0]))
        if bool((mels < 0).any()):
            raise RuntimeError("bad mel")
        # Token t of a row is its first mel value plus t; padding shows up as garbage.
        codes = mels[:, 0, :1].long() + torch.arange(mels.shape[-1]).unsqueeze(0)
        return codes, lens

    return SimpleNamespace(padding=padding), SimpleNamespace(quantize=quantize)


def test_speech_token_batcher_returns_each_rows_tokens_under_concurrency():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _SpeechTokenBatcher

    calls: list[int] = []
    s3, model = _s3(calls)
    batcher = _SpeechTokenBatcher(model, s3, "cpu", max_batch=8)
    results: dict[int, torch.Tensor] = {}
    start = threading.Barrier(24)

    def worker(i):
        start.wait()
        results[i] = batcher.tokens(torch.full((1, 3 + i % 5), float(100 * i)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for i in range(24):
        assert results[i].tolist() == [100 * i + t for t in range(3 + i % 5)]
    assert sum(calls) == 24 and max(calls) <= 8


def test_speech_token_batcher_propagates_errors_and_keeps_serving():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _SpeechTokenBatcher

    s3, model = _s3([])
    batcher = _SpeechTokenBatcher(model, s3, "cpu")
    with pytest.raises(RuntimeError, match="bad mel"):
        batcher.tokens(torch.full((1, 4), -1.0))
    assert batcher.tokens(torch.full((1, 2), 7.0)).tolist() == [7, 8]
