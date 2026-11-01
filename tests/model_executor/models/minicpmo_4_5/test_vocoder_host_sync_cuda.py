# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 vocoder stage's per-step path must not block the host."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import hifigan
from vllm_omni.worker.gpu_generation_model_runner import _HostCopyBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@contextmanager
def _no_host_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def test_istft_is_bitwise_and_sync_free_on_cuda():
    n_fft, hop = 16, 4
    window = torch.hann_window(n_fft, periodic=True, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    magnitude = torch.rand(4, n_fft // 2 + 1, 300, device="cuda", generator=generator) * 3
    phase = torch.randn(4, n_fft // 2 + 1, 300, device="cuda", generator=generator)
    spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    envelopes: dict = {}
    expected = torch.istft(spec, n_fft, hop, n_fft, window=window)
    # The first call per frame count builds and checks the envelope.
    hifigan._istft_without_host_sync(spec, n_fft, hop, window, envelopes)
    with _no_host_sync():
        actual = hifigan._istft_without_host_sync(spec, n_fft, hop, window, envelopes)
    assert torch.equal(actual, expected)


def test_host_copy_batch_copies_without_blocking_until_wait():
    to_host = _HostCopyBatch(pin_memory=True)
    outputs = [torch.randn(1000, device="cuda") * (row + 1) for row in range(4)]
    strided = torch.randn(8, 6, device="cuda").t()
    to_host.copy(torch.zeros(1, device="cuda"))
    to_host.wait()
    with _no_host_sync():
        copies = [to_host.copy(tensor) for tensor in outputs]
        copied_strided = to_host.copy(strided)
    to_host.wait()
    for copy, tensor in zip(copies, outputs, strict=True):
        assert copy.device.type == "cpu" and copy.is_pinned()
        assert torch.equal(copy, tensor.cpu())
    assert copied_strided.is_contiguous() and torch.equal(copied_strided, strided.cpu())
