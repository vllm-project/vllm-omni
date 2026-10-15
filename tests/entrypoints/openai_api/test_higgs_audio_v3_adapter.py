# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.higgs_audio_v3 import HiggsAudioV3Adapter
from vllm_omni.model_executor.models.higgs_audio_v3 import reference_encoder as ref

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _GatedCodec:
    """Encodes only once released; raises for clips starting with a negative sample."""

    config = SimpleNamespace(sample_rate=24000)
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def encode(self, input_values, return_dict=False):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=5)
        if float(input_values.reshape(-1)[0]) < 0:
            raise RuntimeError("encode failed")
        return torch.arange(8).reshape(1, 4, 2)


def _adapter(monkeypatch, codec: _GatedCodec) -> HiggsAudioV3Adapter:
    monkeypatch.setattr(ref, "load_audio_tokenizer", lambda: codec)
    stage = SimpleNamespace(runtime_config=SimpleNamespace(additional_config={"ref_audio_encode_workers": 0}))
    return HiggsAudioV3Adapter(SpeechServingContext(server=SimpleNamespace(_tts_stage=stage)))


def test_ref_code_encode_survives_request_cancellation(monkeypatch):
    codec = _GatedCodec()
    adapter = _adapter(monkeypatch, codec)
    wav = np.full(1920, 0.5, dtype=np.float32)

    async def run():
        await adapter.warmup()
        creator = asyncio.create_task(adapter._higgs_audio_v3_ref_codes(wav, 24000))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(codec.started.wait, 5), timeout=5)
            cancelled_waiter = asyncio.create_task(adapter._higgs_audio_v3_ref_codes(wav, 24000))
            survivor = asyncio.create_task(adapter._higgs_audio_v3_ref_codes(wav, 24000))
            await asyncio.sleep(0)
            for task in (cancelled_waiter, creator):
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            codec.release.set()
            return await asyncio.wait_for(survivor, timeout=5), await adapter._higgs_audio_v3_ref_codes(wav, 24000)
        finally:
            codec.release.set()

    survivor_codes, cached = asyncio.run(run())
    assert codec.calls == 1
    assert survivor_codes.shape == (2 + 4 - 1, 4)
    assert torch.equal(cached, survivor_codes)


def test_failed_ref_code_encode_is_reported_and_not_cached(monkeypatch):
    codec = _GatedCodec()
    codec.release.set()
    adapter = _adapter(monkeypatch, codec)
    wav = np.full(1920, -0.5, dtype=np.float32)
    loop_errors = []

    async def run():
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: loop_errors.append(context))
        for _ in range(2):
            with pytest.raises(RuntimeError, match="encode failed"):
                await adapter._higgs_audio_v3_ref_codes(wav, 24000)

    asyncio.run(run())
    assert codec.calls == 2
    assert not loop_errors
