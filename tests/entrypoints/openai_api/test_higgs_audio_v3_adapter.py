# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import gc
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.higgs_audio_v3 import HiggsAudioV3Adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ref_code_inflight_survives_request_cancellation():
    adapter = HiggsAudioV3Adapter(SpeechServingContext(server=SimpleNamespace()))
    encode_started = threading.Event()
    release_encode = threading.Event()
    calls = 0
    expected_codes = torch.arange(16, dtype=torch.long).reshape(2, 8) + 1

    def encode_reference_audio(_wav, _sr):
        nonlocal calls
        calls += 1
        encode_started.set()
        assert release_encode.wait(timeout=5)
        return torch.arange(16, dtype=torch.long).reshape(2, 8)

    def apply_delay_pattern(codes):
        return codes + 1

    async def resolve():
        return await adapter._resolve_higgs_audio_v3_ref_codes(
            "ref-a",
            object(),
            24000,
            encode_reference_audio,
            apply_delay_pattern,
        )

    async def run():
        creator = asyncio.create_task(resolve())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(encode_started.wait), timeout=5)

            cancelled_waiter = asyncio.create_task(resolve())
            survivor = asyncio.create_task(resolve())
            await asyncio.sleep(0)

            cancelled_waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_waiter

            creator.cancel()
            with pytest.raises(asyncio.CancelledError):
                await creator

            release_encode.set()
            survivor_result = await asyncio.wait_for(survivor, timeout=5)
            await asyncio.sleep(0)
            return survivor_result
        finally:
            release_encode.set()

    codes, cache_hit, inflight_wait = asyncio.run(run())

    assert calls == 1
    assert cache_hit is False
    assert inflight_wait is True
    assert torch.equal(codes, expected_codes)
    assert "ref-a" not in adapter._higgs_audio_v3_ref_code_inflight
    cached = adapter._get_higgs_audio_v3_ref_codes("ref-a")
    assert cached is not None
    assert torch.equal(cached, expected_codes)


def test_ref_code_inflight_consumes_failure_after_all_callers_cancel():
    adapter = HiggsAudioV3Adapter(SpeechServingContext(server=SimpleNamespace()))
    encode_started = threading.Event()
    release_encode = threading.Event()
    loop_errors = []

    def encode_reference_audio(_wav, _sr):
        encode_started.set()
        assert release_encode.wait(timeout=5)
        raise RuntimeError("encode failed")

    async def resolve():
        return await adapter._resolve_higgs_audio_v3_ref_codes(
            "ref-a",
            object(),
            24000,
            encode_reference_audio,
            lambda codes: codes,
        )

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        creator = asyncio.create_task(resolve())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(encode_started.wait), timeout=5)
            waiter = asyncio.create_task(resolve())
            await asyncio.sleep(0)

            creator.cancel()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await creator
            with pytest.raises(asyncio.CancelledError):
                await waiter

            release_encode.set()
            for _ in range(100):
                if "ref-a" not in adapter._higgs_audio_v3_ref_code_inflight:
                    break
                await asyncio.sleep(0.01)
            gc.collect()
            await asyncio.sleep(0)
        finally:
            release_encode.set()

    asyncio.run(run())

    assert "ref-a" not in adapter._higgs_audio_v3_ref_code_inflight
    assert not loop_errors
