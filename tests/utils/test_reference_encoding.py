# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``ReferenceEncodingService``: content keys, single-flight, micro-batching and the byte budget."""

import asyncio
import threading

import numpy as np
import pytest

from vllm_omni.utils.reference_encoding import ReferenceEncodingService, content_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Encoder:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.batches: list[list[int]] = []
        self.gate = gate

    def __call__(self, items: list[int]) -> list[object]:
        if self.gate is not None:
            self.gate.wait(5)
        self.batches.append(list(items))
        return [ValueError("bad clip") if item < 0 else item * 10 for item in items]


def _service(encoder: _Encoder, **kwargs) -> ReferenceEncodingService:
    kwargs.setdefault("window_ms", 20.0)
    return ReferenceEncodingService(encoder, nbytes=lambda artifact: 8, **kwargs)


def test_content_key_depends_on_samples_rate_and_namespace() -> None:
    wav = np.linspace(-1, 1, 2048, dtype=np.float32)
    key = content_key("model-a", wav, 24000)
    assert key == content_key("model-a", wav.astype(np.float64), 24000)
    assert key != content_key("model-a", wav, 16000)
    assert key != content_key("model-b", wav, 24000)
    assert key != content_key("model-a", wav[:-1], 24000)


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_encode_and_later_ones_hit() -> None:
    encoder = _Encoder()
    service = _service(encoder)
    try:
        results = await asyncio.gather(*[service.get("clip", 3) for _ in range(5)])
        assert results == [30] * 5
        assert encoder.batches == [[3]]
        assert await service.get("clip", 3) == 30
        assert encoder.batches == [[3]]
        assert service.stats()["hits"] == 1 and service.stats()["misses"] == 1
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_cold_clips_in_one_window_share_a_batch_and_failures_stay_per_clip() -> None:
    encoder = _Encoder()
    service = _service(encoder)
    try:
        results = await asyncio.gather(
            service.get("a", 1), service.get("b", -1), service.get("c", 2), return_exceptions=True
        )
        assert results[0] == 10 and results[2] == 20
        assert isinstance(results[1], ValueError)
        assert encoder.batches == [[1, -1, 2]]
        # A failed clip is not cached; a retry encodes it again.
        assert len(service) == 2
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_byte_budget_evicts_least_recently_used() -> None:
    encoder = _Encoder()
    service = _service(encoder, max_bytes=16, window_ms=0.0)
    try:
        for key, item in (("a", 1), ("b", 2)):
            await service.get(key, item)
        await service.get("a", 1)  # refresh a
        await service.get("c", 3)  # evicts b
        assert service.lookup("a") == 10 and service.lookup("c") == 30
        assert service.lookup("b") is None
        assert service.cached_bytes == 16
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_the_shared_encode() -> None:
    gate = threading.Event()
    encoder = _Encoder(gate)
    service = _service(encoder, window_ms=0.0)
    try:
        first = asyncio.ensure_future(service.get("clip", 4))
        second = asyncio.ensure_future(service.get("clip", 4))
        await asyncio.sleep(0.05)
        first.cancel()
        gate.set()
        assert await second == 40
        assert encoder.batches == [[4]]
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_generator_encoder_releases_each_item_before_the_batch_ends() -> None:
    first_released = threading.Event()

    def encode(items):
        yield items[0] * 10
        # The second item waits until the first caller holds its result.
        assert first_released.wait(5)
        yield items[1] * 10

    service = ReferenceEncodingService(encode, nbytes=lambda artifact: 8, window_ms=20.0)
    try:
        first = asyncio.ensure_future(service.get("a", 1))
        second = asyncio.ensure_future(service.get("b", 2))
        assert await first == 10
        first_released.set()
        assert await second == 20
    finally:
        await service.aclose()
