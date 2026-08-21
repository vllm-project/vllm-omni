# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MOSS-TTS reference encoder cache/batching layer."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.reference_encoder import (
    create_reference_encoder,
)

pytestmark = [pytest.mark.cpu, pytest.mark.tts]


class _FakeMossProcessor:
    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        fail_batched: bool = False,
    ) -> None:
        self.delay_s = delay_s
        self.fail_batched = fail_batched
        self.calls: list[list[torch.Tensor]] = []
        # Concurrency instrumentation: workers increment ``_active`` on entry so
        # a test can prove that distinct encodes overlap (run in parallel).
        self._active = 0
        self._active_lock = threading.Lock()
        self.max_active = 0

    def encode_audios_from_wav(
        self,
        waveforms: list[torch.Tensor],
        *,
        sampling_rate: int,
        n_vq: int,
    ) -> list[torch.Tensor]:
        self.calls.append([wav.detach().clone() for wav in waveforms])
        with self._active_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.fail_batched and len(waveforms) > 1:
                raise RuntimeError("batched encode failed")

            results: list[torch.Tensor] = []
            for wav in waveforms:
                flat = wav.reshape(-1)
                if flat.numel() > 0 and float(flat[0].item()) < 0:
                    raise RuntimeError("bad waveform")
                value = int(round(float(flat.sum().item()) * 1000)) % 997
                frames = max(1, int(wav.shape[-1]))
                results.append(
                    torch.full((frames, int(n_vq)), value, dtype=torch.long)
                )
            return results
        finally:
            with self._active_lock:
                self._active -= 1


def _resolver_for(payloads: dict[str, tuple[list[float], int]]):
    async def _resolve(ref: str) -> tuple[list[float], int]:
        await asyncio.sleep(0)
        return payloads[ref]

    return _resolve


async def _encode(
    encoder: Any,
    ref: str,
    payloads: dict[str, tuple[list[float], int]],
) -> torch.Tensor:
    return await encoder.encode_reference_codes(
        ref,
        resolve_ref_audio=_resolver_for(payloads),
    )


def test_cache_hit_returns_independent_cpu_long_clone() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor()
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=4,
            sr_target=24000,
            max_batch_wait_ms=0,
            enable_cache=True,
        )
        payloads = {"same": ([0.1, 0.2, 0.3], 24000)}
        try:
            first = await _encode(encoder, "same", payloads)
            first[0, 0] = 123456
            second = await _encode(encoder, "same", payloads)
        finally:
            encoder.close()

        assert len(processor.calls) == 1
        assert second.device.type == "cpu"
        assert second.dtype == torch.long
        assert int(second[0, 0].item()) != 123456
        assert encoder.stats()["hits"] == 1

    asyncio.run(_run())


def test_single_flight_merges_concurrent_same_reference() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor(delay_s=0.05)
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=4,
            sr_target=24000,
            max_batch_wait_ms=10,
            enable_cache=True,
        )
        payloads = {"same": ([0.1, 0.2, 0.3, 0.4], 24000)}
        try:
            outputs = await asyncio.gather(
                *(_encode(encoder, "same", payloads) for _ in range(5))
            )
        finally:
            encoder.close()

        assert len(processor.calls) == 1
        assert all(torch.equal(outputs[0], item) for item in outputs[1:])
        assert len({item.data_ptr() for item in outputs}) == len(outputs)
        stats = encoder.stats()
        assert stats["misses"] == 1
        assert stats["merged"] == 4

    asyncio.run(_run())


def test_single_flight_survives_cancelled_leader() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor(delay_s=0.05)
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=4,
            sr_target=24000,
            max_batch_wait_ms=0,
            enable_cache=True,
        )
        payloads = {"same": ([0.1, 0.2, 0.3, 0.4], 24000)}
        try:
            leader = asyncio.create_task(_encode(encoder, "same", payloads))
            for _ in range(100):
                stats = encoder.stats()
                if stats["misses"] == 1 and stats["inflight"] == 1:
                    break
                await asyncio.sleep(0.001)
            else:
                raise AssertionError("leader did not register inflight encode")

            leader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await leader

            follower = await _encode(encoder, "same", payloads)
            hit = await _encode(encoder, "same", payloads)
        finally:
            encoder.close()

        assert torch.equal(follower, hit)
        assert sum(len(call) for call in processor.calls) == 1
        stats = encoder.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["merged"] == 1
        assert stats["inflight"] == 0

    asyncio.run(_run())


def test_uncached_references_are_batched_with_short_wait_window() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor()
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=3,
            sr_target=24000,
            max_batch_size=4,
            max_batch_wait_ms=40,
            enable_cache=False,
        )
        payloads = {
            "a": ([0.1, 0.2], 24000),
            "b": ([0.3, 0.4], 24000),
            "c": ([0.5, 0.6], 24000),
        }
        try:
            await asyncio.gather(*(_encode(encoder, ref, payloads) for ref in payloads))
        finally:
            encoder.close()

        assert [len(call) for call in processor.calls] == [3]

    asyncio.run(_run())


def test_batch_failure_retries_per_item_and_isolates_bad_waveform() -> None:
    async def _run() -> tuple[torch.Tensor, BaseException]:
        processor = _FakeMossProcessor(fail_batched=True)
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            max_batch_size=2,
            max_batch_wait_ms=40,
            enable_cache=False,
        )
        payloads = {
            "good": ([0.1, 0.2], 24000),
            "bad": ([-0.1, 0.2], 24000),
        }
        try:
            good, bad = await asyncio.gather(
                _encode(encoder, "good", payloads),
                _encode(encoder, "bad", payloads),
                return_exceptions=True,
            )
        finally:
            encoder.close()

        assert isinstance(good, torch.Tensor)
        assert isinstance(bad, BaseException)
        assert [len(call) for call in processor.calls] == [2, 1, 1]
        return good, bad

    good, bad = asyncio.run(_run())
    assert good.shape == (2, 2)
    assert "bad waveform" in str(bad)


def test_lru_evicts_by_item_count() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor()
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            max_batch_wait_ms=0,
            cache_max_items=1,
            enable_cache=True,
        )
        payloads = {
            "a": ([0.1, 0.2], 24000),
            "b": ([0.3, 0.4], 24000),
        }
        try:
            await _encode(encoder, "a", payloads)
            await _encode(encoder, "b", payloads)
            await _encode(encoder, "a", payloads)
        finally:
            encoder.close()

        assert len(processor.calls) == 3
        stats = encoder.stats()
        assert stats["misses"] == 3
        assert stats["entries"] == 1

    asyncio.run(_run())


def test_lru_evicts_by_byte_budget_and_skips_oversized_entry() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor()
        entry_bytes = 2 * 2 * 4
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            max_batch_wait_ms=0,
            cache_max_items=10,
            cache_max_bytes=entry_bytes,
            enable_cache=True,
        )
        payloads = {
            "a": ([0.1, 0.2], 24000),
            "b": ([0.3, 0.4], 24000),
            "big": ([0.5, 0.6, 0.7], 24000),
        }
        try:
            await _encode(encoder, "a", payloads)
            await _encode(encoder, "b", payloads)
            await _encode(encoder, "b", payloads)
            await _encode(encoder, "a", payloads)
            await _encode(encoder, "big", payloads)
            await _encode(encoder, "big", payloads)
        finally:
            encoder.close()

        assert len(processor.calls) == 5
        stats = encoder.stats()
        assert stats["hits"] == 1
        assert stats["entries"] == 1

    asyncio.run(_run())


def test_nonpositive_cache_capacity_fails_fast() -> None:
    processor = _FakeMossProcessor()
    with pytest.raises(ValueError, match="max_items"):
        create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            cache_max_items=0,
        )
    with pytest.raises(ValueError, match="max_bytes"):
        create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            cache_max_bytes=0,
        )


def test_single_flight_failure_does_not_poison_cache() -> None:
    async def _run() -> None:
        class _FlakyProcessor(_FakeMossProcessor):
            def __init__(self) -> None:
                super().__init__(delay_s=0.05)
                self.failures_left = 2

            def encode_audios_from_wav(
                self,
                waveforms: list[torch.Tensor],
                *,
                sampling_rate: int,
                n_vq: int,
            ) -> list[torch.Tensor]:
                if self.failures_left > 0:
                    self.failures_left -= 1
                    self.calls.append([wav.detach().clone() for wav in waveforms])
                    raise RuntimeError("transient encode failure")
                return super().encode_audios_from_wav(
                    waveforms,
                    sampling_rate=sampling_rate,
                    n_vq=n_vq,
                )

        processor = _FlakyProcessor()
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            max_batch_wait_ms=10,
            enable_cache=True,
        )
        payloads = {"same": ([0.1, 0.2], 24000)}
        try:
            first = await asyncio.gather(
                *(_encode(encoder, "same", payloads) for _ in range(2)),
                return_exceptions=True,
            )
            second = await _encode(encoder, "same", payloads)
        finally:
            encoder.close()

        assert all(isinstance(item, BaseException) for item in first)
        assert isinstance(second, torch.Tensor)
        stats = encoder.stats()
        assert stats["entries"] == 1
        assert stats["inflight"] == 0

    asyncio.run(_run())


def test_cache_hit_skips_resolve() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor()
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=4,
            sr_target=24000,
            max_batch_wait_ms=0,
            enable_cache=True,
        )

        resolve_calls = 0

        async def _resolve(ref: str) -> tuple[list[float], int]:
            nonlocal resolve_calls
            resolve_calls += 1
            await asyncio.sleep(0)
            return [0.1, 0.2, 0.3], 24000

        try:
            first = await encoder.encode_reference_codes(
                "spk-1", resolve_ref_audio=_resolve
            )
            second = await encoder.encode_reference_codes(
                "spk-1", resolve_ref_audio=_resolve
            )
        finally:
            encoder.close()

        # The locator-keyed cache must serve the second call without resolving
        # or re-encoding the reference.
        assert resolve_calls == 1
        assert len(processor.calls) == 1
        assert torch.equal(first, second)
        assert encoder.stats()["hits"] == 1

    asyncio.run(_run())


def test_distinct_references_encode_in_parallel() -> None:
    async def _run() -> int:
        processor = _FakeMossProcessor(delay_s=0.1)
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            max_batch_wait_ms=0,
            num_workers=4,
            enable_cache=False,
        )
        payloads = {
            "a": ([0.1, 0.2], 24000),
            "b": ([0.3, 0.4], 24000),
            "c": ([0.5, 0.6], 24000),
            "d": ([0.7, 0.8], 24000),
        }
        try:
            await asyncio.gather(
                *(_encode(encoder, ref, payloads) for ref in payloads)
            )
        finally:
            encoder.close()
        return processor.max_active

    max_active = asyncio.run(_run())
    # A single serial worker would cap observed concurrency at 1; the pool must
    # run distinct cold encodes in parallel.
    assert max_active >= 2


def test_reference_encoder_rejects_long_reference_before_codec() -> None:
    async def _run() -> None:
        processor = _FakeMossProcessor()
        encoder = create_reference_encoder(
            processor,
            variant="tts",
            n_vq=2,
            sr_target=24000,
            enable_cache=True,
        )
        payloads = {"long": ([0.0] * 101, 1)}
        try:
            with pytest.raises(ValueError, match="100"):
                await _encode(encoder, "long", payloads)
        finally:
            encoder.close()

        assert not processor.calls
        assert encoder.stats()["entries"] == 0

    asyncio.run(_run())
