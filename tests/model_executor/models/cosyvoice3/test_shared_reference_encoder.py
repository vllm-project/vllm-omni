# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Artifact ownership and real process-boundary tests, without model weights."""

import asyncio
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.api_server_shared_dir import create_api_server_shared_dir, remove_api_server_shared_dir
from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import (
    COSY_REFERENCE_WIRE,
    CosyReferenceArtifacts,
    CosyVoice3ReferenceEncoder,
)
from vllm_omni.utils.reference_encoding import content_key
from vllm_omni.utils.shared_reference_encoder import ReferenceEncoderPool, ReferenceEncoderSpec
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _artifact(value=1):
    return CosyReferenceArtifacts(
        torch.full((1, 4, 80), float(value)),
        torch.tensor([[value, value + 1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.full((1, 192), float(value)),
    )


def test_wire_round_trip_preserves_shapes_dtypes_and_ownership():
    original = _artifact()
    result = COSY_REFERENCE_WIRE.artifact_from_arrays(COSY_REFERENCE_WIRE.artifact_to_arrays(original))
    for before, after in zip(original.tensors(), result.tensors(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert before.dtype == after.dtype
        assert before.data_ptr() != after.data_ptr()
    request = result.processor_kwargs()
    request["speech_feat"].zero_()
    assert torch.all(result.speech_feat == 1)
    assert original.nbytes == sum(t.numel() * t.element_size() for t in original.tensors())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speech_feat", torch.ones(4, 80)),
        ("speech_token", torch.ones(1, 2)),
        ("speech_token_len", torch.tensor([9], dtype=torch.int32)),
        ("embedding", torch.ones(192)),
        ("speech_feat", None),
    ],
)
def test_invalid_precomputed_artifacts_are_rejected(field, value):
    data = _artifact().processor_kwargs()
    data[field] = value
    with pytest.raises(ValueError, match="Cosy reference"):
        CosyReferenceArtifacts.from_mapping(data)


class _CountingEncoder(CosyVoice3ReferenceEncoder):
    """Use the real batch deduplication with cheap, bounded fake conditioning."""

    def __init__(self, record_path, cache_bytes):
        self.model_path = "test-model"
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._cache = SpeakerEmbeddingCache(max_bytes=cache_bytes)
        self.calls = 0
        self.record_path = record_path
        with open(record_path, "a") as log:
            log.write(f"load {os.getpid()}\n")

    def _encode_one(self, clip):
        wav, sr = clip
        if sr <= 0:
            raise ValueError("invalid sample rate")
        key = content_key(self.model_path, wav, sr)
        hit = self._cache.get(key)
        if hit is not None:
            return CosyReferenceArtifacts.from_mapping(hit)
        self.calls += 1
        with open(self.record_path, "a") as log:
            log.write(f"encode {os.getpid()} {self.calls}\n")
        result = _artifact(int(wav[0]))
        result.embedding[0, :2] = torch.tensor([os.getpid(), self.calls], dtype=torch.float32)
        self._cache.put(key, result.processor_kwargs())
        return result


def test_batch_deduplicates_content_and_isolates_errors(tmp_path):
    encoder = _CountingEncoder(str(tmp_path / "calls"), 4096)
    wave = np.ones(16, dtype=np.float32)
    try:
        first, duplicate, error = encoder.encode_batch([(wave, 16000), (wave.copy(), 16000), (wave, 0)])
        assert encoder.calls == 1
        assert first is duplicate
        assert isinstance(error, ValueError)
        assert encoder.encode_batch([(wave.copy(), 16000)])[0].speech_token.tolist() == [[1, 2]]
        assert encoder.calls == 1
        encoder.encode_batch([(wave * 2, 16000)])
        encoder.encode_batch([(wave, 24000)])
        assert encoder.calls == 3
    finally:
        encoder._executor.shutdown()


def test_worker_cache_is_byte_bounded_and_eviction_reencodes(tmp_path):
    encoder = _CountingEncoder(str(tmp_path / "calls"), _artifact().nbytes)
    one, two = np.ones(16, dtype=np.float32), np.full(16, 2, dtype=np.float32)
    try:
        for wave in (one, one.copy(), two, one):
            assert not isinstance(encoder.encode_batch([(wave, 16000)])[0], Exception)
        assert encoder.calls == 3
        assert encoder._cache.memory_bytes() <= _artifact().nbytes
    finally:
        encoder._executor.shutdown()


def _api_process(shared_dir, record_path, barrier, release, output):
    async def run():
        module = __name__
        spec = ReferenceEncoderSpec(
            name="cosy-test",
            factory=f"{module}:_CountingEncoder",
            kwargs={"record_path": record_path, "cache_bytes": 4096},
            wire="vllm_omni.model_executor.models.cosyvoice3.reference_encoder:COSY_REFERENCE_WIRE",
            namespace="test-model",
            nbytes=lambda artifact: artifact.nbytes,
            workers=1,
            cache_bytes=0,
        )
        pool = ReferenceEncoderPool(spec, shared_dir)
        try:
            await pool.start()
            await asyncio.to_thread(barrier.wait, 60)
            wave = np.ones(16, dtype=np.float32)
            first, concurrent = await asyncio.gather(pool.get(wave, 16000), pool.get(wave.copy(), 16000))
            later = await pool.get(wave, 16000)
            assert torch.equal(first.speech_token, concurrent.speech_token)
            assert torch.equal(first.speech_token, later.speech_token)
            output.put((os.getpid(), first.embedding[0, :2].tolist(), type(pool.backend).__name__))
            await asyncio.to_thread(release.wait, 60)
        finally:
            if pool._service is not None:
                await pool._service.aclose()
            pool.close()

    try:
        asyncio.run(run())
    except BaseException as error:
        output.put(("error", repr(error)))
        raise


@pytest.mark.parametrize("same_server", [True, False])
def test_two_spawned_api_processes_share_only_within_one_server(tmp_path, same_server):
    context = multiprocessing.get_context("spawn")
    barrier, release, output = context.Barrier(2), context.Event(), context.Queue()
    shared_dirs = [create_api_server_shared_dir(), create_api_server_shared_dir()]
    record = str(tmp_path / "encodes.log")
    processes = [
        context.Process(
            target=_api_process,
            args=(shared_dirs[0 if same_server else i], record, barrier, release, output),
        )
        for i in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=90) for _ in processes]
        assert all(row[0] != "error" for row in results), results
        assert len({row[0] for row in results}) == 2
        expected = 1 if same_server else 2
        assert len({row[1][0] for row in results}) == expected
        assert all(row[1][1] == 1 for row in results), "cache was not shared across API requests"
        lines = Path(record).read_text().splitlines()
        assert sum(line.startswith("load ") for line in lines) == expected
        assert sum(line.startswith("encode ") for line in lines) == expected
        if same_server:
            assert {row[2] for row in results} == {"SharedReferenceEncoderHost", "SharedReferenceEncoderClient"}
    finally:
        release.set()
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for path in shared_dirs:
            remove_api_server_shared_dir(path)
    assert all(process.exitcode == 0 for process in processes)
