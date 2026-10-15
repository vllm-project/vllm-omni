# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Process-backed reference encoder workers and model wire formats of the shared encoder."""

import os
import threading
import time

import numpy as np
import pytest

from vllm_omni.entrypoints.api_server_shared_dir import create_api_server_shared_dir, remove_api_server_shared_dir
from vllm_omni.utils import shared_reference_encoder as shared

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "tests.utils.test_shared_reference_encoder"


class FakeEncoder:
    """Built in the worker process: artifact = (scale * sum of the clip, sample rate, worker pid)."""

    def __init__(self, scale: float, delay_s: float = 0.0, fail: bool = False) -> None:
        if fail:
            raise ValueError("no weights")
        self.scale, self.delay_s = scale, delay_s

    def encode_batch(self, clips):
        for wav, sr in clips:
            time.sleep(self.delay_s)
            yield ValueError("bad clip") if wav[0] < 0 else (float(wav.sum()) * self.scale, sr, os.getpid())


FAKE_WIRE = shared.WireFormat(
    clip_to_array=lambda clip: (clip[0], clip[1]),
    clip_from_array=lambda array, sr: (array.copy(), sr),
    artifact_to_arrays=lambda artifact: [np.array(artifact, dtype=np.float64)],
    artifact_from_arrays=lambda arrays: (float(arrays[0][0]), int(arrays[0][1]), int(arrays[0][2])),
)


def _worker(**kwargs) -> shared.ProcessEncoderWorker:
    return shared.ProcessEncoderWorker(f"{_MODULE}:FakeEncoder", kwargs, wire=f"{_MODULE}:FAKE_WIRE")


@pytest.fixture
def host_dir(monkeypatch):
    monkeypatch.setattr(shared, "_host_lock_handles", {})
    path = create_api_server_shared_dir()
    yield path
    remove_api_server_shared_dir(path)


def test_process_worker_round_trip_with_per_clip_errors() -> None:
    worker = _worker(scale=2.0)
    try:
        worker.wait_until_ready()
        clips = [(np.ones(4, dtype=np.float32), 24000), (-np.ones(3, dtype=np.float32), 16000)]
        good, bad = worker(clips)
        assert good[:2] == (8.0, 24000) and good[2] != os.getpid()
        assert isinstance(bad, RuntimeError) and "bad clip" in str(bad)
    finally:
        worker.close()


def test_host_encodes_on_process_workers_in_parallel() -> None:
    workers = [_worker(scale=1.0, delay_s=0.3) for _ in range(2)]
    host = shared.SharedReferenceEncoderHost(
        None, [(w, w.wait_until_ready) for w in workers], wire=FAKE_WIRE, worker_streams=False
    )
    try:
        host.wait_until_ready()
        results: list = []

        def submit():
            results.extend(host.encode([(np.ones(2, dtype=np.float32), 24000)]))

        threads = [threading.Thread(target=submit) for _ in range(2)]
        start = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # Two 0.3 s encodes in parallel, not one after the other.
        assert time.monotonic() - start < 0.55
        assert len({pid for _, _, pid in results}) == 2
    finally:
        host.close()
        for worker in workers:
            worker.close()


def test_client_round_trip_through_the_host_with_a_model_wire_format(host_dir) -> None:
    assert shared.elect_host(host_dir)
    worker = _worker(scale=3.0)
    host = shared.SharedReferenceEncoderHost(
        host_dir, [(worker, worker.wait_until_ready)], wire=FAKE_WIRE, worker_streams=False
    )
    try:
        client = shared.SharedReferenceEncoderClient(host_dir, wire=FAKE_WIRE)
        client.wait_until_ready()
        good, bad = client.encode([(np.full(3, 2.0, dtype=np.float32), 22050), (-np.ones(1, dtype=np.float32), 8000)])
        assert good[:2] == (18.0, 22050)
        assert isinstance(bad, RuntimeError)
    finally:
        host.close()
        worker.close()


def test_failed_worker_process_startup_is_not_ready() -> None:
    worker = _worker(scale=1.0, fail=True)
    host = shared.SharedReferenceEncoderHost(
        None, [(worker, worker.wait_until_ready)], wire=FAKE_WIRE, worker_streams=False
    )
    try:
        with pytest.raises(shared.SharedReferenceEncoderStartupError):
            host.wait_until_ready()
    finally:
        host.close()
        worker.close()


def _spec(workers: int) -> shared.ReferenceEncoderSpec:
    return shared.ReferenceEncoderSpec(
        name="fake-ref",
        factory=f"{_MODULE}:FakeEncoder",
        kwargs={"scale": 2.0},
        wire=f"{_MODULE}:FAKE_WIRE",
        namespace="fake-model",
        nbytes=lambda artifact: 24,
        workers=workers,
    )


@pytest.mark.asyncio
async def test_pool_host_serves_its_clients_and_caches_by_content(host_dir) -> None:
    host = shared.ReferenceEncoderPool(_spec(workers=1), host_dir)
    # A second API process: the host lock is taken, so it sends its clips to the host.
    client = shared.ReferenceEncoderPool(_spec(workers=1), host_dir)
    try:
        assert isinstance(host.backend, shared.SharedReferenceEncoderHost)
        assert isinstance(client.backend, shared.SharedReferenceEncoderClient)
        await host.start()
        await client.start()
        wav = np.full(4, 0.5, dtype=np.float32)
        first = await host.get(wav, 24000)
        assert first[:2] == (4.0, 24000) and first[2] != os.getpid()
        assert await host.get(wav.astype(np.float64), 24000) == first  # same samples: cached
        assert await client.get(wav, 24000) == first  # encoded by the host's process
        assert (await host.get(wav, 16000))[1] == 16000
        assert host._service is not None and host._service.stats()["hits"] == 1
    finally:
        client.close()
        host.close()


@pytest.mark.asyncio
async def test_pool_without_processes_encodes_in_this_process() -> None:
    pool = shared.ReferenceEncoderPool(_spec(workers=0), None)
    try:
        await pool.start()
        _, _, pid = await pool.get(np.ones(3, dtype=np.float32), 8000)
        assert pid == os.getpid()
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_adapter_encodes_references_through_its_spec() -> None:
    from types import SimpleNamespace

    from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, SpeechServingContext

    class NoReferences(ARTTSAdapter):
        name = "plain"

        async def build(self, request, sampling_params_list, has_inline_ref_audio):
            raise NotImplementedError

    class Adapter(NoReferences):
        name = "fake"

        def reference_encoder_spec(self):
            return _spec(workers=0)

    adapter = Adapter(SpeechServingContext(server=SimpleNamespace()))
    try:
        await adapter.warmup()
        assert (await adapter.encode_reference(np.ones(2, dtype=np.float32), 24000))[:2] == (4.0, 24000)
    finally:
        assert adapter._reference_pool is not None
        adapter._reference_pool.close()
    plain = NoReferences(SpeechServingContext(server=SimpleNamespace()))
    await plain.warmup()
    with pytest.raises(RuntimeError, match="does not encode reference audio"):
        await plain.encode_reference(np.ones(2, dtype=np.float32), 24000)
