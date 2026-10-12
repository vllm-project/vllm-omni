# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""API processes share one reference encoder through a Unix socket."""

import threading
from tempfile import TemporaryDirectory

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import shared_reference_encoder as shared

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _codes(wav):
    # Deterministic per clip: frames from the clip length, values from its first sample.
    return torch.full((wav.shape[-1] // 10, 4), int(wav.reshape(-1)[0]), dtype=torch.long)


class _Recorder:
    def __init__(self, gate=None):
        self.batches = []
        self.gate = gate

    def __call__(self, wavs):
        if self.gate is not None:
            self.gate.wait(5)
        self.batches.append(len(wavs))
        return [RuntimeError("bad clip") if int(w.reshape(-1)[0]) < 0 else _codes(w) for w in wavs]


@pytest.fixture
def host_dir(monkeypatch):
    monkeypatch.setattr(shared, "_host_lock_handle", None)
    # AF_UNIX limits the full socket path, including pytest directory names.
    with TemporaryDirectory(prefix="moss-ref-") as path:
        yield path


def test_only_one_process_per_directory_hosts(host_dir):
    assert shared.elect_host(host_dir) is True
    assert shared.elect_host(host_dir) is False  # a second open file description cannot take the lock


def test_close_releases_queued_and_active_callers(host_dir):
    entered, release = threading.Event(), threading.Event()

    def encode(wavs):
        entered.set()
        assert release.wait(5)
        return [_codes(wav) for wav in wavs]

    host = shared.SharedReferenceEncoderHost(host_dir, [(encode, None)])
    results: dict[str, list[torch.Tensor | BaseException]] = {}
    threads = [
        threading.Thread(target=lambda key=key: results.setdefault(key, host.encode([torch.ones(1, 20)])), daemon=True)
        for key in ("active", "queued")
    ]
    try:
        threads[0].start()
        assert entered.wait(2)
        threads[1].start()
        with host._ready:
            assert host._ready.wait_for(lambda: len(host._pending) == 1, timeout=2)
        host.close()
        for thread in threads:
            thread.join(1)
            assert not thread.is_alive(), "closed encoder left a request waiting"
        assert all(isinstance(results[key][0], RuntimeError) for key in results)
    finally:
        release.set()
        host.close()
        for thread in threads:
            if thread.ident is not None:
                thread.join(1)


def test_worker_result_count_mismatch_fails_entire_batch(host_dir):
    host = shared.SharedReferenceEncoderHost(host_dir, [(lambda wavs: [_codes(wavs[0])], None)])
    results = []
    caller = threading.Thread(target=lambda: results.extend(host.encode([torch.ones(1, 20)] * 2)), daemon=True)
    try:
        caller.start()
        caller.join(2)
        assert not caller.is_alive(), "missing encoder result left a request waiting"
        assert len(results) == 2
        assert all(isinstance(result, RuntimeError) for result in results)
    finally:
        host.close()
        caller.join(1)


def test_client_round_trip_and_per_clip_errors(host_dir):
    host = shared.SharedReferenceEncoderHost(host_dir, [(_Recorder(), None)])
    try:
        client = shared.SharedReferenceEncoderClient(host_dir)
        wavs = [torch.full((1, 40), 3.0), torch.full((1, 70), -1.0), torch.full((1, 20), 5.0)]
        out = client.encode(wavs)
        assert out[0].dtype == torch.long and torch.equal(out[0], _codes(wavs[0]))
        assert isinstance(out[1], RuntimeError) and "bad clip" in str(out[1])
        assert torch.equal(out[2], _codes(wavs[2]))
        # The host process's own clips skip the socket.
        assert torch.equal(host.encode([wavs[2]])[0], _codes(wavs[2]))
    finally:
        host.close()


def test_host_merges_clips_from_concurrent_clients(host_dir):
    gate = threading.Event()
    recorder = _Recorder(gate)
    host = shared.SharedReferenceEncoderHost(host_dir, [(recorder, None)])
    try:
        results = {}

        def send(name, value):
            results[name] = shared.SharedReferenceEncoderClient(host_dir).encode([torch.full((1, 30), value)] * 2)

        # Encoding is held until both clients' clips are queued.
        warm = threading.Thread(target=lambda: host.encode([torch.full((1, 10), 1.0)]))
        warm.start()
        threads = [threading.Thread(target=send, args=(n, v)) for n, v in (("a", 2.0), ("b", 4.0))]
        for thread in threads:
            thread.start()
        threading.Timer(0.5, gate.set).start()
        for thread in (warm, *threads):
            thread.join(10)
        # Both clients' clips were encoded together, not one request at a time.
        assert sum(recorder.batches) == 5 and max(recorder.batches) >= 4
        assert all(int(c[0, 0]) == 2 for c in results["a"]) and all(int(c[0, 0]) == 4 for c in results["b"])
    finally:
        host.close()


def test_client_without_host_raises(host_dir, monkeypatch):
    monkeypatch.setattr(shared, "_CONNECT_TIMEOUT_S", 0.3)
    with pytest.raises((FileNotFoundError, ConnectionRefusedError)):
        shared.SharedReferenceEncoderClient(host_dir).encode([torch.zeros(1, 10)])


def test_host_window_collects_across_workers_without_reserving_singletons(host_dir):
    recorder = _Recorder()
    host = shared.SharedReferenceEncoderHost(host_dir, [(recorder, None)] * 3, batch_window_ms=500)
    try:
        barrier = threading.Barrier(3)
        results = []

        def send(value):
            barrier.wait()
            results.extend(host.encode([torch.full((1, 20), value)]))

        threads = [threading.Thread(target=send, args=(v,)) for v in (1.0, 2.0)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
        assert recorder.batches == [2]
        assert sorted(int(x[0, 0]) for x in results) == [1, 2]
    finally:
        host.close()


def test_reference_encoder_client_falls_back_to_local_encode(host_dir, monkeypatch):
    from tests.model_executor.models.moss_tts.test_reference_encoder import _N_VQ, _SR, _FakeProcessor
    from vllm_omni.model_executor.models.moss_tts import reference_encoder as re_mod
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    monkeypatch.setenv(re_mod._SHARED_CODES_DIR_ENV, host_dir)
    monkeypatch.setattr(re_mod, "shared_encoder_role", lambda: "client")
    monkeypatch.setattr(shared, "_CONNECT_TIMEOUT_S", 0.3)
    proc = _FakeProcessor()
    enc = re_mod.MossReferenceEncoder(
        proc, variant="local", n_vq=_N_VQ, sr_target=_SR, speaker_cache=SpeakerEmbeddingCache(max_bytes=1 << 20)
    )
    placed = []
    monkeypatch.setattr(enc, "_place_tokenizer_for_local_encode", lambda: placed.append(True))
    out = enc._encode_batch_sync([([7.0] * 50, _SR)])
    assert placed == [True] and proc.attempt_sizes == [1] and int(out[0][0, 0]) == 7


def test_workers_encode_concurrently_after_their_warmups(host_dir):
    inside, release = threading.Barrier(3, timeout=5), threading.Event()
    warmed = []

    def worker(name):
        def encode(wavs):
            inside.wait()  # both workers are inside an encode at the same time
            release.wait(5)
            return [_codes(w) for w in wavs]

        return encode, lambda: warmed.append(name)

    host = shared.SharedReferenceEncoderHost(host_dir, [worker("a"), worker("b")])
    try:
        out = []

        def submit(value):
            out.append(host.encode([torch.full((1, 20), value)]))

        first = threading.Thread(target=submit, args=(1.0,))
        first.start()
        # The first worker is busy by now, so the second clip goes to the other one.
        threading.Event().wait(0.1)
        second = threading.Thread(target=submit, args=(2.0,))
        second.start()
        inside.wait()
        release.set()
        for thread in (first, second):
            thread.join(5)
        assert sorted(warmed) == ["a", "b"] and sorted(int(o[0][0, 0]) for o in out) == [1, 2]
    finally:
        host.close()


def test_client_requests_in_flight_use_their_own_connections(host_dir):
    entered, release = threading.Semaphore(0), threading.Event()

    def worker():
        def encode(wavs):
            entered.release()
            release.wait(5)
            return [_codes(w) for w in wavs]

        return encode, None

    host = shared.SharedReferenceEncoderHost(host_dir, [worker(), worker()])
    try:
        client = shared.SharedReferenceEncoderClient(host_dir)
        out = []
        threads = [
            threading.Thread(target=lambda v=v: out.append(client.encode([torch.full((1, 20), v)]))) for v in (1.0, 2.0)
        ]
        for thread in threads:
            thread.start()
            # A worker holds this request, still unanswered, before the next is
            # sent. Sent together, one free worker may take both as one batch.
            assert entered.acquire(timeout=5)
        release.set()  # both requests reached the host at once
        for thread in threads:
            thread.join(5)
        assert sorted(int(o[0][0, 0]) for o in out) == [1, 2]
        assert len(client._idle) == 2  # both connections are kept for reuse
    finally:
        host.close()


def test_readiness_waits_for_every_worker_warmup(host_dir, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    recorder = _Recorder()
    warmed = []

    def last_warmup():
        entered.set()
        assert release.wait(5)
        warmed.append("last")

    host = shared.SharedReferenceEncoderHost(
        host_dir, [(recorder, lambda: warmed.append("first")), (recorder, last_warmup)]
    )
    client = shared.SharedReferenceEncoderClient(host_dir)
    try:
        assert entered.wait(5)
        monkeypatch.setattr(shared, "_STARTUP_TIMEOUT_S", 0.03)
        with pytest.raises(shared.SharedReferenceEncoderStartupError):
            host.wait_until_ready()
        with pytest.raises(shared.SharedReferenceEncoderStartupError):
            client.wait_until_ready()
        assert warmed == ["first"] and recorder.batches == [] and not client._idle
        release.set()
        monkeypatch.setattr(shared, "_STARTUP_TIMEOUT_S", 5)
        host.wait_until_ready()
        client.wait_until_ready()
        assert warmed == ["first", "last"] and recorder.batches == []
        assert len(client._idle) == 1
        assert torch.equal(client.encode([torch.ones(1, 20)])[0], _codes(torch.ones(1, 20)))
    finally:
        release.set()
        host.close()


def test_slow_startup_does_not_use_encode_timeout(host_dir, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(shared, "_REQUEST_TIMEOUT_S", 0.05)
    monkeypatch.setattr(shared, "_STARTUP_TIMEOUT_S", 5, raising=False)
    host = shared.SharedReferenceEncoderHost(host_dir, [(_Recorder(), lambda: release.wait(5))])
    timer = threading.Timer(0.2, release.set)
    timer.start()
    try:
        # The first socket waits for readiness before starting its much shorter
        # encode deadline. No API warmup call is required to protect this path.
        client = shared.SharedReferenceEncoderClient(host_dir)
        assert torch.equal(client.encode([torch.ones(1, 20)])[0], _codes(torch.ones(1, 20)))
    finally:
        release.set()
        timer.join(5)
        host.close()


def test_failed_host_warmup_is_not_ready(host_dir):
    def fail():
        raise RuntimeError("warmup failed")

    recorder = _Recorder()
    host = shared.SharedReferenceEncoderHost(host_dir, [(recorder, fail)])
    try:
        with pytest.raises(shared.SharedReferenceEncoderStartupError):
            host.wait_until_ready()
        with pytest.raises(shared.SharedReferenceEncoderStartupError):
            shared.SharedReferenceEncoderClient(host_dir).wait_until_ready()
        with pytest.raises(shared.SharedReferenceEncoderStartupError):
            host.encode([torch.ones(1, 20)])
        assert recorder.batches == []
    finally:
        host.close()


def test_startup_failure_does_not_trigger_local_capture(host_dir, monkeypatch):
    from unittest.mock import Mock

    from tests.model_executor.models.moss_tts.test_reference_encoder import _N_VQ, _SR, _FakeProcessor
    from vllm_omni.model_executor.models.moss_tts import reference_encoder as re_mod
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    monkeypatch.setenv(re_mod._SHARED_CODES_DIR_ENV, host_dir)
    monkeypatch.setattr(re_mod, "shared_encoder_role", lambda: "client")
    enc = re_mod.MossReferenceEncoder(
        _FakeProcessor(),
        variant="local",
        n_vq=_N_VQ,
        sr_target=_SR,
        speaker_cache=SpeakerEmbeddingCache(max_bytes=1 << 20),
    )
    enc._shared_client = Mock()
    enc._shared_client.encode.side_effect = shared.SharedReferenceEncoderStartupError("still compiling")
    local = Mock()
    monkeypatch.setattr(enc, "_place_tokenizer_for_local_encode", local)
    with pytest.raises(shared.SharedReferenceEncoderStartupError):
        enc._encode_shared_or_local([torch.ones(1, 20)])
    local.assert_not_called()
