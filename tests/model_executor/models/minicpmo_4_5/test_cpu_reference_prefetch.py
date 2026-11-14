# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cpu_prefetch_ownership_cancellation_and_failure(mocker):
    main = threading.get_ident()
    cpu_calls = []
    gpu_calls = []

    def cpu(path):
        assert threading.get_ident() != main
        cpu_calls.append(path)
        return torch.tensor([int(path)], dtype=torch.float32)

    def finish(path, *, cpu_features=None):
        assert threading.get_ident() == main
        gpu_calls.append(path)
        x = cpu_features if cpu_features is not None else torch.tensor([int(path)])
        return x, None, x, x, None

    backend = object.__new__(BatchedToken2Wav)
    backend._prompt_features = {}
    backend._setup_cache = {}
    backend.speech_window = torch.zeros(1)
    backend._token2wav = mocker.Mock(_prepare_prompt_cpu=cpu, _prepare_prompt=finish)
    assert backend.prefetch_prompt_cpu("a", "1", 4)
    assert not backend.prefetch_prompt_cpu("a", "1", 4)
    backend.prepare_prompt("a", "1")
    assert cpu_calls == ["1"] and gpu_calls == ["1"]
    assert not backend.prefetch_prompt_cpu("a", "1", 4)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        assert not backend.prefetch_prompt_cpu("b", "2", 4)
        backend.prepare_prompt("b", "2")
        assert torch.get_default_dtype() == torch.bfloat16
    finally:
        torch.set_default_dtype(original)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked(path):
        started.set()
        assert release.wait(10)
        finished.set()
        return torch.ones(1)

    backend._token2wav._prepare_prompt_cpu = blocked
    assert backend.prefetch_prompt_cpu("c", "3", 4)
    assert started.wait(10)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    backend.evict_prompt("c", "3")
    assert finished.is_set() and not backend._cpu_prompt_futures
    timer.join()

    def fail(path):
        raise ValueError("expected feature failure")

    backend._token2wav._prepare_prompt_cpu = fail
    assert backend.prefetch_prompt_cpu("d", "4", 4)
    try:
        backend.prepare_prompt("d", "4")
    except ValueError as exc:
        assert str(exc) == "expected feature failure"
    else:
        raise AssertionError("failure swallowed")
    assert torch.get_default_dtype() == torch.float32
    backend._cpu_prompt_pool.shutdown(wait=True)
    # A queued cancellation must not read its WAV; all outstanding work is bounded.
    queued = object.__new__(BatchedToken2Wav)
    queued._prompt_features = {}
    queued._setup_cache = {}
    queued.speech_window = torch.zeros(1)
    started.clear()
    release.clear()
    finished.clear()
    reads = []

    def blocked_once(path):
        reads.append(path)
        started.set()
        assert release.wait(10)
        return torch.ones(1)

    queued._token2wav = mocker.Mock(_prepare_prompt_cpu=blocked_once, _prepare_prompt=finish)
    for i in range(32):
        assert queued.prefetch_prompt_cpu(str(i), str(i), 1)
    assert started.wait(10)
    assert not queued.prefetch_prompt_cpu("overflow", "overflow", 1)
    queued.evict_prompt("31", "31")
    for i in range(1, 31):
        queued.evict_prompt(str(i), str(i))
    release.set()
    queued.evict_prompt("0", "0")
    queued._cpu_prompt_pool.shutdown(wait=True)
    assert reads == ["0"] and not queued._cpu_prompt_futures
