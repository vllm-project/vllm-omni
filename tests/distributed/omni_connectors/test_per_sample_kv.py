# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-sample KV handoff for SamplingParams.n > 1: one transfer per sample, all of them to the diffusion request."""

import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors import kv_transfer_manager
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVCacheConfig, OmniKVTransferManager
from vllm_omni.distributed.omni_connectors.utils.kv_utils import PER_SAMPLE_KV_COUNT_KEY
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _receiver(monkeypatch, available: set[str]) -> OmniKVTransferManager:
    manager = OmniKVTransferManager(
        OmniKVCacheConfig(connector_config={"type": "mock"}, stage_id=1, engine_input_source=[0], need_recv_cache=True)
    )

    def fake_receive(request_id, target_device=None, **kwargs):
        if request_id not in available:
            return None, 0
        layer = torch.full((2, 1, 4), float(sorted(available).index(request_id)))
        data = {"layer_blocks": {"key_cache": [layer], "value_cache": [layer]}, "metadata": {"key": request_id}}
        return data, 1

    monkeypatch.setattr(manager, "receive_kv_cache_for_request", fake_receive)
    return manager


def _request(num_samples: int) -> OmniDiffusionRequest:
    params = OmniDiffusionSamplingParams(extra_args={PER_SAMPLE_KV_COUNT_KEY: num_samples})
    return OmniDiffusionRequest(prompt="", sampling_params=params, request_id="req")


def test_diffusion_request_receives_every_sample(monkeypatch):
    manager = _receiver(monkeypatch, {"req-sample0", "req-sample1", "req-sample2"})
    req = _request(3)

    assert manager.receive_multi_kv_cache(req)
    caches = req.sampling_params.past_key_values
    assert [cache.key_cache[0][0, 0, 0].item() for cache in caches] == [0.0, 1.0, 2.0]
    assert [metadata["key"] for metadata in req.sampling_params.kv_metadata] == [f"req-sample{i}" for i in range(3)]


def test_missing_sample_fails_the_receive(monkeypatch):
    manager = _receiver(monkeypatch, {"req-sample0", "req-sample2"})
    req = _request(3)

    received: list[str] = []
    receive = manager.receive_kv_cache_for_request
    manager.receive_kv_cache_for_request = lambda request_id, *args, **kwargs: (
        received.append(request_id) or receive(request_id, *args, **kwargs)
    )

    assert not manager.receive_multi_kv_cache(req)
    assert req.sampling_params.past_key_values is None
    assert received == ["req-sample0", "req-sample1", "req-sample2"]  # sample 2 is still received


def test_without_a_sample_count_the_request_receives_one_kv(monkeypatch):
    # Pipelines like Bagel set num_outputs_per_prompt > 1 with a single KV.
    manager = _receiver(monkeypatch, {"req", "req-sample0"})
    req = OmniDiffusionRequest(
        prompt="", sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=4), request_id="req"
    )

    assert manager.receive_multi_kv_cache(req)
    assert req.sampling_params.kv_metadata == {"key": "req"}


def test_samples_share_the_cfg_companion_kv(monkeypatch):
    # A CFG pipeline: each sample has its own KV, one negative-prompt companion serves all of them.
    manager = _receiver(monkeypatch, {"req-sample0", "req-sample1", "req__cfg_text"})
    req = _request(2)
    req.sampling_params.cfg_kv_request_ids = {"cfg_text": "req__cfg_text"}

    def collect(request_id, cfg_request_ids, kv_manager, target_device):
        data, _ = kv_manager.receive_kv_cache_for_request(cfg_request_ids["cfg_text"], target_device)
        return {"cfg_text_kv_metadata": data["metadata"]}

    assert manager.receive_multi_kv_cache(req, cfg_kv_collect_func=collect)
    assert [metadata["key"] for metadata in req.sampling_params.kv_metadata] == ["req-sample0", "req-sample1"]
    assert req.sampling_params.cfg_text_kv_metadata == {"key": "req__cfg_text"}


def test_missing_samples_share_one_receive_deadline(monkeypatch):
    # Each sample waits only for what is left of one recv_timeout, so K missing samples cost one timeout, not K.
    manager = _receiver(monkeypatch, set())
    clock = [1000.0]
    timeouts: list[float] = []

    def slow_miss(request_id, target_device=None, *, timeout=None, **kwargs):
        timeouts.append(timeout)
        clock[0] += 20.0  # each miss waits 20 s
        return None, 0

    monkeypatch.setattr(kv_transfer_manager.time, "time", lambda: clock[0])
    monkeypatch.setattr(manager, "receive_kv_cache_for_request", slow_miss)

    assert not manager.receive_multi_kv_cache(_request(3))
    assert timeouts == [manager.config.recv_timeout, manager.config.recv_timeout - 20.0, 0.0]
