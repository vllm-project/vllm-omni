"""Phase 3 background KV prefetch — CPU-testable mechanism.

Covers the opt-in gating, the start_load_kv/get_loaded_kv round trip over a mock
connector, dedup, per-call sender_info isolation, and the generalized
inflight-buffer drain.  The pinned H2D path (apply_prefetched_kv on GPU) needs
CUDA and is exercised by integration tests, not here.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MockConnector:
    def __init__(self):
        self.store = {}
        self.get_calls = []

    def put(self, from_stage, to_stage, put_key, data):
        self.store[f"{from_stage}->{to_stage}:{put_key}"] = data
        return True, len(str(data)), None

    def get(self, from_stage, to_stage, get_key, metadata=None):
        self.get_calls.append({"get_key": get_key, "metadata": metadata})
        key = f"{from_stage}->{to_stage}:{get_key}"
        if key in self.store:
            return self.store[key], len(str(self.store[key]))
        return None


def _make_sender_receiver(*, async_prefetch=True):
    """Build sender+receiver managers sharing one mock connector, seed one KV."""
    connector = MockConnector()

    sender = OmniKVTransferManager(
        OmniKVCacheConfig(
            connector_config={"type": "mock"}, from_stage="sender", to_stage="receiver", need_send_cache=True
        )
    )
    sender._connector = connector

    receiver = OmniKVTransferManager(
        OmniKVCacheConfig(
            connector_config={"type": "mock"},
            from_stage="sender",
            stage_id="receiver",
            need_recv_cache=True,
            recv_timeout=1.0,
        ),
        async_prefetch=async_prefetch,
    )
    receiver._connector = connector
    return sender, receiver, connector


def _seed_payload(sender, req_id, *, num_layers=2, block_size=4, num_heads=4, head_dim=8):
    kv_caches = [torch.randn(2, 5, block_size, num_heads, head_dim) for _ in range(num_layers)]
    sender.handle_finished_requests_kv_transfer(
        {req_id: {"block_ids": [0, 1], "seq_len": 10}}, kv_caches, block_size, "float32"
    )


# --------------------------------------------------------------------------- #
#  Opt-in gating
# --------------------------------------------------------------------------- #


def test_async_prefetch_defaults_false():
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    assert mgr._async_prefetch is False


def test_from_od_config_enables_prefetch_flag():
    od = SimpleNamespace(omni_kv_config=None, enable_kv_async_prefetch=True)
    assert OmniKVTransferManager.from_od_config(od)._async_prefetch is True
    od2 = SimpleNamespace(omni_kv_config=None)
    assert OmniKVTransferManager.from_od_config(od2)._async_prefetch is False


def test_start_load_kv_noop_when_disabled():
    _, receiver, _ = _make_sender_receiver(async_prefetch=False)
    receiver.start_load_kv({"request_id": "r1", "kv_sender_info": None})
    assert receiver._load_futures == {}


# --------------------------------------------------------------------------- #
#  Round trip
# --------------------------------------------------------------------------- #


def test_prefetch_round_trip_returns_data():
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-1")

    receiver.start_load_kv({"request_id": "rid-1", "kv_sender_info": None})
    assert "rid-1" in receiver._load_futures

    data, size = receiver.get_loaded_kv("rid-1")
    assert data is not None
    assert "layer_blocks" in data
    assert data["metadata"]["seq_len"] == 10
    assert size > 0
    # Future is consumed (popped) on retrieval.
    assert "rid-1" not in receiver._load_futures


def test_get_loaded_kv_miss_returns_none():
    _, receiver, _ = _make_sender_receiver()
    assert receiver.get_loaded_kv("never-prefetched") == (None, 0)


def test_start_load_kv_dedups_same_request():
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-dup")
    receiver.start_load_kv({"request_id": "rid-dup", "kv_sender_info": None})
    fut1 = receiver._load_futures["rid-dup"]
    receiver.start_load_kv({"request_id": "rid-dup", "kv_sender_info": None})
    assert receiver._load_futures["rid-dup"] is fut1


def test_abort_load_kv_drops_future():
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-ab")
    receiver.start_load_kv({"request_id": "rid-ab", "kv_sender_info": None})
    receiver.abort_load_kv("rid-ab")
    assert "rid-ab" not in receiver._load_futures


def test_start_load_kv_sweeps_orphan():
    # A prefetched-but-never-consumed request (e.g. aborted) must be dropped when
    # the next prefetch starts, so it does not leak.
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-orphan")
    _seed_payload(sender, "rid-next")
    receiver.start_load_kv({"request_id": "rid-orphan", "kv_sender_info": None})
    receiver.get_loaded_kv("rid-orphan")  # ensure the orphan's bg get finished
    receiver.start_load_kv({"request_id": "rid-orphan", "kv_sender_info": None})  # re-add to simulate leftover
    receiver.start_load_kv({"request_id": "rid-next", "kv_sender_info": None})
    assert "rid-orphan" not in receiver._load_futures
    assert "rid-next" in receiver._load_futures


def test_shutdown_prefetch_clears_state():
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-sd")
    receiver.start_load_kv({"request_id": "rid-sd", "kv_sender_info": None})
    receiver.shutdown_prefetch()
    assert receiver._load_futures == {}
    assert receiver._load_executor is None


# --------------------------------------------------------------------------- #
#  Correctness: per-call sender_info must not touch instance state
# --------------------------------------------------------------------------- #


def test_receive_with_sender_info_does_not_mutate_instance_state():
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-si")
    assert receiver._sender_base_host is None
    assert receiver._sender_base_zmq_port is None

    receiver.receive_kv_cache_for_request(
        "rid-si",
        target_device=None,
        sender_info={"host": "1.2.3.4", "zmq_port": 50051},
        pin=False,
    )
    # The explicit per-call endpoint must NOT leak into instance fields
    # (which only the synchronous update_sender_info path may set).
    assert receiver._sender_base_host is None
    assert receiver._sender_base_zmq_port is None


# --------------------------------------------------------------------------- #
#  Generalized inflight drain (keep-alive refs without release())
# --------------------------------------------------------------------------- #


def test_drain_handles_keepalive_and_releasable():
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    released = {"called": False}

    class _Releasable:
        def release(self):
            released["called"] = True

    mgr._kv_inflight_buffers.append([torch.randn(2)])  # keep-alive, no release()
    mgr._kv_inflight_buffers.append(_Releasable())
    mgr._drain_inflight_buffers()  # must not raise
    assert mgr._kv_inflight_buffers == []
    assert released["called"] is True


def test_apply_prefetched_kv_cpu_target_applies_without_h2d():
    sender, receiver, _ = _make_sender_receiver()
    _seed_payload(sender, "rid-apply")
    receiver.start_load_kv({"request_id": "rid-apply", "kv_sender_info": None})
    data, _ = receiver.get_loaded_kv("rid-apply")
    assert data is not None

    req = OmniDiffusionRequest(prompts=["p"], sampling_params=OmniDiffusionSamplingParams(), request_id="rid-apply")
    # CPU target → no async H2D; just attaches the payload onto the request.
    receiver.apply_prefetched_kv(req, data, target_device=torch.device("cpu"))
    assert req.past_key_values is not None
