"""Background KV prefetch — CPU-testable mechanism.

Covers opt-in gating, the prefetch_submit / consume_prefetched_kv round trip over a
mock connector, dedup, multi-future coexistence, miss (no _settled), abort (in-flight
removal vs out-of-order _settled), forget, per-call sender_info isolation, role
classification.  GPU H2D and D2D paths need CUDA (integration tests).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
    ReceiveRole,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# prefetch_submit() skips stubs without a sender endpoint, so submissions carry it.
_SENDER_INFO = {"host": "127.0.0.1", "zmq_port": 50051}


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


def _req(rid):
    """A minimal request carrying just ``request_id`` (consume_prefetched_kv only needs that)."""
    return SimpleNamespace(request_id=rid)


# --------------------------------------------------------------------------- #
#  Opt-in gating
# --------------------------------------------------------------------------- #


def test_async_prefetch_defaults_false():
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    assert mgr._async_prefetch is False


def test_from_od_config_enables_prefetch_flag():
    od = SimpleNamespace(omni_kv_config={"enable_kv_async_prefetch": True})
    assert OmniKVTransferManager.from_od_config(od)._async_prefetch is True
    od2 = SimpleNamespace(omni_kv_config=None)
    assert OmniKVTransferManager.from_od_config(od2)._async_prefetch is False
    od3 = SimpleNamespace(omni_kv_config={"enable_kv_async_prefetch": False})
    assert OmniKVTransferManager.from_od_config(od3)._async_prefetch is False


def test_prefetch_preserved_with_connector_config():
    # The flag must survive merging with a populated connector config.
    od = SimpleNamespace(
        omni_kv_config={
            "connector_config": {"type": "mock"},
            "need_recv_cache": True,
            "enable_kv_async_prefetch": True,
        },
    )
    assert OmniKVTransferManager.from_od_config(od)._async_prefetch is True


def test_prefetch_disabled_with_cfg_companion_collector():
    # A CFG companion collector forces the synchronous path (see _resolve_async_prefetch).
    od = SimpleNamespace(
        omni_kv_config={"enable_kv_async_prefetch": True},
        cfg_kv_collect_func=lambda *a, **k: {},
    )
    assert OmniKVTransferManager.from_od_config(od)._async_prefetch is False


def test_from_vllm_config_ar_path_is_always_sync():
    # The AR entry point never enables prefetch.
    model_config = SimpleNamespace(omni_kv_config=None)
    vllm_config = SimpleNamespace(kv_transfer_config=None)
    assert OmniKVTransferManager.from_vllm_config(vllm_config, model_config)._async_prefetch is False


def _await_submitted(receiver, rid, timeout=5.0):
    """Wait for *rid*'s future to appear (CPU-friendly poll)."""
    import time

    assert receiver._prefetch_lock is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with receiver._prefetch_lock:
            if rid in receiver._prefetch_futures:
                return receiver._prefetch_futures[rid]
        time.sleep(0.01)
    raise AssertionError(f"prefetch for {rid} was not submitted within {timeout}s")


def _await_no_future(receiver, rid, timeout=5.0):
    """Wait for *rid* to be absent from futures (SKIP path)."""
    import time

    assert receiver._prefetch_lock is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with receiver._prefetch_lock:
            if rid not in receiver._prefetch_futures:
                return
        time.sleep(0.01)
    raise AssertionError(f"prefetch for {rid} was not drained within {timeout}s")


def test_prefetch_submit_noop_when_disabled():
    _, receiver, _ = _make_sender_receiver(async_prefetch=False)
    # Prefetch state is not created when async_prefetch is False.
    assert receiver._prefetch_lock is None
    receiver.prefetch_submit("r1", _SENDER_INFO)  # no-op, must not raise
    assert receiver.consume_prefetched_kv(_req("r1")) == (None, 0)


# --------------------------------------------------------------------------- #
#  Round trip
# --------------------------------------------------------------------------- #


def test_prefetch_round_trip_returns_data():
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-1")

    receiver.prefetch_submit("rid-1", _SENDER_INFO)
    _await_submitted(receiver, "rid-1")

    data, size = receiver.consume_prefetched_kv(_req("rid-1"))
    assert data is not None
    assert "layer_blocks" in data
    assert data["metadata"]["seq_len"] == 10
    assert size > 0
    # Future is consumed (popped) on retrieval.
    assert "rid-1" not in receiver._prefetch_futures


def test_consume_prefetched_kv_miss_without_prefetch_returns_none():
    _, receiver, _ = _make_sender_receiver(async_prefetch=False)
    assert receiver.consume_prefetched_kv(_req("never-prefetched")) == (None, 0)


def test_prefetch_submit_dedups_same_request():
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-dup")
    receiver.prefetch_submit("rid-dup", _SENDER_INFO)
    fut1 = _await_submitted(receiver, "rid-dup")
    # Second enqueue is idempotent (dedup in-flight).
    receiver.prefetch_submit("rid-dup", _SENDER_INFO)
    assert receiver._prefetch_futures["rid-dup"] is fut1


def test_consume_does_not_drop_other_futures():
    # Multi-future coexistence: consuming one must not discard another.
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-a")
    _seed_payload(sender, "rid-b")
    receiver.prefetch_submit("rid-a", _SENDER_INFO)
    receiver.prefetch_submit("rid-b", _SENDER_INFO)
    _await_submitted(receiver, "rid-a")
    _await_submitted(receiver, "rid-b")

    data_a, _ = receiver.consume_prefetched_kv(_req("rid-a"))
    assert data_a is not None
    # rid-b must still be present (no serial sweep).
    assert "rid-b" in receiver._prefetch_futures


def test_consume_miss_does_not_settle():
    # Serialisation: a miss must NOT add to _settled (no orphan-future window).
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-miss")
    assert receiver.consume_prefetched_kv(_req("rid-miss")) == (None, 0)
    assert "rid-miss" not in receiver._prefetch_settled


def test_abort_removes_inflight_future():
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-abort")
    receiver.prefetch_submit("rid-abort", _SENDER_INFO)
    _await_submitted(receiver, "rid-abort")

    receiver.abort_prefetch("rid-abort")
    # rid present in queue/futures → physically removed, NOT added to _settled.
    assert "rid-abort" not in receiver._prefetch_futures
    assert "rid-abort" not in receiver._prefetch_settled


def test_abort_out_of_order_records_settled():
    # abort for a rid NOT in futures (out-of-order) → recorded in _settled
    # to block a late kv_prefetch.
    _, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    receiver.abort_prefetch("rid-ooo")
    assert "rid-ooo" in receiver._prefetch_settled
    # Late prefetch is refused (no resurrection).
    receiver.prefetch_submit("rid-ooo", _SENDER_INFO)
    assert "rid-ooo" not in receiver._prefetch_futures


def test_shutdown_prefetch_clears_state():
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-sd")
    receiver.prefetch_submit("rid-sd", _SENDER_INFO)
    _await_submitted(receiver, "rid-sd")
    receiver.shutdown_prefetch()
    assert receiver._prefetch_executor is None


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
    )
    # The per-call endpoint must not leak into instance fields.
    assert receiver._sender_base_host is None
    assert receiver._sender_base_zmq_port is None


# --------------------------------------------------------------------------- #
#  Receive-role classification (recv_role) and the follower prefetch guard
# --------------------------------------------------------------------------- #

import vllm_omni.diffusion.distributed.parallel_state as ps  # noqa: E402


def _patch_topo(monkeypatch, *, world_size, world_rank=0, cfg_size=1, cfg_rank=0, sp_size=1, sp_rank=0):
    monkeypatch.setattr(ps, "get_world_group", lambda: SimpleNamespace(world_size=world_size, rank_in_group=world_rank))
    monkeypatch.setattr(ps, "get_classifier_free_guidance_world_size", lambda: cfg_size)
    monkeypatch.setattr(ps, "get_classifier_free_guidance_rank", lambda: cfg_rank)
    monkeypatch.setattr(ps, "get_cfg_group", lambda: None)
    monkeypatch.setattr(ps, "get_sequence_parallel_world_size", lambda: sp_size)
    monkeypatch.setattr(ps, "get_sequence_parallel_rank", lambda: sp_rank)
    monkeypatch.setattr(ps, "get_sp_group", lambda: None)


def _set_tp(mgr, src, tgt):
    mgr._tp_topo = SimpleNamespace(source_tp_size=src, target_tp_size=tgt, local_rank=0)


def _role(mgr):
    """Recompute the receive role from the (patched) parallel state, bypassing the cache."""
    mgr._topo_config = None
    return mgr.topo_config.role


def test_recv_role_world1_is_local(monkeypatch):
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    _set_tp(mgr, 1, 1)
    _patch_topo(monkeypatch, world_size=1)
    assert _role(mgr) is ReceiveRole.LOCAL


def test_recv_role_pure_tp_is_local(monkeypatch):
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    _set_tp(mgr, 2, 2)
    _patch_topo(monkeypatch, world_size=2)
    assert _role(mgr) is ReceiveRole.LOCAL


def test_recv_role_cfg_leader_and_follower(monkeypatch):
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    _set_tp(mgr, 2, 2)
    _patch_topo(monkeypatch, world_size=4, cfg_size=2, cfg_rank=0)
    assert _role(mgr) is ReceiveRole.LEADER
    _patch_topo(monkeypatch, world_size=4, cfg_size=2, cfg_rank=1)
    assert _role(mgr) is ReceiveRole.FOLLOWER


def test_recv_role_sp_leader_and_follower(monkeypatch):
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    _set_tp(mgr, 2, 2)
    _patch_topo(monkeypatch, world_size=4, sp_size=2, sp_rank=0)
    assert _role(mgr) is ReceiveRole.LEADER
    _patch_topo(monkeypatch, world_size=4, sp_size=2, sp_rank=1)
    assert _role(mgr) is ReceiveRole.FOLLOWER


def test_recv_role_tp_and_sp_leader_per_tp_rank(monkeypatch):
    # TP=2 + SP=2: sp_rank 0 is leader (pulls), sp_rank 1 is follower (gets broadcast).
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    _set_tp(mgr, 2, 2)
    _patch_topo(monkeypatch, world_size=4, sp_size=2, sp_rank=0)
    assert _role(mgr) is ReceiveRole.LEADER
    _patch_topo(monkeypatch, world_size=4, sp_size=2, sp_rank=1)
    assert _role(mgr) is ReceiveRole.FOLLOWER


def test_recv_role_hsdp_world_broadcast(monkeypatch):
    # TP inactive, world > 1 (HSDP): rank-0 leader, others follower.
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    _set_tp(mgr, 1, 1)
    _patch_topo(monkeypatch, world_size=2, world_rank=0)
    assert _role(mgr) is ReceiveRole.LEADER
    _patch_topo(monkeypatch, world_size=2, world_rank=1)
    assert _role(mgr) is ReceiveRole.FOLLOWER


def test_recv_role_uninitialized_defaults_local(monkeypatch):
    mgr = OmniKVTransferManager(OmniKVCacheConfig())

    def _boom():
        raise AssertionError("world group is not initialized")

    monkeypatch.setattr(ps, "get_world_group", _boom)
    assert _role(mgr) is ReceiveRole.LOCAL


def test_prefetch_skips_follower(monkeypatch):
    # A follower never pulls, so _submit_locked skips (no future).
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-fol")
    _set_tp(receiver, 2, 2)
    _patch_topo(monkeypatch, world_size=4, cfg_size=2, cfg_rank=1)
    receiver.prefetch_submit("rid-fol", _SENDER_INFO)
    _await_no_future(receiver, "rid-fol")
    assert receiver._prefetch_futures == {}


def test_prefetch_submits_for_owner(monkeypatch):
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-own")
    _set_tp(receiver, 2, 2)
    _patch_topo(monkeypatch, world_size=4, cfg_size=2, cfg_rank=0)
    receiver.prefetch_submit("rid-own", _SENDER_INFO)
    _await_submitted(receiver, "rid-own")
    assert "rid-own" in receiver._prefetch_futures


def test_consume_then_apply_attaches_payload():
    sender, receiver, _ = _make_sender_receiver()
    receiver.init_prefetch(target_device=None)
    _seed_payload(sender, "rid-apply")
    receiver.prefetch_submit("rid-apply", _SENDER_INFO)
    _await_submitted(receiver, "rid-apply")
    data, _ = receiver.consume_prefetched_kv(_req("rid-apply"))
    assert data is not None

    req = OmniDiffusionRequest(prompt="p", sampling_params=OmniDiffusionSamplingParams(), request_id="rid-apply")
    # Mirror consume_and_distribute_kv_cache's LOCAL apply path (CPU: record_stream no-op).
    receiver._record_stream_for_prefetched(data)
    receiver.apply_kv_cache_to_request(req, data)
    assert req.past_key_values is not None
