# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for rank-aware KV transfer (TP > 1) and heterogeneous TP support.

Covers:
- _build_rank_aware_send_keys / _build_rank_aware_recv_keys
- _get_kv_source_ranks / _get_kv_target_ranks / get_kv_connector_key
- update_sender_info storing base host/port
- receive path constructing per-rank metadata for connector.get()
- Mooncake connector _query_metadata_at and partial-metadata get() path
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    KVCacheTransferData,
    OmniKVCacheConfig,
    OmniKVTransferManager,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    KV_RANK_PORT_STRIDE,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import (
    KVTPTopology,
    build_rank_aware_recv_keys,
    build_rank_aware_send_keys,
    get_kv_connector_key,
    get_kv_source_ranks,
    get_kv_target_ranks,
    merge_received_rank_shards,
    slice_received_rank_shard,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_manager(
    from_tp: int = 1,
    to_tp: int = 1,
    local_rank: int = 0,
    from_stage: str = "stage0",
    to_stage: str = "stage1",
    stage_id: str = "stage1",
    need_recv: bool = True,
    need_send: bool = False,
    recv_timeout: float = 0.3,
) -> OmniKVTransferManager:
    """Build a manager with TP params injected, bypassing torch.distributed."""
    config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage=from_stage,
        to_stage=to_stage,
        stage_id=stage_id,
        need_recv_cache=need_recv,
        need_send_cache=need_send,
        recv_timeout=recv_timeout,
        from_tp=from_tp,
        to_tp=to_tp,
    )
    with (
        patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_local_tp_rank", return_value=local_rank),
        patch(
            "vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_tp_world_size",
            return_value=max(from_tp, to_tp),
        ),
    ):
        mgr = OmniKVTransferManager(config)
    return mgr


def _make_payload(head_values: list[float], request_id: str = "req-1") -> dict:
    head_tensor = torch.tensor(head_values, dtype=torch.float32).view(1, len(head_values), 1).repeat(2, 1, 1)
    return {
        "request_id": request_id,
        "layer_blocks": {
            "key_cache": [head_tensor.clone()],
            "value_cache": [(head_tensor + 100).clone()],
        },
        "block_ids": [0],
        "metadata": {"seq_len": 2},
    }


def _make_transfer_data(head_values: list[float], request_id: str = "req-1") -> KVCacheTransferData:
    payload = _make_payload(head_values, request_id=request_id)
    return KVCacheTransferData(
        request_id=request_id,
        layer_blocks=payload["layer_blocks"],
        block_ids=payload["block_ids"],
        metadata=payload["metadata"],
    )


# ── Key format helper ────────────────────────────────────────────────


class TestConnectorKeyFormat:
    def test_key_format_matches_pr2677(self):
        key = get_kv_connector_key("req-1", "stage0", 0, 1, 2)
        assert key == "req-1_stage0_0_1_2"

    def test_key_fields_are_positional(self):
        key = get_kv_connector_key("r", "s", 5, 3, 7)
        parts = key.split("_")
        assert parts == ["r", "s", "5", "3", "7"]


# ── Source / target rank mapping ─────────────────────────────────────


class TestRankMapping:
    """Verify get_kv_target_ranks and get_kv_source_ranks for various TP configs."""

    def test_homogeneous_tp2_rank0(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=2, local_rank=0)
        assert get_kv_target_ranks(topo) == [0]
        assert get_kv_source_ranks(topo) == [0]

    def test_homogeneous_tp2_rank1(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=2, local_rank=1)
        assert get_kv_target_ranks(topo) == [1]
        assert get_kv_source_ranks(topo) == [1]

    def test_homogeneous_tp4_rank3(self):
        topo = KVTPTopology(source_tp_size=4, target_tp_size=4, local_rank=3)
        assert get_kv_target_ranks(topo) == [3]
        assert get_kv_source_ranks(topo) == [3]

    def test_sender_gt_receiver_tp4_to_tp2_rank0(self):
        """Receiver rank 0 should receive from sender rank 0 and 1."""
        topo = KVTPTopology(source_tp_size=4, target_tp_size=2, local_rank=0)
        assert get_kv_source_ranks(topo) == [0, 1]

    def test_sender_gt_receiver_tp4_to_tp2_rank1(self):
        """Receiver rank 1 should receive from sender rank 2 and 3."""
        topo = KVTPTopology(source_tp_size=4, target_tp_size=2, local_rank=1)
        assert get_kv_source_ranks(topo) == [2, 3]

    def test_sender_lt_receiver_tp2_to_tp4_rank0(self):
        """Sender rank 0 should send to receiver ranks 0 and 1."""
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=0)
        assert get_kv_target_ranks(topo) == [0, 1]

    def test_sender_lt_receiver_tp2_to_tp4_rank1(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=1)
        assert get_kv_target_ranks(topo) == [2, 3]

    def test_receiver_lt_sender_source_ranks(self):
        """Receiver rank 0 with tp2_to_tp4 should source from rank 0 only."""
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=0)
        assert get_kv_source_ranks(topo) == [0]

    def test_invalid_topology_raises(self):
        topo = KVTPTopology(source_tp_size=3, target_tp_size=2, local_rank=0)
        with pytest.raises(ValueError, match="divisible"):
            get_kv_source_ranks(topo)


# ── _build_rank_aware_recv_keys ──────────────────────────────────────


class TestBuildRankAwareRecvKeys:
    """Verify build_rank_aware_recv_keys returns (key, from_rank) tuples."""

    def test_tp1_returns_legacy_key_with_none_rank(self):
        topo = KVTPTopology(source_tp_size=1, target_tp_size=1, local_rank=0)
        pairs = build_rank_aware_recv_keys("req-1", "stage0", "stage1", topo)
        assert len(pairs) == 1
        key, rank = pairs[0]
        assert key == "omni_stage0_to_stage1_kv_cache_req-1"
        assert rank is None

    def test_homogeneous_tp2_rank0(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=2, local_rank=0)
        pairs = build_rank_aware_recv_keys("req-1", "stage0", "stage1", topo)
        assert len(pairs) == 1
        key, rank = pairs[0]
        assert key == "req-1_stage0_0_0_0"
        assert rank == 0

    def test_homogeneous_tp2_rank1(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=2, local_rank=1)
        pairs = build_rank_aware_recv_keys("req-1", "stage0", "stage1", topo)
        assert len(pairs) == 1
        key, rank = pairs[0]
        assert key == "req-1_stage0_0_1_1"
        assert rank == 1

    def test_heterogeneous_tp4_to_tp2_rank0_gets_two_keys(self):
        """Receiver rank 0 with source_tp=4, target_tp=2 should get 2 keys."""
        topo = KVTPTopology(source_tp_size=4, target_tp_size=2, local_rank=0)
        pairs = build_rank_aware_recv_keys("req-1", "stage0", "stage1", topo)
        assert len(pairs) == 2

        keys = [k for k, _ in pairs]
        ranks = [r for _, r in pairs]
        assert keys == ["req-1_stage0_0_0_0", "req-1_stage0_0_1_0"]
        assert ranks == [0, 1]

    def test_heterogeneous_tp4_to_tp2_rank1_gets_two_keys(self):
        topo = KVTPTopology(source_tp_size=4, target_tp_size=2, local_rank=1)
        pairs = build_rank_aware_recv_keys("req-1", "stage0", "stage1", topo)
        assert len(pairs) == 2

        ranks = [r for _, r in pairs]
        assert ranks == [2, 3]

    def test_heterogeneous_tp2_to_tp4_rank2_gets_one_key(self):
        """Receiver rank 2 with source_tp=2, target_tp=4 should get 1 key from sender rank 1."""
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=2)
        pairs = build_rank_aware_recv_keys("req-1", "stage0", "stage1", topo)
        assert len(pairs) == 1
        key, rank = pairs[0]
        assert rank == 1
        assert key == "req-1_stage0_0_1_2"


# ── _build_rank_aware_send_keys ──────────────────────────────────────


class TestBuildRankAwareSendKeys:
    def test_tp1_returns_legacy_key(self):
        topo = KVTPTopology(source_tp_size=1, target_tp_size=1, local_rank=0)
        keys = build_rank_aware_send_keys("req-1", "stage0", "stage1", topo)
        assert keys == ["omni_stage0_to_stage1_kv_cache_req-1"]

    def test_homogeneous_tp2_rank0(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=2, local_rank=0)
        keys = build_rank_aware_send_keys("req-1", "stage0", "stage1", topo)
        assert keys == ["req-1_stage0_0_0_0"]

    def test_sender_lt_receiver_tp2_to_tp4_rank0_sends_two_keys(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=0)
        keys = build_rank_aware_send_keys("req-1", "stage0", "stage1", topo)
        assert len(keys) == 2
        assert keys == ["req-1_stage0_0_0_0", "req-1_stage0_0_0_1"]


# ── update_sender_info stores base host/port ─────────────────────────


class TestUpdateSenderInfoBase:
    def test_stores_base_host_and_port(self):
        mgr = _make_manager(from_tp=2, to_tp=2, local_rank=0)
        mgr.update_sender_info({"host": "10.0.0.1", "zmq_port": 50151})

        assert mgr._sender_base_host == "10.0.0.1"
        assert mgr._sender_base_zmq_port == 50151

    def test_rank1_adjusts_default_port_but_preserves_base(self):
        mgr = _make_manager(from_tp=2, to_tp=2, local_rank=1)
        mgr.update_sender_info({"host": "10.0.0.1", "zmq_port": 50151})

        assert mgr._sender_base_host == "10.0.0.1"
        assert mgr._sender_base_zmq_port == 50151
        expected_adjusted = 50151 + 1 * KV_RANK_PORT_STRIDE
        assert mgr.config.connector_config["sender_zmq_port"] == expected_adjusted

    def test_nested_sender_info_resolves_correctly(self):
        """Nested sender_info keyed by integer stage id should resolve
        using recv_stages (engine_input_source → recv_from)."""
        config = OmniKVCacheConfig(
            connector_config={"type": "mock"},
            stage_id=2,
            engine_input_source=[1],
            need_recv_cache=True,
            from_tp=2,
            to_tp=2,
        )
        with (
            patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_local_tp_rank", return_value=0),
            patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_tp_world_size", return_value=2),
        ):
            mgr = OmniKVTransferManager(config)
        mgr.update_sender_info(
            {
                0: {"host": "10.0.0.1", "zmq_port": 50151},
                1: {"host": "10.0.0.2", "zmq_port": 50152},
            }
        )
        assert mgr._sender_base_host == "10.0.0.2"
        assert mgr._sender_base_zmq_port == 50152


# ── receive path constructs per-rank metadata ────────────────────────


class TestReceiveConstructsMetadata:
    """Verify that receive_kv_cache_for_request passes metadata with
    correct (host, port) to connector.get() for heterogeneous TP."""

    def test_tp1_no_metadata_passed(self):
        """TP=1: connector.get() should be called WITHOUT metadata."""
        mgr = _make_manager(from_tp=1, to_tp=1, local_rank=0, recv_timeout=0.05)
        mgr.update_sender_info({"host": "10.0.0.1", "zmq_port": 50151})

        calls = []

        class _Connector:
            def get(self, from_stage, to_stage, get_key, metadata=None):
                calls.append({"key": get_key, "metadata": metadata})
                return None

        mgr._connector = _Connector()
        mgr.receive_kv_cache_for_request("req-1")

        assert len(calls) > 0
        assert calls[0]["metadata"] is None

    def test_homogeneous_tp2_rank0_passes_metadata(self):
        """TP=2 rank 0: metadata should point to sender rank 0's port."""
        mgr = _make_manager(from_tp=2, to_tp=2, local_rank=0, recv_timeout=0.05)
        mgr.update_sender_info({"host": "10.0.0.1", "zmq_port": 50151})

        calls = []

        class _Connector:
            def get(self, from_stage, to_stage, get_key, metadata=None):
                calls.append({"key": get_key, "metadata": metadata})
                return None

        mgr._connector = _Connector()
        mgr.receive_kv_cache_for_request("req-1")

        assert len(calls) > 0
        meta = calls[0]["metadata"]
        assert meta is not None
        assert meta["source_host"] == "10.0.0.1"
        assert meta["source_port"] == 50151 + 0 * KV_RANK_PORT_STRIDE

    def test_homogeneous_tp2_rank1_passes_metadata_with_offset(self):
        mgr = _make_manager(from_tp=2, to_tp=2, local_rank=1, recv_timeout=0.05)
        mgr.update_sender_info({"host": "10.0.0.1", "zmq_port": 50151})

        calls = []

        class _Connector:
            def get(self, from_stage, to_stage, get_key, metadata=None):
                calls.append({"key": get_key, "metadata": metadata})
                return None

        mgr._connector = _Connector()
        mgr.receive_kv_cache_for_request("req-1")

        meta = calls[0]["metadata"]
        assert meta["source_port"] == 50151 + 1 * KV_RANK_PORT_STRIDE

    def test_heterogeneous_tp4_to_tp2_rank0_multiple_metadata(self):
        """Receiver rank 0 with source_tp=4, target_tp=2 should call get() with
        two different metadata entries for sender ranks 0 and 1."""
        mgr = _make_manager(from_tp=4, to_tp=2, local_rank=0, recv_timeout=0.05)
        mgr.update_sender_info({"host": "10.0.0.1", "zmq_port": 50151})

        calls = []

        class _Connector:
            def get(self, from_stage, to_stage, get_key, metadata=None):
                calls.append({"key": get_key, "metadata": metadata})
                return None

        mgr._connector = _Connector()
        mgr.receive_kv_cache_for_request("req-1")

        seen_ports = set()
        for c in calls:
            if c["metadata"]:
                seen_ports.add(c["metadata"]["source_port"])
        expected_ports = {
            50151 + 0 * KV_RANK_PORT_STRIDE,
            50151 + 1 * KV_RANK_PORT_STRIDE,
        }
        assert expected_ports.issubset(seen_ports)


# ── Mooncake connector _query_metadata_at ────────────────────────────


class TestMooncakeQueryMetadataAt:
    """Test the connector's _query_metadata_at method and partial-metadata
    path in get() without requiring real RDMA/Mooncake."""

    def test_query_metadata_at_returns_full_metadata(self):
        """Mock the ZMQ interaction to verify _query_metadata_at returns
        complete metadata including data_size."""

        try:
            from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
                MooncakeTransferEngineConnector,
                QueryResponse,
            )
        except ImportError:
            pytest.skip("Mooncake not available")

        import msgspec

        connector = MagicMock(spec=MooncakeTransferEngineConnector)
        connector._get_req_socket = MagicMock()

        mock_socket = MagicMock()
        resp = QueryResponse(request_id="test_key@s0_s1", data_size=4096, is_fast_path=True)
        mock_socket.recv.return_value = msgspec.msgpack.encode(resp)
        connector._get_req_socket.return_value = mock_socket

        result = MooncakeTransferEngineConnector._query_metadata_at(
            connector,
            "test_key@s0_s1",
            "10.0.0.1",
            50151,
        )

        assert result is not None
        assert result["source_host"] == "10.0.0.1"
        assert result["source_port"] == 50151
        assert result["data_size"] == 4096
        assert result["is_fast_path"] is True

    def test_query_metadata_at_returns_none_on_not_found(self):
        try:
            from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
                INFO_NOT_FOUND,
                MooncakeTransferEngineConnector,
            )
        except ImportError:
            pytest.skip("Mooncake not available")

        connector = MagicMock(spec=MooncakeTransferEngineConnector)
        mock_socket = MagicMock()
        mock_socket.recv.return_value = INFO_NOT_FOUND
        connector._get_req_socket.return_value = mock_socket

        result = MooncakeTransferEngineConnector._query_metadata_at(
            connector,
            "test_key@s0_s1",
            "10.0.0.1",
            50151,
        )
        assert result is None


# ── Merge / slice hooks ──────────────────────────────────────────────


class TestMergeSliceHooks:
    def test_single_shard_passes_through(self):
        payload = {"layer_blocks": {"key_cache": [1]}}
        assert merge_received_rank_shards([payload]) == payload

    def test_default_merger_concats_head_dim(self):
        p0 = _make_payload([0.0])
        p1 = _make_payload([1.0])
        result = merge_received_rank_shards([p0, p1])
        key_cache = result["layer_blocks"]["key_cache"][0]
        value_cache = result["layer_blocks"]["value_cache"][0]
        assert key_cache.shape == (2, 2, 1)
        assert value_cache.shape == (2, 2, 1)
        assert torch.equal(key_cache[:, :, 0], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        assert torch.equal(value_cache[:, :, 0], torch.tensor([[100.0, 101.0], [100.0, 101.0]]))

    def test_custom_merger_hook_called(self):
        merged = {"merged": True}
        assert merge_received_rank_shards([{}, {}], merger=lambda payloads: merged) == merged

    def test_slicer_hook_called(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=0)
        sliced = {"sliced": True}
        assert slice_received_rank_shard({"full": True}, topo, slicer=lambda payload: sliced) == sliced

    def test_default_slicer_extracts_rank_local_heads(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=1)
        payload = _make_payload([0.0, 1.0])
        result = slice_received_rank_shard(payload, topo)
        key_cache = result["layer_blocks"]["key_cache"][0]
        value_cache = result["layer_blocks"]["value_cache"][0]
        assert key_cache.shape == (2, 1, 1)
        assert value_cache.shape == (2, 1, 1)
        assert torch.equal(key_cache[:, :, 0], torch.tensor([[1.0], [1.0]]))
        assert torch.equal(value_cache[:, :, 0], torch.tensor([[101.0], [101.0]]))

    def test_presliced_payload_is_not_sliced_twice(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=1)
        payload = _make_payload([1.0])
        payload["metadata"]["tp_head_slice"] = {"applied": True, "target_rank": 1}
        result = slice_received_rank_shard(payload, topo)
        assert result is payload

    def test_round_trip_merge_from_tp4_to_tp2(self):
        topo = KVTPTopology(source_tp_size=4, target_tp_size=2, local_rank=1)
        source_ranks = get_kv_source_ranks(topo)
        payloads = [_make_payload([float(rank)]) for rank in source_ranks]
        result = merge_received_rank_shards(payloads)
        key_cache = result["layer_blocks"]["key_cache"][0]
        assert torch.equal(key_cache[:, :, 0], torch.tensor([[2.0, 3.0], [2.0, 3.0]]))

    def test_round_trip_slice_from_tp2_to_tp4(self):
        topo = KVTPTopology(source_tp_size=2, target_tp_size=4, local_rank=3)
        payload = _make_payload([2.0, 3.0])
        result = slice_received_rank_shard(payload, topo)
        key_cache = result["layer_blocks"]["key_cache"][0]
        assert torch.equal(key_cache[:, :, 0], torch.tensor([[3.0], [3.0]]))


class TestSenderSideSlicing:
    def test_transfer_slices_before_sending_to_multiple_targets(self):
        mgr = _make_manager(
            from_tp=2,
            to_tp=4,
            local_rank=0,
            need_send=True,
            need_recv=False,
        )
        sent_payloads = []

        class _Connector:
            supports_raw_data = False

            def put(self, from_stage, to_stage, put_key, data):
                sent_payloads.append((put_key, KVCacheTransferData.from_bytes(data)))
                return True, len(data), {}

        mgr._connector = _Connector()
        mgr._transfer_kv_cache(_make_transfer_data([0.0, 1.0]), "req-1")

        assert [key for key, _ in sent_payloads] == ["req-1_stage0_0_0_0", "req-1_stage0_0_0_1"]
        assert sent_payloads[0][1]["layer_blocks"]["key_cache"][0].shape == (2, 1, 1)
        assert sent_payloads[1][1]["layer_blocks"]["key_cache"][0].shape == (2, 1, 1)
        assert torch.equal(
            sent_payloads[0][1]["layer_blocks"]["key_cache"][0][:, :, 0],
            torch.tensor([[0.0], [0.0]]),
        )
        assert torch.equal(
            sent_payloads[1][1]["layer_blocks"]["key_cache"][0][:, :, 0],
            torch.tensor([[1.0], [1.0]]),
        )
        assert sent_payloads[0][1]["metadata"]["tp_head_slice"]["target_rank"] == 0
        assert sent_payloads[1][1]["metadata"]["tp_head_slice"]["target_rank"] == 1


class _MockBroadcastGroup:
    def __init__(self, world_size: int, rank_in_group: int, broadcast_value=None, recv_value=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.broadcast_value = broadcast_value
        self.recv_value = recv_value
        self.broadcast_calls = []
        self.send_calls = []
        self.recv_calls = []
        self.shm_broadcaster = None

    def broadcast_object(self, obj=None, src: int = 0):
        self.broadcast_calls.append((obj, src))
        return self.broadcast_value if self.broadcast_value is not None else obj

    def send_object(self, obj, dst: int):
        self.send_calls.append((dst, obj))

    def recv_object(self, src: int):
        self.recv_calls.append(src)
        return self.recv_value


class TestDistributedReceive:
    def test_tp_cfg_leader_receives_then_sends_branch_local_payloads(self):
        mgr = _make_manager(from_tp=2, to_tp=4, local_rank=0)
        req = SimpleNamespace(request_id="req-1", sampling_params=SimpleNamespace())
        world_group = _MockBroadcastGroup(world_size=4, rank_in_group=2)
        cfg_group = _MockBroadcastGroup(world_size=3, rank_in_group=0)

        def _receive(req_obj, cfg_func, target_device):
            req_obj.past_key_values = SimpleNamespace(key_cache=[torch.tensor([1.0])])
            req_obj.kv_metadata = {"source": "leader"}
            req_obj.sampling_params.past_key_values = req_obj.past_key_values
            req_obj.sampling_params.kv_metadata = req_obj.kv_metadata
            req_obj.sampling_params.cfg_text_past_key_values = SimpleNamespace(key_cache=[torch.tensor([2.0])])
            req_obj.sampling_params.cfg_text_kv_metadata = {"source": "cfg_text"}
            req_obj.sampling_params.cfg_img_past_key_values = SimpleNamespace(key_cache=[torch.tensor([3.0])])
            req_obj.sampling_params.cfg_img_kv_metadata = {"source": "cfg_img"}
            return True

        mgr.receive_multi_kv_cache = MagicMock(side_effect=_receive)
        with (
            patch("vllm_omni.diffusion.distributed.parallel_state.get_world_group", return_value=world_group),
            patch(
                "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_world_size",
                return_value=3,
            ),
            patch(
                "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_rank",
                return_value=0,
            ),
            patch("vllm_omni.diffusion.distributed.parallel_state.get_cfg_group", return_value=cfg_group),
        ):
            assert mgr.receive_multi_kv_cache_distributed(req) is True

        mgr.receive_multi_kv_cache.assert_called_once()
        assert mgr.receive_multi_kv_cache.call_args.args[2] == torch.device("cpu")
        assert req.kv_metadata == {"source": "leader"}
        assert cfg_group.broadcast_calls == []
        assert [dst for dst, _ in cfg_group.send_calls] == [1, 2]
        rank1_payload = cfg_group.send_calls[0][1]
        rank2_payload = cfg_group.send_calls[1][1]
        assert torch.equal(rank1_payload["past_key_values"].key_cache[0], torch.tensor([1.0]))
        assert torch.equal(rank2_payload["past_key_values"].key_cache[0], torch.tensor([1.0]))
        assert rank1_payload["sp.cfg_active_branch"] == "cfg_text"
        assert rank2_payload["sp.cfg_active_branch"] == "cfg_img"
        assert rank1_payload["sp.cfg_branch_roles"] == ["cfg_text", "cfg_img"]
        assert rank2_payload["sp.cfg_branch_roles"] == ["cfg_text", "cfg_img"]
        assert "sp.cfg_branch_past_key_values" in rank1_payload
        assert "sp.cfg_branch_past_key_values" in rank2_payload
        assert list(rank1_payload["sp.cfg_branch_past_key_values"].keys()) == ["cfg_text"]
        assert list(rank2_payload["sp.cfg_branch_past_key_values"].keys()) == ["cfg_img"]
        assert "sp.cfg_text_past_key_values" in rank1_payload
        assert "sp.cfg_img_past_key_values" not in rank1_payload
        assert "sp.cfg_img_past_key_values" in rank2_payload
        assert "sp.cfg_text_past_key_values" not in rank2_payload

    def test_tp_cfg_follower_receives_local_payload_without_receiving(self):
        mgr = _make_manager(from_tp=2, to_tp=4, local_rank=1)
        req = SimpleNamespace(request_id="req-1", sampling_params=SimpleNamespace())
        world_group = _MockBroadcastGroup(world_size=4, rank_in_group=3)
        cfg_payload = {
            "past_key_values": SimpleNamespace(key_cache=[torch.tensor([1.0])]),
            "kv_metadata": {"source": "main"},
            "sp.past_key_values": SimpleNamespace(key_cache=[torch.tensor([1.0])]),
            "sp.kv_metadata": {"source": "main"},
            "sp.cfg_active_branch": "cfg_text",
            "sp.cfg_branch_roles": ["cfg_text", "cfg_img"],
            "sp.cfg_branch_past_key_values": {
                "cfg_text": SimpleNamespace(key_cache=[torch.tensor([2.0])]),
            },
            "sp.cfg_branch_kv_metadata": {"cfg_text": {"source": "cfg-text"}},
            "sp.cfg_text_past_key_values": SimpleNamespace(key_cache=[torch.tensor([2.0])]),
        }
        cfg_group = _MockBroadcastGroup(world_size=2, rank_in_group=1, recv_value=cfg_payload)

        mgr.receive_multi_kv_cache = MagicMock(return_value=True)
        with (
            patch("vllm_omni.diffusion.distributed.parallel_state.get_world_group", return_value=world_group),
            patch(
                "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_world_size",
                return_value=2,
            ),
            patch(
                "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_rank",
                return_value=1,
            ),
            patch("vllm_omni.diffusion.distributed.parallel_state.get_cfg_group", return_value=cfg_group),
        ):
            assert mgr.receive_multi_kv_cache_distributed(req) is True

        mgr.receive_multi_kv_cache.assert_not_called()
        assert req.kv_metadata == {"source": "main"}
        assert torch.equal(req.past_key_values.key_cache[0], torch.tensor([1.0]))
        assert torch.equal(req.sampling_params.past_key_values.key_cache[0], torch.tensor([1.0]))
        assert req.sampling_params.cfg_active_branch == "cfg_text"
        assert req.sampling_params.cfg_branch_roles == ["cfg_text", "cfg_img"]
        assert torch.equal(
            req.sampling_params.cfg_branch_past_key_values["cfg_text"].key_cache[0],
            torch.tensor([2.0]),
        )
        assert req.sampling_params.cfg_branch_kv_metadata == {"cfg_text": {"source": "cfg-text"}}
        assert torch.equal(req.sampling_params.cfg_text_past_key_values.key_cache[0], torch.tensor([2.0]))
        assert cfg_group.broadcast_calls == []
        assert cfg_group.recv_calls == [0]

    def test_tp_without_cfg_keeps_independent_receive_path(self):
        mgr = _make_manager(from_tp=2, to_tp=2, local_rank=1)
        req = SimpleNamespace(request_id="req-1", sampling_params=SimpleNamespace())
        world_group = _MockBroadcastGroup(world_size=2, rank_in_group=1)
        mgr.receive_multi_kv_cache = MagicMock(return_value=True)

        with patch("vllm_omni.diffusion.distributed.parallel_state.get_world_group", return_value=world_group):
            assert mgr.receive_multi_kv_cache_distributed(req, target_device=torch.device("cpu")) is True

        mgr.receive_multi_kv_cache.assert_called_once_with(req, None, torch.device("cpu"))


# ── TP auto-detect ───────────────────────────────────────────────────


class TestAutoDetectTP:
    def test_auto_detect_when_config_defaults(self):
        """When config from_tp/to_tp == 1 (default), manager should auto-detect."""
        config = OmniKVCacheConfig(
            connector_config={"type": "mock"},
            from_stage="s0",
            stage_id="s1",
            need_recv_cache=True,
        )
        with (
            patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_local_tp_rank", return_value=0),
            patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_tp_world_size", return_value=4),
        ):
            mgr = OmniKVTransferManager(config)
        assert mgr._tp_topo.source_tp_size == 4
        assert mgr._tp_topo.target_tp_size == 4

    def test_explicit_tp_overrides_auto_detect(self):
        config = OmniKVCacheConfig(
            connector_config={"type": "mock"},
            from_stage="s0",
            stage_id="s1",
            need_recv_cache=True,
            from_tp=2,
            to_tp=4,
        )
        with (
            patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_local_tp_rank", return_value=0),
            patch("vllm_omni.distributed.omni_connectors.kv_transfer_manager.get_tp_world_size", return_value=8),
        ):
            mgr = OmniKVTransferManager(config)
        assert mgr._tp_topo.source_tp_size == 2
        assert mgr._tp_topo.target_tp_size == 4
