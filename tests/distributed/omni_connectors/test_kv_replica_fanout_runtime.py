# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime tests for EP replica KV fanout object collectives."""

from __future__ import annotations

import os
import socket
import traceback
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.distributed.omni_connectors.utils.parallel_plan import (
    KVParallelRankCoord,
    ParallelAxis,
    build_kv_receive_distribution_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CPUObjectGroup:
    """Small GroupCoordinator-compatible wrapper over a gloo process group."""

    def __init__(self, *, rank: int, world_size: int, cpu_group: dist.ProcessGroup):
        self.rank_in_group = rank
        self.world_size = world_size
        self.cpu_group = cpu_group
        self.ranks = list(range(world_size))

    def send_object(self, obj: Any, dst: int) -> None:
        dist.send_object_list([obj], dst=self.ranks[dst], group=self.cpu_group)

    def recv_object(self, src: int) -> Any:
        objects: list[Any] = [None]
        dist.recv_object_list(objects, src=self.ranks[src], group=self.cpu_group)
        return objects[0]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_plan(*, rank: int, world_size: int):
    tp_rank = rank // 2
    ep_rank = rank % 2
    coord = KVParallelRankCoord(
        axes=(
            ParallelAxis("tp", 2, tp_rank, "tensor_shard"),
            ParallelAxis("pp", 1, 0, "tensor_shard"),
            ParallelAxis("ring", 1, 0, "tensor_shard"),
            ParallelAxis("ulysses", 1, 0, "tensor_shard"),
            ParallelAxis("cfg", 1, 0, "branch"),
            ParallelAxis("sp", 1, 0, "container"),
            ParallelAxis("dp", 1, 0, "replica"),
            ParallelAxis("ep", 2, ep_rank, "replica"),
        )
    )
    return build_kv_receive_distribution_plan(
        coord=coord,
        world_size=world_size,
        world_rank=rank,
        source_tp_size=2,
        target_tp_size=2,
    )


def _make_request(request_id: str = "req-1"):
    return SimpleNamespace(request_id=request_id, sampling_params=SimpleNamespace())


def _fanout_worker(rank: int, world_size: int, port: int, result_queue) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )

    try:
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        replica_group = _CPUObjectGroup(
            rank=rank,
            world_size=world_size,
            cpu_group=dist.group.WORLD,
        )
        plan = _make_plan(rank=rank, world_size=world_size)
        mgr = OmniKVTransferManager.__new__(OmniKVTransferManager)

        fanout = mgr._replica_fanout_peer_ranks(
            plan=plan,
            replica_group=replica_group,
            participates=True,
        )
        assert fanout is not None

        success_req = _make_request()

        def _receive_success(req, cfg_kv_collect_func, target_device):
            req.past_key_values = SimpleNamespace(key_cache=[torch.tensor([float(rank)])])
            req.kv_metadata = {"owner_rank": rank}
            req.sampling_params.past_key_values = req.past_key_values
            req.sampling_params.kv_metadata = req.kv_metadata
            return True

        mgr.receive_multi_kv_cache = MagicMock(side_effect=_receive_success)
        success_payload = mgr._receive_remote_kv_payload_from_replica_fanout(
            req=success_req,
            replica_group=replica_group,
            fanout=fanout,
            cfg_kv_collect_func=None,
        )
        assert success_payload is not None
        success_remote_receives = mgr.receive_multi_kv_cache.call_count
        dist.barrier()

        failure_req = _make_request("req-fail")
        mgr.receive_multi_kv_cache = MagicMock(return_value=False)
        failure_payload = mgr._receive_remote_kv_payload_from_replica_fanout(
            req=failure_req,
            replica_group=replica_group,
            fanout=fanout,
            cfg_kv_collect_func=None,
        )
        failure_remote_receives = mgr.receive_multi_kv_cache.call_count
        dist.barrier()

        result_queue.put(
            {
                "rank": rank,
                "fanout": fanout,
                "success_owner_rank": success_req.kv_metadata["owner_rank"],
                "success_tensor_value": float(success_req.past_key_values.key_cache[0].item()),
                "success_remote_receives": success_remote_receives,
                "failure_remote_receives": failure_remote_receives,
                "failure_payload": failure_payload,
                "failure_has_kv": hasattr(failure_req, "past_key_values"),
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_replica_fanout_runtime_pairs_owners_and_sends_none_sentinel():
    world_size = 4
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_fanout_worker, args=(rank, world_size, port, result_queue)) for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)

    alive = [process.pid for process in processes if process.is_alive()]
    if alive:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(timeout=5)
        pytest.fail(f"replica fanout runtime test deadlocked; live pids={alive}")

    results = [result_queue.get(timeout=1) for _ in range(world_size)]
    errors = [result for result in results if "error" in result]
    assert errors == []
    assert [process.exitcode for process in processes] == [0] * world_size

    by_rank = {result["rank"]: result for result in results}
    assert {rank: by_rank[rank]["fanout"] for rank in range(world_size)} == {
        0: (0, (0, 1)),
        1: (0, (0, 1)),
        2: (2, (2, 3)),
        3: (2, (2, 3)),
    }

    assert by_rank[0]["success_owner_rank"] == 0
    assert by_rank[1]["success_owner_rank"] == 0
    assert by_rank[2]["success_owner_rank"] == 2
    assert by_rank[3]["success_owner_rank"] == 2
    assert by_rank[1]["success_tensor_value"] == 0.0
    assert by_rank[3]["success_tensor_value"] == 2.0

    for rank, result in by_rank.items():
        expected_owner_call_count = 1 if rank in (0, 2) else 0
        assert result["success_remote_receives"] == expected_owner_call_count
        assert result["failure_remote_receives"] == expected_owner_call_count
        assert result["failure_payload"] is None
        assert not result["failure_has_kv"]
