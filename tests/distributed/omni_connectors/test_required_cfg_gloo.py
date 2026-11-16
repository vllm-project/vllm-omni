# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Receive failures with real CPU collectives and request-isolated groups.

Only the remote connector is stubbed. GroupCoordinator object transport, CFG
scatter, SP broadcast and the manager's receive entry points execute normally.
No model weights or accelerator are needed.
"""

import time
from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import RankGenerator
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    KVCacheReceiveError,
    KVPrefetchConsumeError,
    OmniKVCacheConfig,
    OmniKVTransferManager,
    ReceiveRole,
    _TransferTopoConfig,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.bagel import collect_cfg_kv_caches

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.cache, pytest.mark.parallel]


def _coordinator(members, rank, process_group):
    # Bypass accelerator initialization, retaining the real transport methods.
    group = object.__new__(GroupCoordinator)
    group.ranks = members
    group.world_size = len(members)
    group.rank_in_group = members.index(rank)
    group.cpu_group = process_group
    group.shm_broadcaster = None
    return group


def _run_topology(rank, shape):
    tp_size, cfg_size, sp_size, dp_size = shape
    layout = RankGenerator(tp=tp_size, sp=sp_size, pp=1, cfg=cfg_size, dp=dp_size)
    groups = {}
    created = []
    for axis in ("tp", "cfg", "sp"):
        for members in layout.get_ranks(axis):
            pg = dist.new_group(members, backend="gloo", timeout=timedelta(seconds=20))
            if rank in members:
                groups[axis] = _coordinator(members, rank, pg)
                created.append(pg)
    tp, cfg, sp = (groups[axis].rank_in_group for axis in ("tp", "cfg", "sp"))
    dp = rank // (tp_size * cfg_size * sp_size)
    local = tp_size > 1 and cfg_size == sp_size == 1
    leader = cfg == sp == 0 if tp_size > 1 else rank == 0
    role = ReceiveRole.LOCAL if local else ReceiveRole.LEADER if leader else ReceiveRole.FOLLOWER
    world = _coordinator(list(range(dist.get_world_size())), rank, dist.group.WORLD)
    manager = OmniKVTransferManager(OmniKVCacheConfig(need_recv_cache=True))
    manager._topo_config = _TransferTopoConfig(
        role,
        tp_size > 1,
        cfg_size,
        cfg,
        groups["cfg"],
        sp_size,
        sp,
        groups["sp"],
        Mock() if tp_size > 1 else world,
        groups["tp"],
        tp_size,
    )
    manager._async_prefetch = True
    manager._record_stream_for_prefetched = Mock()
    real_distribution = manager.distribute_kv_cache
    tp_broadcast = Mock(wraps=groups["tp"].broadcast_object)
    groups["tp"].broadcast_object = tp_broadcast

    # Alternate broken fetching ranks, then use the same groups for healthy work.
    # Replica 1 either processes a different healthy request or does no receive.
    scenarios: list[tuple[str, int | None, bool]] = [
        (mode, failed_tp, False)
        for mode in ("sync", "sync-error", "miss-error", "consumed-error")
        for failed_tp in range(tp_size)
    ]
    if dp_size > 1:
        scenarios.append(("sync", 0, True))
    # Equal KV signatures are insufficient when every fetching rank lacks CFG.
    scenarios.append(("sync-all-missing", 0, False))
    scenarios.extend((mode, None, False) for mode in ("hit", "miss", "sync"))
    for index, (mode, failed_tp, idle_replica) in enumerate(scenarios):
        tp_broadcast.reset_mock()
        error_message = None
        if not (idle_replica and dp > 0):
            rid = f"request-{index}-replica-{dp}"
            params = OmniDiffusionSamplingParams(
                seed=0,
                cfg_kv_request_ids={"cfg_text": rid + "__cfg_text"} if mode in ("sync", "sync-all-missing") else None,
            )
            req = OmniDiffusionRequest(prompt="test", request_id=rid, sampling_params=params)
            broken = dp == 0 and (tp == failed_tp or mode == "sync-all-missing")
            expected_failure = dp == 0 and failed_tp is not None
            payload = {"layer_blocks": {"key_cache": [], "value_cache": []}, "metadata": {"owner": rid}}

            def fetch(request_id, target_device=None):
                if broken and mode in ("sync-error", "miss-error"):
                    raise OSError("injected connector failure")
                if request_id.endswith("__cfg_text") and broken:
                    return None, 0
                return payload, 1

            def consume(request):
                if broken and mode == "consumed-error":
                    raise KVPrefetchConsumeError("injected consumed payload failure")
                return (None, 0) if mode in ("miss", "miss-error") else (payload, 1)

            manager.receive_kv_cache_for_request = Mock(side_effect=fetch)
            manager.consume_prefetched_kv = Mock(side_effect=consume)
            manager.distribute_kv_cache = Mock(wraps=real_distribution)
            try:
                if mode in ("sync", "sync-error", "sync-all-missing"):
                    received = manager.receive_multi_kv_cache_distributed(req, collect_cfg_kv_caches)
                else:
                    received = manager.consume_and_distribute_kv_cache(req)
            except KVCacheReceiveError as exc:
                assert expected_failure, (shape, rank, mode, str(exc))
                error_message = str(exc)
                assert "cfg_text" in error_message or "injected" in error_message
                assert getattr(params, "past_key_values", None) is None
                manager.distribute_kv_cache.assert_not_called()
                if mode == "sync-all-missing" and local and tp == 0:
                    tp_broadcast.assert_called_once_with((True, ("Missing required CFG KV: cfg_text",)), src=0)
                if broken and mode == "consumed-error" and not manager.topo_config.is_follower:
                    manager.receive_kv_cache_for_request.assert_not_called()
            else:
                assert not expected_failure, (shape, rank, mode)
                assert received
                assert params.kv_metadata["owner"] == rid
                manager.distribute_kv_cache.assert_called_once()
            if tp_size > 1:
                assert not manager.topo_config.world.mock_calls
        # The test-wide audit includes idle replicas; production request
        # collectives above must never wait for them.
        messages = [None] * dist.get_world_size()
        dist.all_gather_object(messages, error_message)
        request_ranks = tp_size * cfg_size * sp_size
        if failed_tp is not None:
            assert messages[0] is not None
            assert len(set(messages[:request_ranks])) == 1
        assert all(message is None for message in messages[request_ranks:])
    for pg in reversed(created):
        dist.destroy_process_group(pg)


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8, timeout=timedelta(seconds=30))
    try:
        for shape in ((2, 1, 1, 4), (2, 2, 2, 1), (2, 1, 2, 2), (1, 2, 4, 1)):
            _run_topology(rank, shape)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required")
def test_required_cfg_real_collectives_and_replica_isolation(tmp_path):
    context = mp.spawn(_worker, args=((tmp_path / "store").as_uri(),), nprocs=8, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("required-CFG workers did not finish within 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
