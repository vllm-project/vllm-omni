# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    KVCacheTransferData,
    OmniKVCacheConfig,
    OmniKVTransferManager,
    ReceiveRole,
    _TransferTopoConfig,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request():
    return OmniDiffusionRequest(
        prompt="KV distribution regression",
        sampling_params=OmniDiffusionSamplingParams(seed=42),
        request_id="kv-distribution-regression",
    )


def _attach_kv(manager, req, offset=0):
    key = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + offset
    data = KVCacheTransferData(
        request_id=req.request_id,
        layer_blocks={"key_cache": [key], "value_cache": [key + 100]},
        block_ids=[0],
        metadata={"seq_len": 2},
    )
    manager.apply_kv_cache_to_request(req, data.to_dict())


@pytest.mark.parametrize("need_send_cache", [False, True])
@pytest.mark.parametrize("received", [False, True])
def test_disabled_distribution_preserves_request_without_topology(need_send_cache, received):
    manager = OmniKVTransferManager(OmniKVCacheConfig(need_recv_cache=False, need_send_cache=need_send_cache))
    req = _request()
    _attach_kv(manager, req)
    original_kv = req.past_key_values
    original_metadata = req.kv_metadata
    req.payload_sender_info = {"sender_host": "127.0.0.1", "sender_port": 50051}
    original_sender_info = req.payload_sender_info

    assert manager.distribute_kv_cache(req, torch.device("cpu"), received=received) is None

    # Disabled distribution must not initialize groups or touch existing inputs.
    assert manager._topo_config is None
    assert manager.config.need_send_cache is need_send_cache
    assert req.past_key_values is original_kv
    assert req.sampling_params.past_key_values is original_kv
    assert req.kv_metadata is original_metadata
    assert req.payload_sender_info is original_sender_info
    torch.testing.assert_close(original_kv.key_cache[0], torch.arange(8).reshape(2, 2, 2).float(), rtol=0, atol=0)


def _topology(group, mode):
    rank = group.rank_in_group
    return _TransferTopoConfig(
        role=ReceiveRole.LOCAL if mode == "local" else ReceiveRole.LEADER if rank == 0 else ReceiveRole.FOLLOWER,
        tp_active=mode != "world",
        cfg_size=2 if mode == "cfg" else 1,
        cfg_rank=rank if mode == "cfg" else 0,
        cfg_group=group if mode == "cfg" else None,
        sp_size=2 if mode == "sp" else 1,
        sp_rank=rank if mode == "sp" else 0,
        sp_group=group if mode == "sp" else None,
        world=group,
        tp_group=group,
        tp_size=2,
    )


def _distributed_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    group = GroupCoordinator([[0, 1]], local_rank=rank, torch_distributed_backend="gloo")
    # These tests deliberately exercise real CPU transport on any platform.
    group.device = torch.device("cpu")
    try:
        for mode in ("world", "sp", "cfg", "local"):
            manager = OmniKVTransferManager(
                OmniKVCacheConfig(need_recv_cache=True, from_tp=1 if mode == "world" else 2)
            )
            manager._topo_config = _topology(group, mode)

            # Followers have received=False and must still enter distribution.
            req = _request()
            if rank == 0:
                _attach_kv(manager, req)
                if mode == "cfg":
                    branch = _request()
                    _attach_kv(manager, branch, offset=10)
                    req.sampling_params.cfg_branch_past_key_values = {"cfg_text": branch.past_key_values}
                    req.sampling_params.cfg_branch_kv_metadata = {"cfg_text": {"branch": "negative"}}
            payload = manager.distribute_kv_cache(req, torch.device("cpu"), received=rank == 0)
            if mode == "local":
                assert payload is None
                assert (getattr(req, "past_key_values", None) is not None) == (rank == 0)
            else:
                assert payload is not None
                manager._apply_request_kv_payload(req, payload, torch.device("cpu"))
                expected = torch.arange(8).reshape(2, 2, 2).float()
                for kv in (req.past_key_values, req.sampling_params.past_key_values):
                    torch.testing.assert_close(kv.key_cache[0], expected, rtol=0, atol=0)
                    torch.testing.assert_close(kv.value_cache[0], expected + 100, rtol=0, atol=0)
                assert req.kv_metadata == {"seq_len": 2}
                if mode == "cfg" and rank == 1:
                    assert req.sampling_params.cfg_active_branch == "cfg_text"
                    assert list(req.sampling_params.cfg_branch_past_key_values) == ["cfg_text"]
                    branch_kv = req.sampling_params.cfg_branch_past_key_values["cfg_text"]
                    torch.testing.assert_close(branch_kv.key_cache[0], expected + 10, rtol=0, atol=0)
                    assert req.sampling_params.cfg_text_kv_metadata == {"branch": "negative"}

            # Metadata-only payloads use the production object-transport path.
            req = _request()
            if rank == 0:
                req.kv_metadata = {"seq_len": 2}
            payload = manager.distribute_kv_cache(req, torch.device("cpu"), received=rank == 0)
            if mode == "local":
                assert payload is None
            else:
                assert payload["kv_metadata"] == {"seq_len": 2}

            # An enabled receiver's miss must keep the ranks in lockstep.
            assert manager.distribute_kv_cache(_request(), torch.device("cpu"), received=False) is None

            manager.config.need_recv_cache = False
            sequence_before = group.cpu_group._get_sequence_number_for_group()
            device_sequence_before = group.device_group._get_sequence_number_for_group()
            assert manager.distribute_kv_cache(_request(), torch.device("cpu"), received=False) is None
            assert group.cpu_group._get_sequence_number_for_group() == sequence_before
            assert group.device_group._get_sequence_number_for_group() == device_sequence_before
            dist.barrier()
    finally:
        group.destroy()
        dist.destroy_process_group()


def test_enabled_distribution_real_gloo(tmp_path):
    mp.spawn(_distributed_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=2, join=True)
