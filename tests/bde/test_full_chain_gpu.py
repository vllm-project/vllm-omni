# SPDX-License-Identifier: Apache-2.0
"""Full-chain BDE wiring + pool test on real GPUs (GPU 2,3).

Spawns a rank-0 / rank-1 ``DiffusionWorker`` pair (world_size=2, NCCL rendezvous)
with the BDE runner override, asserts both workers build ``BDEModelRunner``, then
constructs a ``BDEKVCache`` with GPU-resident pools and runs a write→gather
roundtrip on each rank. No model weights needed.

Run explicitly (needs >= 4 GPUs):
    CUDA_VISIBLE_DEVICES=2,3 python -m pytest tests/bde/test_full_chain_gpu.py -v -s
"""

import multiprocessing as mp

import pytest
import torch

requires_gpus_2_3 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2,
    reason="requires >= 2 GPUs",
)

RUNNER_OVERRIDE = "vllm_omni.bde.runner.BDEModelRunner"
BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def _worker_entry(rank: int, od_config, q) -> None:
    try:
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        worker = DiffusionWorker(local_rank=rank, rank=rank, od_config=od_config, skip_load_model=True)
        device = str(worker.device)
        runner_name = type(worker.model_runner).__name__

        # --- pool write + gather on the worker's GPU ------------------------
        from vllm_omni.bde.kv_cache import BDEKVCache, BDEKVConfig

        cfg = BDEKVConfig(enable=True, chunk_size=BLOCK, window_chunks=2)
        kv = BDEKVCache(
            cfg,
            num_layers=1,
            num_kv_heads=N_HEADS,
            head_size=HEAD_DIM,
            dtype=torch.float32,
            block_size=BLOCK,
            max_model_len=512,
            available_bytes=1 << 28,
            device=worker.device,
        )

        # Verify pools are on the worker's GPU.
        assert kv._k_pools[0].device == worker.device
        assert kv._v_pools[0].device == worker.device
        assert kv._k_pools[0].shape == (kv.num_blocks * BLOCK, N_HEADS, HEAD_DIM)

        # Write a chunk → gather window → verify roundtrip.
        adapter = kv.begin_request("r")
        kv.allocate_chunk(adapter)
        write_slots = kv.chunk_write_slots(adapter)

        new_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, device=worker.device)
        new_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, device=worker.device)
        kv.write_chunk_kv(0, new_k, new_v, adapter)
        kv.commit_chunk(adapter)

        # Read back from the pool at the write slots.
        pool_read_k = kv._k_pools[0][write_slots]
        pool_read_v = kv._v_pools[0][write_slots]
        assert torch.allclose(pool_read_k, new_k[0], atol=1e-5), f"gathered K ≠ written (rank {rank})"
        assert torch.allclose(pool_read_v, new_v[0], atol=1e-5), f"gathered V ≠ written (rank {rank})"

        # Allocate a second chunk then gather the window.
        kv.allocate_chunk(adapter)
        window = kv.gather_window(0, adapter)
        assert window.device == worker.device
        assert window.shape == (2, 1, kv.spec.sliding_window, N_HEADS, HEAD_DIM)

        q.put((rank, device, runner_name, "pool_roundtrip_ok", None))
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        q.put((rank, None, None, None, traceback.format_exc()))


@requires_gpus_2_3
def test_bde_full_chain_two_gpu_with_pools():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    od_config = OmniDiffusionConfig.from_kwargs(
        model="dummy",
        model_class_name="DummyPipeline",
        num_gpus=2,
        parallel_config={"cfg_parallel_size": 2},
        diffusion_load_format="dummy",
        diffusion_model_runner_cls=RUNNER_OVERRIDE,
    )
    assert od_config.diffusion_model_runner_cls == RUNNER_OVERRIDE

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker_entry, args=(rank, od_config, q)) for rank in (0, 1)]
    for p in procs:
        p.start()

    results = {}
    try:
        for _ in range(2):
            rank, device, runner, pool_status, err = q.get(timeout=300)
            results[rank] = (device, runner, pool_status, err)
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

    for rank in (0, 1):
        assert rank in results, f"rank {rank} produced no result (hung?)"
        device, runner, pool_status, err = results[rank]
        assert err is None, f"rank {rank} failed:\n{err}"
        assert runner == "BDEModelRunner", f"rank {rank} built {runner}, expected BDEModelRunner"
        assert pool_status == "pool_roundtrip_ok"

    assert results[0][0] == "cuda:0"
    assert results[1][0] == "cuda:1"
