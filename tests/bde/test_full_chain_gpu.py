# SPDX-License-Identifier: Apache-2.0
"""Full-chain BDE wiring test on real GPUs (GPU 0,1).

Spawns a rank-0 / rank-1 ``DiffusionWorker`` pair (world_size=2, NCCL rendezvous)
with the BDE runner override threaded through ``od_config``, and asserts both
workers build ``BDEModelRunner`` on their own GPU. ``skip_load_model=True`` avoids
needing DreamZero weights — this verifies the engine→worker→runner wiring across
real process + GPU boundaries, not a full rollout.

Run explicitly (needs >= 2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/bde/test_full_chain_gpu.py -v -s
"""

import multiprocessing as mp

import pytest
import torch

requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >= 2 GPUs",
)

RUNNER_OVERRIDE = "vllm_omni.bde.runner.BDEModelRunner"


def _worker_entry(rank: int, od_config, q) -> None:
    try:
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        worker = DiffusionWorker(
            local_rank=rank, rank=rank, od_config=od_config, skip_load_model=True
        )
        q.put((rank, str(worker.device), type(worker.model_runner).__name__, None))
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        q.put((rank, None, None, traceback.format_exc()))


@requires_two_gpus
def test_bde_full_chain_two_gpu():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    od_config = OmniDiffusionConfig.from_kwargs(
        model="dummy",
        model_class_name="DummyPipeline",
        num_gpus=2,
        # CFG-parallel across the 2 GPUs (the natural diffusion 2-way split).
        parallel_config={"cfg_parallel_size": 2},
        diffusion_load_format="dummy",
        diffusion_model_runner_cls=RUNNER_OVERRIDE,
    )
    # Both ranks share this config (same master_port) so they rendezvous.
    assert od_config.diffusion_model_runner_cls == RUNNER_OVERRIDE

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker_entry, args=(rank, od_config, q)) for rank in (0, 1)]
    for p in procs:
        p.start()

    results = {}
    try:
        for _ in range(2):
            rank, device, runner, err = q.get(timeout=300)
            results[rank] = (device, runner, err)
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

    for rank in (0, 1):
        assert rank in results, f"rank {rank} produced no result (hung?)"
        device, runner, err = results[rank]
        assert err is None, f"rank {rank} worker construction failed:\n{err}"
        assert runner == "BDEModelRunner", f"rank {rank} built {runner}, expected BDEModelRunner"

    # Each rank initialized its own GPU.
    assert results[0][0] == "cuda:0"
    assert results[1][0] == "cuda:1"
