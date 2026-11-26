# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-rank cache decisions through the production backend, with small tensors."""

import os
from contextlib import nullcontext
from datetime import timedelta
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tests.diffusion.cache.test_inter_request_backend import FakePipeline, make_req
from vllm_omni.diffusion.cache.inter_request.backend import InterRequestCacheBackend
from vllm_omni.diffusion.cache.inter_request.cache_store import StepLatentData
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _exercise_decisions(rank, device):
    def run_wave(requests, expected_hits, expected_steps):
        with torch.inference_mode():
            hits, remaining = backend.short_circuit_requests(requests, target_device=device)
        assert [index for index, _ in hits] == expected_hits
        assert [req.prompt for req in remaining] == [
            req.prompt for i, req in enumerate(requests) if i not in expected_hits
        ]
        for _, output in hits:
            assert output.output.item() == rank + 10
        assert [req.sampling_params.resume_from_step for req in remaining] == expected_steps
        for req in remaining:
            step = req.sampling_params.resume_from_step
            if step:
                assert req.sampling_params.resume_latents.item() == rank * 10 + step - 1
            else:
                assert req.sampling_params.resume_latents is None
            # Model one collective per executed denoising iteration. Divergent
            # decisions either violate the expectations above or fail this wave.
            for _ in range(step, req.sampling_params.num_inference_steps):
                signal = torch.ones(1, device=device)
                dist.all_reduce(signal)
                assert signal.item() == 2

    backend = InterRequestCacheBackend(DiffusionCacheConfig())
    backend.enable(FakePipeline())
    final = torch.tensor([rank + 10.0])
    try:
        # Mixed batch: both have one output, rank 0 alone has another.
        if rank == 0:
            backend.store(make_req("one rank"), final)
        backend.store(make_req("both ranks"), final)
        run_wave([make_req("one rank"), make_req("both ranks"), make_req("cold")], [1], [0, 0])
        run_wave([make_req("both ranks")], [0], [])
        fault = patch.object(backend.cache_store, "get", side_effect=OSError("lost cache file"))
        with fault if rank == 1 else nullcontext():
            run_wave([make_req("both ranks")], [], [0])
        dummy = make_req("both ranks")
        dummy.is_dummy_run = lambda: True
        run_wave([dummy], [], [0])

        # Both local lookups succeed but disagree on checkpoint identity.
        real_digest = backend._model_digest
        backend._model_digest = f"checkpoint-{rank}"
        backend.store(make_req("both ranks"), final)
        run_wave([make_req("both ranks")], [], [0])
        backend._model_digest = real_digest

        # Equal prompt and similarity can conceal distinct source seeds. Tensor
        # contents are intentionally rank-local, as with a distributed shard.
        backend.clear()
        backend._clip_model = object()
        backend.encode_prompt = lambda prompt: torch.tensor([1.0, 0.0])
        steps = [StepLatentData(i, float(20 - i), torch.tensor([rank * 10.0 + i])) for i in range(3)]
        backend.store(make_req("source", seed=123 + rank), final, step_latents=steps)
        run_wave([make_req("query")], [], [0])

        backend.clear()
        with torch.inference_mode():
            source_hash = backend.store(make_req("source"), final, step_latents=steps)
        if rank == 1:
            # Only rank 1 has finished its background image embedding. Its
            # hybrid score is below threshold while rank 0 has a text-only hit.
            backend.cache_store.update_image_embedding(source_hash, torch.tensor([-1.0, 0.0]))
        run_wave([make_req("query")], [], [0])
        if rank == 1:
            backend.cache_store.update_image_embedding(source_hash, torch.tensor([1.0, 0.0]))
        run_wave([make_req("query")], [], [3])
    finally:
        backend.shutdown()


def _exercise_disk_restart(rank, device):
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    group = get_world_group().cpu_group
    with TemporaryDirectory() if rank == 0 else nullcontext() as root:
        roots = [root]
        dist.broadcast_object_list(roots, src=0, group=group)
        cfg = DiffusionCacheConfig(inter_request_persistent_cache_dir=roots[0])
        backend = InterRequestCacheBackend(cfg)
        backend.enable(FakePipeline())
        try:
            backend.store(make_req("persist"), torch.tensor([rank + 10.0]))
        finally:
            backend.shutdown()

        dist.barrier(group=group)
        restored = InterRequestCacheBackend(cfg)
        restored.enable(FakePipeline())
        try:
            hits, remaining = restored.short_circuit_requests([make_req("persist")], target_device=device)
            assert not remaining and len(hits) == 1
            assert hits[0][1].output.item() == rank + 10
        finally:
            restored.shutdown()
        dist.barrier(group=group)


def _cpu_worker(rank, init_method):
    from vllm_omni.diffusion.distributed import parallel_state

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        # Standalone CPU workers use the same Gloo group for votes and work.
        world = SimpleNamespace(cpu_group=dist.group.WORLD)
        with (
            patch.object(parallel_state, "get_world_group", return_value=world),
            patch.object(parallel_state, "get_data_parallel_rank", return_value=0),
        ):
            _exercise_decisions(rank, torch.device("cpu"))
            _exercise_disk_restart(rank, torch.device("cpu"))

        # Independent DP replicas may each have a different exact-hit source.
        # DLO still needs all replicas to take the same forward/skip path.
        with (
            patch.object(parallel_state, "get_world_group", return_value=world),
            patch.object(parallel_state, "get_data_parallel_rank", return_value=rank),
        ):
            backend = InterRequestCacheBackend(DiffusionCacheConfig())
            backend.enable(FakePipeline())
            try:
                req = make_req(prompt=f"replica-{rank}")
                backend.store(req, torch.tensor([rank + 10.0]))
                hits, remaining = backend.short_circuit_requests([req], target_device="cpu")
                assert not remaining and hits[0][1].output.item() == rank + 10
                if rank == 1:
                    backend.clear()
                hits, remaining = backend.short_circuit_requests([req], target_device="cpu")
                assert not hits and remaining == [req]
            finally:
                backend.shutdown()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs Gloo")
def test_two_rank_cache_decisions(tmp_path):
    torch.multiprocessing.spawn(_cpu_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)


if __name__ == "__main__":
    # Used by the bounded two-card acceptance run. The production initializer
    # builds the real diffusion control group and TP/DP topology.
    import argparse

    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=("npu", "cuda"))
    device_type = parser.parse_args().device
    if device_type == "npu":
        import torch_npu  # noqa: F401

    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("Run with exactly two ranks")
    device = torch.device(device_type, rank)
    getattr(torch, device_type).set_device(device)
    backend_name = "hccl" if device_type == "npu" else "nccl"
    dist.init_process_group(backend_name, timeout=timedelta(seconds=60))
    try:
        init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend=backend_name)
        initialize_model_parallel(tensor_parallel_size=2, data_parallel_size=1)
        _exercise_decisions(rank, device)
        _exercise_disk_restart(rank, device)
        if rank == 0:
            print("PASS: two-rank cache decisions, forward collectives and snapshot restart", flush=True)
    finally:
        dist.destroy_process_group()
