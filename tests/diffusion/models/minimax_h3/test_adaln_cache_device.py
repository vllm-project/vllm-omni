# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real vLLM projection/collective parity without loading an H3 checkpoint."""

from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


def _projection_worker(rank, world_size, rendezvous):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.models.minimax_h3.adaln_cache import MiniMaxH3RuntimeAdalnCache
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3AdalnProj,
        MiniMaxH3DiTArchConfig,
    )

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{rendezvous}",
            backend="nccl",
            timeout=timedelta(seconds=45),
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        try:
            arch = MiniMaxH3DiTArchConfig(hidden_size=256, time_embed_dim=128)
            cache = MiniMaxH3RuntimeAdalnCache()
            generator = torch.Generator(device=device).manual_seed(11)
            for name, expansion, modalities in (("blocks.0", 6, 3), ("final_layer", 2, 1)):
                cache.clear()
                projection = MiniMaxH3AdalnProj(
                    arch,
                    arch.hidden_size * expansion * modalities,
                    None,
                    expand_ratio=expansion,
                    modality_num=modalities,
                    prefix=name + ".adaln_proj",
                    adaln_cache=cache,
                ).to(device)
                with torch.no_grad():
                    for parameter in projection.parameters():
                        parameter.normal_(std=0.05, generator=generator)
                with torch.inference_mode():
                    for width in (1, 2, 3, 4):
                        embedding = torch.randn(width, arch.time_embed_dim, device=device, generator=generator)
                        # No prepared input: run the original vLLM projector.
                        cache.clear()
                        expected = projection(embedding)
                        cache.prepare(embedding)
                        first = projection(embedding)
                        hits = cache.hits
                        second = projection(embedding)
                        assert cache.hits == hits + 1
                        for reference, cold, warm in zip(expected, first, second, strict=True):
                            assert torch.equal(reference, cold)
                            assert torch.equal(reference, warm)

                        # Only one rank loses its entry. All ranks must execute
                        # the gather again; a local-only hit would deadlock.
                        if rank == world_size - 1:
                            cache.clear()
                        cache.prepare(embedding)
                        misses = cache.misses
                        rebuilt = projection(embedding)
                        assert cache.misses == misses + 1
                        for reference, actual in zip(expected, rebuilt, strict=True):
                            assert torch.equal(reference, actual)

                        if rank == world_size - 1:
                            projection.linear.weight.add_(0.01)
                        misses = cache.misses
                        changed = projection(embedding)
                        assert cache.misses == misses + 1
                        cache.clear()
                        reference = projection(embedding)
                        for actual, original in zip(changed, reference, strict=True):
                            assert torch.equal(actual, original)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(1, marks=hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1)),
        pytest.param(2, marks=hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=2)),
    ],
)
def test_real_projection_cache_parity_and_rank_local_invalidation(tmp_path, world_size):
    if not current_omni_platform.is_cuda():
        pytest.skip("Requires CUDA")
    context = mp.spawn(
        _projection_worker,
        args=(world_size, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        join=False,
    )
    try:
        if not context.join(timeout=90):
            # ProcessContext may reap one successful worker per join.
            if not context.join(timeout=30):
                pytest.fail("AdaLN projection workers timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
