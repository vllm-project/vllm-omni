# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Denoise-graph runners against real device graphs (CUDA graphs / XPU graphs)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.graph import build_denoise_graph_runner, invalidate_denoise_graphs
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@torch.inference_mode()
def test_step_replay_matches_eager_across_shapes_and_invalidation():
    device = torch.device(current_omni_platform.device_type, 0)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)).to(device, torch.bfloat16)

    def step(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x) * t

    od_config = SimpleNamespace(
        diffusion_use_graph=True,
        enforce_eager=False,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        cache_backend="none",
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    runner = build_denoise_graph_runner(od_config, step, "device")
    assert runner is not None

    # Three shapes through the two-graph LRU, then back to the first; then the
    # LoRA-style invalidation, after which the next step recaptures.
    for rows in (64, 96, 128, 64, 64):
        x = torch.randn(rows, 256, device=device, dtype=torch.bfloat16)
        t = torch.rand(1, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(runner(x=x, t=t), step(x=x, t=t), rtol=0, atol=0)
    invalidate_denoise_graphs("test")
    assert runner.num_graphs == 0
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    t = torch.rand(1, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(runner(x=x, t=t), step(x=x, t=t), rtol=0, atol=0)

    stats = runner.stats()
    assert runner.enabled
    assert (stats["captures"], stats["eager_calls"]) == (5, 0)
