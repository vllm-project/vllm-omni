# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    CFM,
    Aggregator,
    CFMGraphExecutor,
    CFMGraphExecutorPool,
    DiT,
)

torch = pytest.importorskip("torch")
pytest.importorskip("x_transformers")

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for graph capture"),
]

_LATENT_DIM = 8
_PATCH_SIZE = 4
_HIS_PATCH_SIZE = 8
_LLM_HIDDEN = 16
_DIT_HIDDEN = 32
_AGG_HIDDEN = 32
_NUM_HEADS = 4
_DEPTH = 2
_STEPS = 5
_DTYPE = torch.float32


def _warmup_pipeline(cfm: CFM, aggregator: Aggregator, stop_head: torch.nn.Linear, device: torch.device) -> None:
    llm_cond = torch.randn(1, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
    lat_cond = torch.randn(1, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)
    y0 = torch.randn(1, _PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)
    t = torch.linspace(0.0, 1.0, _STEPS + 1, device=device, dtype=_DTYPE)
    sde_args = torch.tensor([2.0, 0.25, 0.0], device=device, dtype=_DTYPE)
    sde_rnd = torch.randn(_STEPS, 1, _PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)

    with torch.no_grad():
        gen_lat = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)
        aggregator(gen_lat)
        stop_head(llm_cond[:, -1, :]).softmax(dim=-1)
    torch.accelerator.synchronize(device)


def _build_pipeline():
    device = torch.device("cuda")
    dit = (
        DiT(
            in_channels=_LATENT_DIM,
            hidden_size=_DIT_HIDDEN,
            depth=_DEPTH,
            num_heads=_NUM_HEADS,
            mlp_ratio=2.0,
            llm_cond_dim=_LLM_HIDDEN,
        )
        .to(device=device, dtype=_DTYPE)
        .eval()
    )
    cfm = CFM(dit, steps=_STEPS, sway_sampling_coef=-1.0).to(device=device, dtype=_DTYPE).eval()
    aggregator = (
        Aggregator(
            in_channels=_LATENT_DIM,
            hidden_size=_AGG_HIDDEN,
            depth=_DEPTH,
            num_heads=_NUM_HEADS,
            mlp_ratio=2.0,
            llm_input_dim=_LLM_HIDDEN,
        )
        .to(device=device, dtype=_DTYPE)
        .eval()
    )
    stop_head = torch.nn.Linear(_LLM_HIDDEN, 2).to(device=device, dtype=_DTYPE).eval()

    config = SimpleNamespace(steps=_STEPS, patch_size=_PATCH_SIZE)
    _warmup_pipeline(cfm, aggregator, stop_head, device)
    return config, cfm, aggregator, stop_head, device


class TestCFMGraphExecutor:
    """Capture once, replay twice: outputs must stay consistently-shaped."""

    def test_execute_shapes_and_replay(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()
        executor = CFMGraphExecutor(config, cfm, aggregator, stop_head)

        bsz = 1
        input_tensor = torch.randn(bsz, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        his_lat = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)

        gen_lat, inputs_embeds, stop_out = executor.execute(input_tensor, his_lat)
        torch.accelerator.synchronize()

        assert gen_lat.shape == (bsz, _PATCH_SIZE, _LATENT_DIM)
        assert inputs_embeds.shape == (bsz, 1, _LLM_HIDDEN)
        assert stop_out.shape == (bsz, 2)
        assert torch.isfinite(gen_lat).all()
        assert torch.isfinite(inputs_embeds).all()
        # stop_head output is softmax-normalized across the last dim.
        assert torch.allclose(stop_out.sum(dim=-1), torch.ones(bsz, device=device, dtype=_DTYPE), atol=1e-4)

        # Replay the captured graph with fresh inputs — shapes must match.
        new_input = torch.randn_like(input_tensor)
        new_his = torch.randn_like(his_lat)
        gen_lat2, inputs_embeds2, stop_out2 = executor.execute(new_input, new_his)
        torch.accelerator.synchronize()
        assert gen_lat2.shape == gen_lat.shape
        assert inputs_embeds2.shape == inputs_embeds.shape
        assert stop_out2.shape == stop_out.shape
        assert executor.initialized is True

    def test_execute_is_noninplace_on_inputs(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()
        executor = CFMGraphExecutor(config, cfm, aggregator, stop_head)

        input_tensor = torch.randn(1, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        his_lat = torch.randn(1, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)
        snapshot_input = input_tensor.clone()
        snapshot_his = his_lat.clone()

        executor.execute(input_tensor, his_lat)
        torch.accelerator.synchronize()
        assert torch.equal(input_tensor, snapshot_input)
        assert torch.equal(his_lat, snapshot_his)


class TestCFMGraphExecutorPool:
    def test_pool_acquires_and_releases(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()
        pool = CFMGraphExecutorPool(config, cfm, aggregator, stop_head, pool_size=2)

        input_tensor = torch.randn(1, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        his_lat = torch.randn(1, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)

        gen_lat, inputs_embeds, stop_out = pool.execute(input_tensor, his_lat)
        torch.accelerator.synchronize()
        assert gen_lat.shape == (1, _PATCH_SIZE, _LATENT_DIM)
        assert inputs_embeds.shape == (1, 1, _LLM_HIDDEN)
        assert stop_out.shape == (1, 2)
        assert pool.pool.qsize() == 2
