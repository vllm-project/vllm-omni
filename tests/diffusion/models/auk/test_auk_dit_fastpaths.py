# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AuK DiT fast paths keep the reference numerics.

Two paths are checked: the grouped conv position embedding as one batched
GEMM, and the adaLN modulations precomputed for a whole time grid, eager and
under the per-step CUDA graph.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.auk import auk_transformer as T
from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, build_time_grid
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_dit(device: str = "cpu") -> AuKTransformer:
    torch.manual_seed(5)
    return (
        AuKTransformer(
            dim=32,
            heads=2,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device)
    )


def _reference_conv_pos_embed(module: T.ConvPosEmbedding, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """The channels-first conv stack as the reference implementation writes it."""
    keep = None if mask is None else mask.unsqueeze(1)
    x = x.transpose(1, 2)
    if keep is not None:
        x = x.masked_fill(~keep, 0.0)
    for layer in module.conv1d:
        x = layer(x)
        if keep is not None and isinstance(layer, torch.nn.Conv1d):
            x = x.masked_fill(~keep, 0.0)
    return x.transpose(1, 2)


@torch.inference_mode()
def test_grouped_conv_as_batched_gemm_matches_conv1d() -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(64, 64, 31, groups=16, padding=15)
    x = torch.randn(2, 37, 64)
    expected = conv(x.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(T._grouped_conv1d_btc(x, conv), expected, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("masked", [False, True])
def test_conv_pos_embedding_matches_the_channels_first_stack(masked: bool) -> None:
    torch.manual_seed(1)
    module = T.ConvPosEmbedding(64).eval()
    x = torch.randn(2, 37, 64)
    mask = None
    if masked:
        mask = torch.ones(2, 37, dtype=torch.bool)
        mask[1, 30:] = False
    torch.testing.assert_close(module(x, mask), _reference_conv_pos_embed(module, x, mask), atol=1e-5, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("cfg_infer", [False, True])
def test_modulation_table_matches_per_step_modulation(cfg_infer: bool) -> None:
    dit = _make_dit()
    text = torch.randn(1, 5, 8)
    ref = torch.randn(1, 3, 4)
    x = torch.randn(1, 6, 4)
    grid = build_time_grid(nfe=4, sway_sampling_coef=-1.0, t_grid=None, device="cpu")
    plain = dit.prepare(text, target_len=6, ref=ref, cfg_infer=cfg_infer)
    tabled = dit.prepare(text, target_len=6, ref=ref, cfg_infer=cfg_infer, timesteps=grid[:-1])
    assert tabled.modulation is not None and tabled.modulation.shape[0] == 4
    for i in range(4):
        expected = dit.step(x, grid[i], plain)
        torch.testing.assert_close(dit.step(x, grid[i], tabled, step_index=i), expected, atol=1e-5, rtol=1e-5)
        index = torch.tensor(i)
        torch.testing.assert_close(dit.step(x, grid[i], tabled, step_index=index), expected, atol=1e-5, rtol=1e-5)


def _padded_inputs(device: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(3)
    text = torch.randn(1, 7, 8, generator=generator)
    c_mask = torch.ones(1, 7, dtype=torch.bool)
    c_mask[:, 5:] = False
    text[:, 5:] = 0.0
    ref = torch.randn(1, 6, 4, generator=generator)
    ref_mask = torch.ones(1, 6, dtype=torch.bool)
    ref_mask[:, 4:] = False
    mask = torch.ones(1, 9, dtype=torch.bool)
    mask[:, 7:] = False
    inputs = {"text": text, "c_mask": c_mask, "ref": ref, "ref_mask": ref_mask, "mask": mask}
    return {name: value.to(device) for name, value in inputs.items()}


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_graph_with_time_grid_matches_eager_with_time_grid(cfg_strength: float) -> None:
    dit = _make_dit("cuda")
    inputs = _padded_inputs("cuda")
    grid = build_time_grid(nfe=4, sway_sampling_coef=-1.0, t_grid=None, device="cuda")
    x = torch.randn(1, 9, 4, device="cuda")
    wrapper = AuKCUDAGraphWrapper(dit)
    common = dict(
        text=inputs["text"],
        c_mask=inputs["c_mask"],
        ref=inputs["ref"],
        ref_mask=inputs["ref_mask"],
        cfg_strength=cfg_strength,
        timesteps=grid[:-1],
    )
    for i in range(4):
        graph = wrapper(x=x, timestep=grid[i], new_request=i == 0, step_index=i, **common)
        ctx = dit.prepare(
            inputs["text"],
            target_len=9,
            c_mask=inputs["c_mask"],
            ref=inputs["ref"],
            ref_mask=inputs["ref_mask"],
            cfg_infer=cfg_strength >= 1e-5,
            timesteps=grid[:-1],
        )
        eager = wrapper._step(x, grid[i], ctx, cfg_strength, i)
        torch.testing.assert_close(graph, eager, atol=1e-4, rtol=1e-4)
    # One graph serves every step of the grid.
    assert len(wrapper._cache) == 1
    assert next(iter(wrapper._cache))[-1] == 4
