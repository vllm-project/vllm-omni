# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The opt-in FP8 GEMMs of the AuK DiT blocks.

FP8 E4M3 keeps three mantissa bits, so parity with the bf16 linears is a
tolerance on the relative error, not closeness per element. The checks cover
which linears are swapped, the numerics of one FP8 linear, saturation of
out-of-range activations, and the denoise step eager and under its CUDA graph.
"""

import pytest
import torch
from torch import nn
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, build_time_grid
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper
from vllm_omni.diffusion.models.auk.fp8_linear import Fp8Linear, fp8_supported, quantize_block_linears

pytestmark = [pytest.mark.core_model]

_needs_fp8 = pytest.mark.skipif(
    not (torch.cuda.is_available() and fp8_supported(torch.device("cuda"))),
    reason="FP8 GEMMs need an Ada or Hopper CUDA device",
)


def _make_dit(device: str = "cpu", dtype: torch.dtype = torch.float32) -> AuKTransformer:
    torch.manual_seed(5)
    return (
        AuKTransformer(
            dim=64,
            heads=4,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device=device, dtype=dtype)
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.cpu
def test_fp8_is_unsupported_off_cuda() -> None:
    assert not fp8_supported(torch.device("cpu"))


@pytest.mark.cpu
def test_only_the_token_wise_block_linears_are_swapped() -> None:
    dit = _make_dit()
    adaln_before = [module.linear for module in dit._adaln_modules()]

    # Two double blocks (two streams of qkv, out, ff in, ff out) and two single blocks.
    assert quantize_block_linears(dit) == 2 * 8 + 2 * 4
    for block in dit.transformer_blocks:
        assert isinstance(block.attn.to_qkv, Fp8Linear) and isinstance(block.attn.to_qkv_c, Fp8Linear)
        assert isinstance(block.attn.to_out[0], Fp8Linear) and isinstance(block.attn.to_out_c, Fp8Linear)
        assert isinstance(block.ff_x.linear_in, Fp8Linear) and isinstance(block.ff_c.linear_out, Fp8Linear)
    for block in dit.single_transformer_blocks:
        assert isinstance(block.attn.to_qkv, Fp8Linear) and isinstance(block.ff.linear_out, Fp8Linear)
    # The modulations, the embeddings and the output projection keep the model dtype.
    assert [module.linear for module in dit._adaln_modules()] == adaln_before
    for kept in (dit.proj_out, dit.txt_proj, dit.audio_embed.linear, *adaln_before):
        assert type(kept) is nn.Linear
    # A second pass finds nothing left to swap.
    assert quantize_block_linears(dit) == 0


@pytest.mark.cpu
def test_linears_the_fp8_gemm_cannot_take_are_left_alone() -> None:
    dit = _make_dit()
    block = dit.single_transformer_blocks[0]
    block.ff.linear_out = nn.Linear(block.ff.linear_out.in_features, 40, bias=False)

    quantize_block_linears(dit)

    assert type(block.ff.linear_out) is nn.Linear
    assert isinstance(block.ff.linear_in, Fp8Linear)


@pytest.mark.cpu
def test_fp8_weights_round_trip_within_the_format_precision() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 48)
    quantized = Fp8Linear(linear)

    assert quantized.weight.dtype == torch.float8_e4m3fn
    # Stored as W^T for the column-major operand of the GEMM.
    assert quantized.weight.shape == (64, 48)
    restored = quantized.weight.float().t() * quantized.weight_scale
    # Three mantissa bits round to within 1/16 of a value.
    assert _relative_error(restored, linear.weight) < 0.05
    torch.testing.assert_close(quantized.bias, linear.bias, atol=0, rtol=0)


@pytest.mark.cpu
def test_scales_stay_fp32_under_a_half_precision_default_dtype() -> None:
    # The diffusion loader builds the pipeline with the model dtype as the default dtype.
    with set_default_torch_dtype(torch.bfloat16):
        quantized = Fp8Linear(nn.Linear(64, 48))

    # The FP8 GEMM takes fp32 scales only.
    assert quantized.input_scale.dtype == torch.float32
    assert quantized.weight_scale.dtype == torch.float32
    assert quantized.out_dtype == torch.bfloat16


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_needs_fp8
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [True, False])
def test_fp8_linear_matches_the_model_dtype_linear(bias: bool, dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    linear = nn.Linear(64, 96, bias=bias).to("cuda", dtype)
    quantized = Fp8Linear(linear)
    x = torch.randn(2, 37, 64, device="cuda", dtype=dtype)

    # _scaled_mm only fuses the bias into half-precision outputs; fp32 adds it after the GEMM.
    out = quantized(x)

    expected = linear(x)
    assert out.shape == expected.shape and out.dtype == expected.dtype
    # Weights and activations each carry about 2.5% rounding noise: 3.7% measured.
    assert _relative_error(out, expected) < 0.06
    assert torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0) > 0.995
    if dtype == torch.bfloat16:
        # fp32 activations, as the layer norms hand them over under autocast, take the same path.
        torch.testing.assert_close(quantized(x.float()), out, atol=0, rtol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_needs_fp8
@torch.inference_mode()
def test_fp8_step_runs_with_an_fp32_model() -> None:
    # The QKV and attention output projections carry a bias, which an fp32 output cannot fuse.
    reference = _make_dit("cuda", torch.float32)
    dit = _make_dit("cuda", torch.float32)
    assert quantize_block_linears(dit) == 24

    torch.manual_seed(3)
    text = torch.randn(1, 32, 8, device="cuda")
    c_mask = torch.ones(1, 32, dtype=torch.bool, device="cuda")
    ref = torch.randn(1, 50, 4, device="cuda")
    ref_mask = torch.ones(1, 50, dtype=torch.bool, device="cuda")
    x = torch.randn(1, 64, 4, device="cuda")
    grid = build_time_grid(nfe=4, sway_sampling_coef=-1.0, t_grid=None, device="cuda")

    def velocity(model: AuKTransformer) -> torch.Tensor:
        ctx = model.prepare(
            text, target_len=64, c_mask=c_mask, ref=ref, ref_mask=ref_mask, cfg_infer=True, timesteps=grid[:-1]
        )
        return model.step(x, grid[0], ctx, step_index=0)

    fp8 = velocity(dit)
    assert fp8.dtype == torch.float32 and torch.isfinite(fp8).all()
    assert _relative_error(fp8, velocity(reference)) < 0.05


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_needs_fp8
@torch.inference_mode()
def test_out_of_range_activations_saturate() -> None:
    torch.manual_seed(2)
    linear = nn.Linear(64, 64, bias=False).to("cuda", torch.bfloat16)
    quantized = Fp8Linear(linear)
    x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    x[0, 0], x[1, 3] = 5000.0, -9000.0

    out = quantized(x)

    assert torch.isfinite(out).all()
    limit = torch.finfo(torch.float8_e4m3fn).max
    assert _relative_error(out, linear(x.clamp(-limit, limit))) < 0.06


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_needs_fp8
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_fp8_step_tracks_bf16_and_replays_under_the_cuda_graph(cfg_strength: float) -> None:
    reference = _make_dit("cuda", torch.bfloat16)
    dit = _make_dit("cuda", torch.bfloat16)
    assert quantize_block_linears(dit) == 24
    wrapper = AuKCUDAGraphWrapper(dit, enabled=True)

    torch.manual_seed(3)
    text = torch.randn(1, 32, 8, device="cuda", dtype=torch.bfloat16)
    c_mask = torch.ones(1, 32, dtype=torch.bool, device="cuda")
    ref = torch.randn(1, 50, 4, device="cuda")
    ref_mask = torch.ones(1, 50, dtype=torch.bool, device="cuda")
    x = torch.randn(1, 64, 4, device="cuda")
    grid = build_time_grid(nfe=4, sway_sampling_coef=-1.0, t_grid=None, device="cuda")
    guided = cfg_strength >= 1e-5

    def eager(model: AuKTransformer, step: int) -> torch.Tensor:
        ctx = model.prepare(
            text, target_len=64, c_mask=c_mask, ref=ref, ref_mask=ref_mask, cfg_infer=guided, timesteps=grid[:-1]
        )
        velocity = model.step(x, grid[step], ctx, step_index=step)
        if not guided:
            return velocity
        conditional, unconditional = velocity.chunk(2, dim=0)
        return conditional + (conditional - unconditional) * cfg_strength

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for step in (0, 2):
            fp8 = eager(dit, step)
            assert torch.isfinite(fp8).all()
            # The residual stream dominates this small random model: 0.6% unguided, 1.9% guided measured.
            assert _relative_error(fp8, eager(reference, step)) < 0.05
            replay = wrapper(
                x=x,
                text=text,
                c_mask=c_mask,
                ref=ref,
                ref_mask=ref_mask,
                timestep=grid[step],
                cfg_strength=cfg_strength,
                new_request=step == 0,
                timesteps=grid[:-1],
                step_index=step,
            )
            # The graph replays the same FP8 kernels as the eager step.
            torch.testing.assert_close(replay.float(), fp8.float(), atol=1e-3, rtol=1e-3)
    assert len(wrapper._cache) == 1
