# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5's fused Triton kernels against the eager ops they replace, one kernel at a time.

Each kernel runs on the same inputs as its eager counterpart, in both serving
dtypes and in the deployed mixed-dtype layout: bfloat16 GEMM weights, K/V and
GEMM inputs, with the float32 residual stream, norms and head that
``_to_bfloat16_for_inference`` keeps. The shapes are the deployed ones: a
968-token prefix on the Gemma 2B widths and a 50-token action chunk on the
Gemma 300M ones, with 8 query heads, 1 KV head and head_dim 256.

The kernels perform eager's operations in its order and round where it
rounds, so what a test expects depends on whether a kernel reorders a
reduction:

* It does not: bit-exact. That is the GELU, the residual sums, and every
  epilogue a GEMM feeds (RoPE and the K/V slot writes, the gated residual, the
  GELU product) when the GEMM is cuBLAS's: float32 serves that way, and
  bfloat16 is run that way here too to pin its rounding points.
* It does (a Triton GEMM's accumulation, the RMS variance, softmax sums): the
  result may differ where the reordering flips a rounding, and must be as
  accurate as eager against a float64 evaluation of the same math: within
  1.25x eager's relative L2 error and 1.5x its largest error. Eager's cuBLAS
  split-K GEMMs round twice in bfloat16, so the fused path is often the more
  accurate one.

Needs a CUDA GPU; no checkpoint::

    python -m pytest tests/diffusion/models/pi05/test_pi05_fused_kernels.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from vllm_omni.diffusion.models.pi05 import modeling_pi05 as M

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The fused kernels need a CUDA GPU."),
    pytest.mark.skipif(not M.HAS_TRITON, reason="The fused kernels need Triton."),
]

DTYPES = [torch.float32, torch.bfloat16]
DEVICE = "cuda"
HEADS, HEAD_DIM = 8, 256
# (tokens, width, MLP width): the prefix on Gemma 2B, the action chunk on Gemma 300M.
SHAPES = {"prefix": (968, 2048, 16384), "suffix": (50, 1024, 4096)}


def _dtype_id(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _randn(*shape, dtype=torch.float32, scale=1.0, seed=0) -> torch.Tensor:
    return (torch.randn(*shape, device=DEVICE, generator=_generator(seed)) * scale).to(dtype)


def _gemma_config(width: int, mlp_width: int) -> GemmaConfig:
    return GemmaConfig(
        hidden_size=width,
        intermediate_size=mlp_width,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        hidden_act="gelu_pytorch_tanh",
        num_hidden_layers=1,
    )


def _randomize(module: nn.Module, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.data = torch.randn(param.shape, generator=generator) * param.shape[-1] ** -0.5
    return module


def _proj_dtype_module(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """GEMM weights in the serving dtype, as ``_set_inference_dtype`` lays them out."""
    return module.to(device=DEVICE, dtype=dtype)


def _assert_as_accurate_as_eager(actual: torch.Tensor, expected: torch.Tensor, reference: torch.Tensor) -> None:
    """``actual`` (fused) is as close to the float64 ``reference`` as ``expected`` (eager) is."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    reference = reference.double()

    def errors(t):
        diff = t.double() - reference
        return (diff.norm() / reference.norm()).item(), diff.abs().max().item()

    fused_rel, fused_max = errors(actual)
    eager_rel, eager_max = errors(expected)
    assert fused_rel <= 1.25 * eager_rel and fused_max <= 1.5 * eager_max, (
        f"fused error vs float64: rel {fused_rel:.3e}, max {fused_max:.3e}; "
        f"eager: rel {eager_rel:.3e}, max {eager_max:.3e}"
    )


def _assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    mismatches = int((actual != expected).sum())
    assert mismatches == 0, f"{mismatches} of {actual.numel()} elements differ from eager"


def _prefix_pad_masks(seq_len: int) -> torch.Tensor:
    """One real camera, two missing ones, then 120 live text tokens: ``embed_prefix``'s layout."""
    pad = torch.zeros(1, seq_len, dtype=torch.bool, device=DEVICE)
    pad[:, :256] = True
    pad[:, 768 : 768 + 120] = True
    return pad


# ── Elementwise: bit-exact ─────────────────────────────────────────────
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_gelu_mul_is_bit_exact(dtype):
    """The prefix MLP's ``act_fn(gate) · up`` between its cuBLAS GEMMs."""
    tokens, width, mlp_width = SHAPES["prefix"]
    mlp = GemmaMLP(_gemma_config(width, mlp_width))
    gate = _randn(1, tokens, mlp_width, dtype=dtype, scale=3.0, seed=1)
    up = _randn(1, tokens, mlp_width, dtype=dtype, seed=2)

    expected = mlp.act_fn(gate) * up
    actual = torch.empty_like(gate)
    M._fused_gelu_mul(gate, up, actual)
    _assert_bit_exact(actual, expected)

    M._fused_gelu_mul(gate, up, gate)  # in place, as the prefix runs it
    _assert_bit_exact(gate, expected)


# ── Norms: the RMS variance is a reordered reduction ──────────────────
def _rms_norm_reference(x: torch.Tensor, eps: float, weight=None, scale=None, shift=None) -> torch.Tensor:
    x = x.double()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return normed * (1.0 + weight.double())
    return normed * (1.0 + scale.double()) + shift.double()


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("residual_add", [False, True], ids=["norm", "residual_add_norm"])
def test_rms_norm_matches_gemma_rms_norm(dtype, residual_add):
    """``_match(norm([sublayer_out +] residual), next_proj)`` in the prefix: the
    residual sum is bit-exact, the norm as accurate as eager's."""
    tokens, width, _ = SHAPES["prefix"]
    norm = _randomize(GemmaRMSNorm(width), seed=3).to(DEVICE)  # float32 in both layouts
    residual = _randn(tokens, width, scale=4.0, seed=4)
    sublayer_out = _randn(tokens, width, dtype=dtype, scale=2.0, seed=5) if residual_add else None

    summed = sublayer_out + residual if residual_add else residual
    expected = norm(summed).to(dtype)

    actual = torch.empty(tokens, width, dtype=dtype, device=DEVICE)
    summed_out = torch.full_like(residual, float("nan")) if residual_add else None
    M._fused_rms_norm(residual, norm, actual, add=sublayer_out, res_out=summed_out)
    if residual_add:
        _assert_bit_exact(summed_out, summed)
    _assert_as_accurate_as_eager(actual, expected, _rms_norm_reference(summed, norm.eps, weight=norm.weight))


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_final_rms_norm_keeps_the_residual_dtype(dtype):
    """The prefix's final norm returns the residual stream's dtype, as eager's does."""
    tokens, width, _ = SHAPES["prefix"]
    norm = _randomize(GemmaRMSNorm(width), seed=6).to(DEVICE)
    residual = _randn(tokens, width, scale=4.0, seed=7)
    mlp_out = _randn(tokens, width, dtype=dtype, seed=8)

    summed = mlp_out + residual
    expected = norm(summed)
    actual = torch.empty_like(residual)
    M._fused_rms_norm(residual, norm, actual, add=mlp_out)
    _assert_as_accurate_as_eager(actual, expected, _rms_norm_reference(summed, norm.eps, weight=norm.weight))


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_adarms_norm_matches_pi05_adarms_norm(dtype):
    """The action expert's ``_match(Pi05AdaRMSNorm(x, cond)[0], next_proj)``."""
    tokens, width, _ = SHAPES["suffix"]
    norm = _randomize(M.Pi05AdaRMSNorm(width, cond_dim=width), seed=9).to(DEVICE)
    x = _randn(tokens, width, scale=4.0, seed=10)
    cond = _randn(1, width, seed=11)

    expected = norm(x[None], cond)[0][0].to(dtype)
    modulation = M._adarms_modulation(norm, cond)
    actual = torch.empty(tokens, width, dtype=dtype, device=DEVICE)
    M._fused_rms_norm(x, norm, actual, modulation=modulation)
    scale, shift, _ = modulation.chunk(3)
    _assert_as_accurate_as_eager(actual, expected, _rms_norm_reference(x, norm.eps, scale=scale, shift=shift))


# ── GEMMs with their epilogues ─────────────────────────────────────────
@pytest.fixture(params=["cublas", "triton"])
def gemm(request, monkeypatch):
    """Which GEMM feeds the epilogues: cuBLAS (float32's, and bfloat16 forced
    onto it to pin its rounding points) or the Triton one bfloat16 serves with."""
    if request.param == "cublas":
        monkeypatch.setattr(M, "_TRITON_GEMM_DTYPES", ())
    return request.param


def _check_gemm_epilogue(gemm: str, dtype: torch.dtype, actual, expected, reference) -> None:
    if gemm == "cublas":
        _assert_bit_exact(actual, expected)
    else:
        _assert_as_accurate_as_eager(actual, expected, reference)


def _skip_unserved(gemm: str, dtype: torch.dtype) -> None:
    if gemm == "triton" and dtype not in M._TRITON_GEMM_DTYPES:
        pytest.skip(f"{dtype} GEMMs stay on cuBLAS")


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("part", SHAPES, ids=str)
def test_qkv_rope_matches_eager(gemm, dtype, part):
    """``q/k/v_proj`` then ``apply_rotary_pos_emb`` with the tower's RoPE table;
    K and V land in their cache rows and nowhere else."""
    _skip_unserved(gemm, dtype)
    tokens, width, mlp_width = SHAPES[part]
    config = _gemma_config(width, mlp_width)
    attn = _proj_dtype_module(_randomize(GemmaAttention(config, layer_idx=0), seed=12), dtype)
    rotary = GemmaRotaryEmbedding(config).to(DEVICE)
    x = _randn(tokens, width, dtype=dtype, seed=13)
    if part == "prefix":
        position_ids = torch.cumsum(_prefix_pad_masks(tokens), dim=1) - 1
    else:
        position_ids = 376 + torch.arange(tokens, device=DEVICE)[None]

    shape = (1, tokens, -1, HEAD_DIM)
    q = attn.q_proj(x[None]).view(shape).transpose(1, 2)
    k = attn.k_proj(x[None]).view(shape).transpose(1, 2)
    v = attn.v_proj(x[None]).view(shape).transpose(1, 2)
    # Eager's table, in its dtype: an input of the kernel, not part of its math.
    cos, sin = rotary(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    x64 = x.double()
    q64 = (x64 @ attn.q_proj.weight.double().t()).view(shape).transpose(1, 2)
    k64 = (x64 @ attn.k_proj.weight.double().t()).view(shape).transpose(1, 2)
    q64, k64 = apply_rotary_pos_emb(q64, k64, cos.double(), sin.double(), unsqueeze_dim=1)
    v64 = x64 @ attn.v_proj.weight.double().t()

    offset = 7
    key_cache = torch.full((offset + tokens + 5, HEAD_DIM), float("nan"), dtype=dtype, device=DEVICE)
    value_cache = torch.full_like(key_cache, float("nan"))
    q_out = torch.empty(HEADS, tokens, HEAD_DIM, dtype=dtype, device=DEVICE)
    rows = slice(offset, offset + tokens)
    M._fused_qkv_rope(x, attn, cos[0], sin[0], q_out, key_cache[rows], value_cache[rows])

    for cache in (key_cache, value_cache):
        assert torch.isnan(cache[:offset]).all() and torch.isnan(cache[offset + tokens :]).all()
    _check_gemm_epilogue(gemm, dtype, q_out, q[0].contiguous(), q64[0])
    _check_gemm_epilogue(gemm, dtype, key_cache[rows], k[0, 0], k64[0, 0])
    _check_gemm_epilogue(gemm, dtype, value_cache[rows], v[0, 0], v64)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("gated", [True, False], ids=["gated", "plain"])
@pytest.mark.parametrize("n_in", [2048, 4096], ids=["o_proj", "down_proj"])
def test_linear_residual_matches_gated_residual(gemm, dtype, gated, n_in):
    """``_gated_residual(residual, linear(x), gate)``, the float32 residual stream
    updated in place as the fused denoising step runs it."""
    _skip_unserved(gemm, dtype)
    tokens, width, _ = SHAPES["suffix"]
    linear = _proj_dtype_module(_randomize(nn.Linear(n_in, width, bias=False), seed=17), dtype)
    x = _randn(tokens, n_in, dtype=dtype, seed=18)
    residual = _randn(tokens, width, scale=4.0, seed=19)
    gate = _randn(width, seed=20) if gated else None

    expected = M._gated_residual(residual, linear(x[None])[0], gate)
    projection64 = x.double() @ linear.weight.double().t()
    reference = residual.double() + (gate.double() * projection64 if gated else projection64)
    actual = residual.clone()
    M._fused_linear_residual(x, linear, actual, actual, gate=gate)
    _check_gemm_epilogue(gemm, dtype, actual, expected, reference)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_linear_gelu_mul_matches_gemma_mlp_input(gemm, dtype):
    """``act_fn(gate_proj(x)) · up_proj(x)``, the action expert's MLP up to ``down_proj``."""
    _skip_unserved(gemm, dtype)
    tokens, width, mlp_width = SHAPES["suffix"]
    mlp = _proj_dtype_module(_randomize(GemmaMLP(_gemma_config(width, mlp_width)), seed=21), dtype)
    x = _randn(tokens, width, dtype=dtype, seed=22)

    expected = (mlp.act_fn(mlp.gate_proj(x[None])) * mlp.up_proj(x[None]))[0]
    x64 = x.double()
    reference = torch.nn.functional.gelu(x64 @ mlp.gate_proj.weight.double().t(), approximate="tanh") * (
        x64 @ mlp.up_proj.weight.double().t()
    )
    actual = torch.empty_like(expected)
    M._fused_linear_gelu_mul(x, mlp, actual)
    _check_gemm_epilogue(gemm, dtype, actual, expected, reference)


# ── Reordered reductions: attention and the output head ────────────────
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("part", SHAPES, ids=str)
def test_attention_matches_eager_attend(dtype, part):
    """``_attend`` over one KV head with eager's float mask: the prefix's
    bidirectional mask with padded (fully masked) query rows, and a denoising
    step's queries over the ``[prefix | suffix]`` cache row."""
    tokens = SHAPES[part][0]
    prefix_pad = _prefix_pad_masks(SHAPES["prefix"][0])
    if part == "prefix":
        n_keys = tokens
        att_2d = M.make_att_2d_masks(prefix_pad, torch.zeros_like(prefix_pad))
    else:
        n_keys = prefix_pad.shape[1] + tokens
        suffix_pad = torch.ones(1, tokens, dtype=torch.bool, device=DEVICE)
        suffix_att = torch.zeros(1, tokens, device=DEVICE)
        suffix_att[:, 0] = 1
        att_2d = torch.cat(
            [prefix_pad[:, None, :].expand(1, tokens, -1), M.make_att_2d_masks(suffix_pad, suffix_att)], dim=2
        )
    mask = M.prepare_attention_masks_4d(att_2d)
    q = _randn(1, HEADS, tokens, HEAD_DIM, dtype=dtype, scale=2.0, seed=14)
    key = _randn(n_keys, HEAD_DIM, dtype=dtype, scale=2.0, seed=15)
    value = _randn(n_keys, HEAD_DIM, dtype=dtype, seed=16)
    scaling = 1.0 / math.sqrt(HEAD_DIM)

    def attend(q, key, value, mask):
        out = M._attend(q, key[None, None], value[None, None], mask, num_kv_groups=HEADS, scaling=scaling)
        return out.transpose(1, 2).reshape(tokens, HEADS * HEAD_DIM)

    expected = attend(q, key, value, mask)
    reference = attend(q.double(), key.double(), value.double(), mask.double())
    actual = torch.empty_like(expected)
    scores = torch.empty(HEADS * tokens, n_keys, dtype=dtype, device=DEVICE)
    M._fused_attention(q[0], key, value, mask[0, 0], scaling, scores, actual)
    _assert_as_accurate_as_eager(actual, expected, reference)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_final_head_matches_norm_and_action_out_proj(dtype):
    """The expert's final AdaRMS norm and ``action_out_proj``, both float32 in
    either layout, so ``dtype`` only names the layout."""
    tokens, width, _ = SHAPES["suffix"]
    norm = _randomize(M.Pi05AdaRMSNorm(width, cond_dim=width), seed=23).to(DEVICE)
    head = _randomize(nn.Linear(width, 32), seed=24).to(DEVICE)
    x = _randn(tokens, width, scale=4.0, seed=25)
    cond = _randn(1, width, seed=26)

    normed, _ = norm(x[None], cond)
    expected = head(normed.to(head.weight.dtype))[0]
    modulation = M._adarms_modulation(norm, cond)
    scale, shift, _ = modulation.chunk(3)
    normed64 = _rms_norm_reference(x, norm.eps, scale=scale, shift=shift)
    reference = normed64 @ head.weight.double().t() + head.bias.double()
    actual = torch.empty_like(expected)
    M._fused_final_head(x, norm, modulation, head, actual)
    _assert_as_accurate_as_eager(actual, expected, reference)
