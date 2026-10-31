# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5's fused Triton kernels against the eager ops they replace, one kernel at a time.

Each kernel runs on the same inputs as its eager counterpart, in both serving
dtypes and in the deployed mixed-dtype layout: bfloat16 GEMM weights, K/V and
GEMM inputs, with the float32 residual stream, norms and head that
``_to_bfloat16_for_inference`` keeps. The shapes are the deployed ones: a
968-token prefix on the Gemma 2B widths and a 50-token action chunk on the
Gemma 300M ones, with 8 query heads, 1 KV head and head_dim 256. The action
expert's GEMM kernels also run a 256-token chunk: every chunk length uses the
same tiles.

The kernels perform eager's operations in its order and round where it
rounds, so what a test expects depends on whether a kernel reorders a
reduction:

* It does not: bit-exact. That is the GELU, the residual sums, and every
  epilogue a GEMM feeds (RoPE and the K/V slot writes, the gated residual, the
  GELU product) when the GEMM is cuBLAS's: float32 and the bfloat16 prefix
  serve that way, and the bfloat16 action expert is run that way here too to
  pin its rounding points.
* It does (a Triton GEMM's accumulation, the RMS variance, softmax sums): the
  result may differ where the reordering flips a rounding, and must be as
  accurate as eager against a float64 evaluation of the same math: within
  1.25x eager's relative L2 error and 1.5x its largest error. Eager's cuBLAS
  split-K GEMMs round twice in bfloat16, so the fused path is often the more
  accurate one.

Needs a CUDA GPU; no checkpoint. ``-s`` prints each fused kernel's max absolute
and relative L2 error against its eager op::

    python -m pytest tests/diffusion/models/pi05/test_pi05_fused_kernels.py -v -s
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

from vllm_omni.diffusion.models.pi05 import exact_qk_pi05 as qk_kernels
from vllm_omni.diffusion.models.pi05 import modeling_pi05 as M
from vllm_omni.diffusion.models.pi05 import precision_kernels_pi05 as kernels

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
# Action-chunk lengths for the GEMM kernels: the deployed one and a long one. A
# chunk of 256 tokens or more once looked up tiles that did not exist.
CHUNKS = [SHAPES["suffix"][0], 256]


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


_DTYPE_NAMES = {torch.float32: "fp32", torch.bfloat16: "bf16"}
_MODULATION_NOTE = "The scale, shift and gate modulation is still computed by the torch Linear norm.dense, as in eager."


def _print_vs_eager(actual, expected, kernel: str, eager: str, dtype: torch.dtype, note: str) -> None:
    """Print the fused kernel's distance from its eager op (shown with ``-s``)."""
    diff = actual.double() - expected.double()
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / expected.double().norm()).item()
    print(
        f"\nFused {kernel} kernel versus eager {eager} (dtype={_DTYPE_NAMES[dtype]}): "
        f"max absolute error = {max_abs:.3e}; relative L2 error = {rel_l2:.3e}." + (f" {note}" if note else "")
    )


def _assert_as_accurate_as_eager(
    actual: torch.Tensor,
    expected: torch.Tensor,
    reference: torch.Tensor,
    kernel: str,
    eager: str,
    dtype: torch.dtype,
    note: str = "",
) -> None:
    """``actual`` (fused) is as close to the float64 ``reference`` as ``expected`` (eager) is."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    _print_vs_eager(actual, expected, kernel, eager, dtype, note)
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


def _assert_bit_exact(
    actual: torch.Tensor, expected: torch.Tensor, kernel: str, eager: str, dtype: torch.dtype, note: str = ""
) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    _print_vs_eager(actual, expected, kernel, eager, dtype, note)
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
    _assert_bit_exact(actual, expected, "GELU-multiply", "GemmaMLP act_fn(gate) * up", dtype)

    M._fused_gelu_mul(gate, up, gate)  # in place, as the prefix runs it
    _assert_bit_exact(gate, expected, "in-place GELU-multiply", "GemmaMLP act_fn(gate) * up", dtype)


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
    """``_match(norm([sublayer_out +] residual), next_proj)`` at the prefix's
    shape (the served prefix keeps eager's norm): the residual sum is
    bit-exact, the norm as accurate as eager's."""
    tokens, width, _ = SHAPES["prefix"]
    norm = _randomize(GemmaRMSNorm(width), seed=3).to(DEVICE)  # float32 in both layouts
    residual = _randn(tokens, width, scale=4.0, seed=4)
    sublayer_out = _randn(tokens, width, dtype=dtype, scale=2.0, seed=5) if residual_add else None

    summed = sublayer_out + residual if residual_add else residual
    expected = norm(summed).to(dtype)

    actual = torch.empty(tokens, width, dtype=dtype, device=DEVICE)
    summed_out = torch.full_like(residual, float("nan")) if residual_add else None
    M._fused_rms_norm(residual, norm, actual, add=sublayer_out, res_out=summed_out)
    kernel = "residual-add RMSNorm" if residual_add else "RMSNorm"
    if residual_add:
        _assert_bit_exact(summed_out, summed, kernel, "sublayer_out + residual", dtype)
    _assert_as_accurate_as_eager(
        actual, expected, _rms_norm_reference(summed, norm.eps, weight=norm.weight), kernel, "GemmaRMSNorm", dtype
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
def test_final_rms_norm_keeps_the_residual_dtype(dtype):
    """The final Gemma norm at the prefix's shape returns the residual stream's
    dtype, as eager's does (the served prefix keeps eager's final norm)."""
    tokens, width, _ = SHAPES["prefix"]
    norm = _randomize(GemmaRMSNorm(width), seed=6).to(DEVICE)
    residual = _randn(tokens, width, scale=4.0, seed=7)
    mlp_out = _randn(tokens, width, dtype=dtype, seed=8)

    summed = mlp_out + residual
    expected = norm(summed)
    actual = torch.empty_like(residual)
    M._fused_rms_norm(residual, norm, actual, add=mlp_out)
    reference = _rms_norm_reference(summed, norm.eps, weight=norm.weight)
    _assert_as_accurate_as_eager(
        actual, expected, reference, "residual-add final RMSNorm", "GemmaRMSNorm (model.norm)", dtype
    )


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
    reference = _rms_norm_reference(x, norm.eps, scale=scale, shift=shift)
    _assert_as_accurate_as_eager(actual, expected, reference, "AdaRMSNorm", "Pi05AdaRMSNorm", dtype, _MODULATION_NOTE)


# ── GEMMs with their epilogues ─────────────────────────────────────────
@pytest.fixture(params=["cublas", "triton"])
def gemm(request, monkeypatch):
    """Which GEMM feeds the epilogues: cuBLAS (float32's, and bfloat16 forced
    onto it to pin its rounding points) or the Triton one bfloat16 serves with."""
    if request.param == "cublas":
        monkeypatch.setattr(M, "_TRITON_GEMM_DTYPES", ())
    return request.param


def _check_gemm_epilogue(
    gemm: str, dtype: torch.dtype, actual, expected, reference, kernel: str, eager: str, cublas_note: str
) -> None:
    """``cublas_note`` says what stays in torch when cuBLAS runs the GEMM."""
    if gemm == "cublas":
        _assert_bit_exact(actual, expected, kernel, eager, dtype, cublas_note)
    else:
        _assert_as_accurate_as_eager(actual, expected, reference, kernel, eager, dtype)


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
    note = (
        "The q/k/v projection GEMMs run in torch (cuBLAS); the Triton kernel only applies RoPE and writes q, K and V."
    )
    _check_gemm_epilogue(
        gemm, dtype, q_out, q[0].contiguous(), q64[0], "QKV+RoPE", f"q_proj + RoPE on the {part}", note
    )
    _check_gemm_epilogue(
        gemm, dtype, key_cache[rows], k[0, 0], k64[0, 0], "QKV+RoPE", f"k_proj + RoPE on the {part}", note
    )
    _check_gemm_epilogue(gemm, dtype, value_cache[rows], v[0, 0], v64, "QKV+RoPE", f"v_proj on the {part}", note)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("gated", [True, False], ids=["gated", "plain"])
@pytest.mark.parametrize("n_in", [2048, 4096], ids=["o_proj", "down_proj"])
@pytest.mark.parametrize("tokens", CHUNKS, ids=lambda tokens: f"chunk={tokens}")
def test_linear_residual_matches_gated_residual(gemm, dtype, gated, n_in, tokens):
    """``_gated_residual(residual, linear(x), gate)``, the float32 residual stream
    updated in place as the fused denoising step runs it."""
    _skip_unserved(gemm, dtype)
    _, width, _ = SHAPES["suffix"]
    linear = _proj_dtype_module(_randomize(nn.Linear(n_in, width, bias=False), seed=17), dtype)
    x = _randn(tokens, n_in, dtype=dtype, seed=18)
    residual = _randn(tokens, width, scale=4.0, seed=19)
    gate = _randn(width, seed=20) if gated else None

    expected = M._gated_residual(residual, linear(x[None])[0], gate)
    projection64 = x.double() @ linear.weight.double().t()
    reference = residual.double() + (gate.double() * projection64 if gated else projection64)
    actual = residual.clone()
    M._fused_linear_residual(x, linear, actual, actual, gate=gate)
    proj = {2048: "o_proj", 4096: "down_proj"}[n_in]
    residual_add = "gated residual add" if gated else "residual add"
    note = f"The {proj} GEMM runs in torch (cuBLAS); the Triton kernel only applies the {residual_add}."
    _check_gemm_epilogue(gemm, dtype, actual, expected, reference, "linear-residual", f"{proj} + {residual_add}", note)


@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("tokens", CHUNKS, ids=lambda tokens: f"chunk={tokens}")
def test_linear_gelu_mul_matches_gemma_mlp_input(gemm, dtype, tokens):
    """``act_fn(gate_proj(x)) · up_proj(x)``, the action expert's MLP up to ``down_proj``."""
    _skip_unserved(gemm, dtype)
    _, width, mlp_width = SHAPES["suffix"]
    mlp = _proj_dtype_module(_randomize(GemmaMLP(_gemma_config(width, mlp_width)), seed=21), dtype)
    x = _randn(tokens, width, dtype=dtype, seed=22)

    expected = (mlp.act_fn(mlp.gate_proj(x[None])) * mlp.up_proj(x[None]))[0]
    x64 = x.double()
    reference = torch.nn.functional.gelu(x64 @ mlp.gate_proj.weight.double().t(), approximate="tanh") * (
        x64 @ mlp.up_proj.weight.double().t()
    )
    actual = torch.empty_like(expected)
    M._fused_linear_gelu_mul(x, mlp, actual)
    eager = "GemmaMLP act_fn(gate_proj(x)) * up_proj(x)"
    note = "The gate_proj and up_proj GEMMs run in torch (cuBLAS); the Triton kernel only applies the GELU and product."
    _check_gemm_epilogue(gemm, dtype, actual, expected, reference, "linear GELU-multiply", eager, note)


# ── Reordered reductions: attention and the output head ────────────────
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("part", SHAPES, ids=str)
def test_attention_matches_eager_attend(dtype, part):
    """``_attend`` over one KV head with eager's float mask: a prefix-shaped
    bidirectional mask with padded (fully masked) query rows (the served prefix
    keeps eager's ``_attend``), and a denoising step's queries over the
    ``[prefix | suffix]`` cache row."""
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
    if M._triton_gemms(dtype):
        note = (
            "It runs as three Triton kernels (scores, masked softmax, values) that keep the full score matrix "
            "in memory, not as one flash-attention kernel."
        )
    else:
        note = "The QK^T and PV matmuls run in torch (cuBLAS); only the masked softmax is a Triton kernel."
    _assert_as_accurate_as_eager(actual, expected, reference, "attention", f"_attend on the {part}", dtype, note)


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
    _assert_as_accurate_as_eager(
        actual,
        expected,
        reference,
        "final AdaRMSNorm + action_out_proj",
        "Pi05AdaRMSNorm + action_out_proj",
        dtype,
        _MODULATION_NOTE,
    )


def _exact_norm_reference(x, weight, add=None, shift=None, eps=1e-6):
    residual = x if add is None else x + add
    value = residual.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    value = value * (1.0 + weight.float())
    if shift is not None:
        value = value + shift.float()
    return residual, value.to(residual.dtype)


@pytest.mark.parametrize("shape", [(50, 1024), (968, 2048)])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("has_add", [False, True])
def test_exact_norm_cuda_matches_eager_bitwise(shape, adaptive, has_add):
    if not kernels.HAS_TRITON or not kernels._supported_torch_reduction():
        pytest.skip("Fast reduction not audited for the installed Torch release/backend")
    generator = torch.Generator(device="cuda").manual_seed(173)
    x = torch.randn(shape, device="cuda", generator=generator).to(torch.bfloat16)
    add = torch.randn(shape, device="cuda", generator=generator).to(torch.bfloat16) if has_add else None
    weight = torch.randn(shape[-1], device="cuda", generator=generator) * 0.125
    shift = torch.randn(shape[-1], device="cuda", generator=generator) * 0.125 if adaptive else None
    assert kernels.can_use_exact_rms_norm(x, weight, add=add, shift=shift)
    actual = kernels.residual_rms_norm(x, weight, add=add, shift=shift)
    expected = _exact_norm_reference(x, weight, add, shift)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("kind", ["short", "other_width", "strided", "fp32", "old_torch"])
def test_exact_norm_cuda_unsupported_cases_fall_back(monkeypatch, kind):
    shape = (15, 1024) if kind == "short" else (50, 17 if kind == "other_width" else 1024)
    dtype = torch.float32 if kind == "fp32" else torch.bfloat16
    x = torch.randn(*shape, device="cuda").to(dtype)
    if kind == "strided":
        x = torch.stack((x, x), dim=-1)[..., 0]
        assert not x.is_contiguous()
    if kind == "old_torch":
        monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    weight = torch.randn(shape[-1], device="cuda")
    assert not kernels.can_use_exact_rms_norm(x, weight)
    actual = kernels.residual_rms_norm(x, weight)
    expected = _exact_norm_reference(x, weight)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("width", [1024, 2048])
@torch.no_grad()
def test_exact_norm_cuda_graph_refreshes_inputs(width):
    if not kernels.HAS_TRITON or not kernels._supported_torch_reduction():
        pytest.skip("Fast reduction not audited for the installed Torch release/backend")
    first = torch.randn(50, width, device="cuda").to(torch.bfloat16)
    second = torch.randn_like(first)
    x = first.clone()
    add = torch.randn_like(x)
    weight = torch.randn(width, device="cuda")
    shift = torch.randn_like(weight)
    assert kernels.can_use_exact_rms_norm(x, weight, add=add, shift=shift)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kernels.residual_rms_norm(x, weight, add=add, shift=shift)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = kernels.residual_rms_norm(x, weight, add=add, shift=shift)
    for value in (first, second, first):
        x.copy_(value)
        graph.replay()
        expected = _exact_norm_reference(x, weight, add, shift)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


_EXACT_SOFTMAX_SHAPES = [(50, 1018), (968, 968)]


def _exact_softmax_inputs(mask_kind="holes", shape=(50, 1018)):
    queries, keys = shape
    generator = torch.Generator(device="cuda").manual_seed(193)
    scores = torch.randn(1, 8, queries, keys, device="cuda", generator=generator).to(torch.bfloat16)
    mask = torch.zeros(1, 1, queries, keys, device="cuda")
    if mask_kind == "holes":
        mask[..., :968:3] = torch.finfo(torch.float32).min
    elif mask_kind == "causal":
        blocked = torch.arange(queries, device="cuda")[None, :] > torch.arange(queries, device="cuda")[:, None]
        mask[..., keys - queries :].masked_fill_(blocked, torch.finfo(torch.float32).min)
    elif mask_kind == "all_masked":
        mask.fill_(torch.finfo(torch.float32).min)
    elif mask_kind == "soft_bias":
        mask = torch.randn(mask.shape, device="cuda", generator=generator)
    return scores, mask


def _require_audited_softmax(shape):
    if not kernels._supported_torch_reduction() or (shape == (968, 968) and torch.version.cuda != "13.2"):
        pytest.skip("Softmax geometry not audited for the installed Torch release/CUDA build")


def _exact_softmax_reference(scores, mask, scale=1.0 / 16, output_dtype=None):
    logits = scores * scale
    if mask is not None:
        logits = logits + mask
    return torch.softmax(logits, dim=-1, dtype=torch.float32).to(output_dtype or scores.dtype)


@pytest.mark.parametrize("mask_kind", ["all_valid", "holes", "causal", "all_masked", "soft_bias"])
@pytest.mark.parametrize("shape", _EXACT_SOFTMAX_SHAPES)
def test_exact_softmax_cuda_matches_eager_bitwise(mask_kind, shape):
    _require_audited_softmax(shape)
    scores, mask = _exact_softmax_inputs(mask_kind, shape)
    assert kernels.can_use_exact_scaled_masked_softmax(scores, mask, 1.0 / 16)
    actual = kernels.scaled_masked_softmax(scores, mask, 1.0 / 16)
    assert torch.equal(actual, _exact_softmax_reference(scores, mask))


@pytest.mark.parametrize(
    "kind",
    ["shape", "score_stride", "mask_stride", "fp32", "mask_bf16", "no_mask", "scale", "output_fp32", "old_torch"],
)
def test_exact_softmax_cuda_unsupported_cases_fall_back(monkeypatch, kind):
    scores, mask = _exact_softmax_inputs()
    scale, output_dtype = 1.0 / 16, None
    if kind == "shape":
        scores, mask = scores[..., :9].contiguous(), mask[..., :9].contiguous()
    elif kind == "score_stride":
        scores = torch.stack((scores, scores), dim=-1)[..., 0]
    elif kind == "mask_stride":
        mask = torch.stack((mask, mask), dim=-1)[..., 0]
    elif kind == "fp32":
        scores = scores.float()
    elif kind == "mask_bf16":
        mask = mask.to(torch.bfloat16)
    elif kind == "no_mask":
        mask = None
    elif kind == "scale":
        scale = 0.2
    elif kind == "output_fp32":
        output_dtype = torch.float32
    elif kind == "old_torch":
        monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    assert not kernels.can_use_exact_scaled_masked_softmax(scores, mask, scale, output_dtype=output_dtype)
    actual = kernels.scaled_masked_softmax(scores, mask, scale, output_dtype=output_dtype)
    assert torch.equal(actual, _exact_softmax_reference(scores, mask, scale, output_dtype))


@pytest.mark.parametrize("kind", ["broadcast_mask", "cuda_build"])
def test_exact_prefix_softmax_unsupported_cases_fall_back(monkeypatch, kind):
    scores, mask = _exact_softmax_inputs(shape=(968, 968))
    if kind == "broadcast_mask":
        mask = mask[..., :1, :].contiguous()
    else:
        monkeypatch.setattr(torch.version, "cuda", "13.1")
    assert not kernels.can_use_exact_scaled_masked_softmax(scores, mask, 1.0 / 16)
    actual = kernels.scaled_masked_softmax(scores, mask, 1.0 / 16)
    assert torch.equal(actual, _exact_softmax_reference(scores, mask))


def test_exact_prefix_softmax_rejects_mismatched_expert_mask():
    scores, _ = _exact_softmax_inputs(shape=(968, 968))
    _, expert_mask = _exact_softmax_inputs()
    assert not kernels.can_use_exact_scaled_masked_softmax(scores, expert_mask, 1.0 / 16)


def test_exact_softmax_cuda_grad_uses_eager():
    scores, mask = _exact_softmax_inputs("soft_bias")
    scores.requires_grad_()
    mask.requires_grad_()
    assert not kernels.can_use_exact_scaled_masked_softmax(scores, mask, 1.0 / 16)
    actual = kernels.scaled_masked_softmax(scores, mask, 1.0 / 16)
    expected = _exact_softmax_reference(scores, mask)
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (scores, mask), gradient, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (scores, mask), gradient)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))


@torch.no_grad()
@pytest.mark.parametrize("shape", _EXACT_SOFTMAX_SHAPES)
def test_exact_softmax_cuda_graph_refreshes_scores_and_mask(shape):
    _require_audited_softmax(shape)
    first, first_mask = _exact_softmax_inputs("holes", shape)
    second = -first
    _, second_mask = _exact_softmax_inputs("causal", shape)
    scores, mask = first.clone(), first_mask.clone()
    assert kernels.can_use_exact_scaled_masked_softmax(scores, mask, 1.0 / 16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kernels.scaled_masked_softmax(scores, mask, 1.0 / 16)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = kernels.scaled_masked_softmax(scores, mask, 1.0 / 16)
    for request_scores, request_mask in ((first, first_mask), (second, second_mask), (first, first_mask)):
        scores.copy_(request_scores)
        mask.copy_(request_mask)
        graph.replay()
        assert torch.equal(actual, _exact_softmax_reference(scores, mask))


@pytest.mark.parametrize("shape", _EXACT_SOFTMAX_SHAPES)
def test_exact_softmax_cuda_attention_preserves_qk_pv(shape):
    _require_audited_softmax(shape)
    queries, keys = shape
    query = _randn(1, 8, queries, 256, dtype=torch.bfloat16, seed=191)
    key = _randn(1, 1, keys, 256, dtype=torch.bfloat16, seed=192)
    value = _randn(1, 1, keys, 256, dtype=torch.bfloat16, seed=193)
    _, mask = _exact_softmax_inputs(shape=shape)
    expected = M._attend(query, key, value, mask, 8, 1.0 / 16)
    actual = M._attend(query, key, value, mask, 8, 1.0 / 16, exact_bf16_softmax=True)
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_accurate_expert_qkv_dispatch_and_graph_refresh():
    """Audit dispatch and graph ownership without claiming arbitrary-weight equality."""
    tokens, width, mlp_width = SHAPES["suffix"]
    dtype = torch.bfloat16
    config = _gemma_config(width, mlp_width)
    attn = _proj_dtype_module(_randomize(GemmaAttention(config, layer_idx=0), seed=12), dtype)
    rotary = GemmaRotaryEmbedding(config).to(DEVICE)
    first = _randn(tokens, width, dtype=dtype, seed=13)
    second = _randn(tokens, width, dtype=dtype, seed=14)
    positions = 376 + torch.arange(tokens, device=DEVICE)[None]
    first_rope = rotary(first, positions)
    second_rope = rotary(second, positions + 7)
    x = first.clone()
    cos, sin = (value[0].clone() for value in first_rope)
    q_out = torch.empty(HEADS, tokens, HEAD_DIM, dtype=dtype, device=DEVICE)
    key_cache = torch.full((tokens + 12, HEAD_DIM), float("nan"), dtype=dtype, device=DEVICE)
    value_cache = torch.full_like(key_cache, float("nan"))
    k_out, v_out = key_cache[7:-5], value_cache[7:-5]
    args = (x, attn, cos, sin, q_out, k_out, v_out)
    selected = M._can_use_accurate_expert_triton_qkv(*args)
    expected_selection = (
        str(torch.__version__).split("+")[0] == "2.13.0"
        and torch.version.cuda == "13.2"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    )
    assert selected == expected_selection
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            M._fused_qkv_rope(*args, triton_gemm=selected)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        M._fused_qkv_rope(*args, triton_gemm=selected)
    for request, rope in ((first, first_rope), (second, second_rope), (first, first_rope)):
        x.copy_(request)
        cos.copy_(rope[0][0])
        sin.copy_(rope[1][0])
        graph.replay()
        actual = [value.clone() for value in (q_out, k_out, v_out)]
        M._fused_qkv_rope(*args, triton_gemm=selected)
        assert all(torch.equal(a, b) for a, b in zip(actual, (q_out, k_out, v_out)))
        for cache in (key_cache, value_cache):
            assert torch.isnan(cache[:7]).all() and torch.isnan(cache[-5:]).all()

        shape = (1, tokens, -1, HEAD_DIM)
        q = attn.q_proj(x[None]).view(shape).transpose(1, 2)
        k = attn.k_proj(x[None]).view(shape).transpose(1, 2)
        v = attn.v_proj(x[None])[0]
        q, k = apply_rotary_pos_emb(q, k, cos[None], sin[None])
        q64 = (x.double() @ attn.q_proj.weight.double().t()).view(shape).transpose(1, 2)
        k64 = (x.double() @ attn.k_proj.weight.double().t()).view(shape).transpose(1, 2)
        q64, k64 = apply_rotary_pos_emb(q64, k64, cos[None].double(), sin[None].double())
        v64 = x.double() @ attn.v_proj.weight.double().t()
        for result, eager, reference in zip(actual, (q[0], k[0, 0], v), (q64[0], k64[0, 0], v64)):
            _check_gemm_epilogue(
                "triton" if selected else "cublas",
                dtype,
                result,
                eager,
                reference,
                "accurate expert QKV",
                "separate projections + RoPE",
                "Unsupported runtime/device uses the original cuBLAS path.",
            )


def _exact_qk_inputs(seed):
    # Dyadic operands keep all partial sums exactly representable in FP32.
    # This pins mapping/output rounding independently of a cuBLAS GEMM's
    # arbitrary-input reduction. Real-checkpoint parity covers serving values.
    generator = _generator(seed)
    query = torch.randint(-4, 5, (1, 8, 50, 256), device=DEVICE, generator=generator).to(torch.bfloat16) / 16
    key = torch.randint(-4, 5, (1, 1, 1018, 256), device=DEVICE, generator=generator).to(torch.bfloat16) / 16
    return query, key


def _require_audited_qk(query, key):
    if not qk_kernels.can_use_exact_bf16_qk(query, key):
        pytest.skip("Requires audited Torch 2.13/CUDA 13.2 on SM90 or SM120 with Gluon.")


def _exact_qk(query, key):
    output = qk_kernels.exact_bf16_qk(query, key, h100_score_kernel=M._attn_scores_kernel)
    assert output is not None
    return output


@torch.no_grad()
def test_exact_qk_cuda_scores_and_attention():
    query, key = _exact_qk_inputs(26)
    _require_audited_qk(query, key)
    expected_scores = query @ M._repeat_kv(key, 8).transpose(2, 3)
    actual_scores = _exact_qk(query, key)
    assert actual_scores.dtype == torch.bfloat16
    assert actual_scores.shape == (1, 8, 50, 1018)
    assert torch.equal(actual_scores, expected_scores)
    value = _randn(1, 1, 1018, 256, dtype=torch.bfloat16, seed=27)
    _, mask = _exact_softmax_inputs("holes")
    expected = M._attend(query, key, value, mask, 8, 1.0 / 16)
    actual = M._attend(query, key, value, mask, 8, 1.0 / 16, exact_bf16_qk=True, exact_bf16_softmax=True)
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_exact_qk_cuda_graph_refresh_and_output_ownership():
    first, second = _exact_qk_inputs(0), _exact_qk_inputs(26)
    _require_audited_qk(*first)
    # Nonzero aligned storage offsets model K/V cache slices; sentinels also
    # detect any unintended writes to caller-owned inputs.
    storage = [torch.full((value.numel() + 4,), float("nan"), dtype=value.dtype, device=DEVICE) for value in first]
    query, key = [buffer[2:-2].view_as(value) for buffer, value in zip(storage, first)]
    query.copy_(first[0])
    key.copy_(first[1])
    assert qk_kernels.can_use_exact_bf16_qk(query, key)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _exact_qk(query, key)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _exact_qk(query, key)
    captured_ptr = captured.data_ptr()
    for request_query, request_key in (first, second, first):
        query.copy_(request_query)
        key.copy_(request_key)
        graph.replay()
        expected = request_query @ M._repeat_kv(request_key, 8).transpose(2, 3)
        assert captured.data_ptr() == captured_ptr and torch.equal(captured, expected)
        actual = captured.clone()
        assert torch.equal(actual, _exact_qk(query, key))
        for buffer in storage:
            assert torch.isnan(buffer[:2]).all() and torch.isnan(buffer[-2:]).all()
        assert torch.equal(query, request_query) and torch.equal(key, request_key)
    retained = _exact_qk(*first)
    expected_retained = retained.clone()
    _exact_qk(*second)
    assert torch.equal(retained, expected_retained)
