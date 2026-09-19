# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    _qwen_image_qk_norm_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

BATCH = 2
Q_HEADS = 24
K_HEADS = 24
HEAD_DIM = 128
EPS = 1e-6
SEED = 2026


@dataclass(frozen=True)
class QwenImageQKInput:
    q: torch.Tensor
    k: torch.Tensor
    norm_q: nn.RMSNorm
    norm_k: nn.RMSNorm
    freqs: torch.Tensor


def _make_input(
    *,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    packed_qkv_view: bool,
) -> QwenImageQKInput:
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + seq_len)

    if packed_qkv_view:
        qkv_dim = (Q_HEADS + K_HEADS + K_HEADS) * HEAD_DIM
        qkv = torch.randn(BATCH, seq_len, qkv_dim, device=device, dtype=dtype, generator=gen)
        q, k, _ = qkv.split(
            [
                Q_HEADS * HEAD_DIM,
                K_HEADS * HEAD_DIM,
                K_HEADS * HEAD_DIM,
            ],
            dim=-1,
        )
        q = q.unflatten(-1, (Q_HEADS, HEAD_DIM))
        k = k.unflatten(-1, (K_HEADS, HEAD_DIM))
    else:
        q = torch.randn(BATCH, seq_len, Q_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)
        k = torch.randn(BATCH, seq_len, K_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)

    norm_q = nn.RMSNorm(HEAD_DIM, eps=EPS, device=device, dtype=dtype)
    norm_k = nn.RMSNorm(HEAD_DIM, eps=EPS, device=device, dtype=dtype)
    norm_q.requires_grad_(False)
    norm_k.requires_grad_(False)
    with torch.no_grad():
        norm_q.weight.copy_(torch.randn(HEAD_DIM, device=device, dtype=dtype, generator=gen))
        norm_k.weight.copy_(torch.randn(HEAD_DIM, device=device, dtype=dtype, generator=gen))
    angles = torch.randn(seq_len, HEAD_DIM // 2, device=device, dtype=torch.float32, generator=gen)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return QwenImageQKInput(q=q, k=k, norm_q=norm_q, norm_k=norm_k, freqs=freqs)


def _eager_rotary(data: QwenImageQKInput) -> tuple[torch.Tensor, torch.Tensor]:
    q = data.norm_q(data.q)
    k = data.norm_k(data.k)
    rope = RotaryEmbedding(is_neox_style=False)
    cos = data.freqs.real.to(data.q.dtype)
    sin = data.freqs.imag.to(data.q.dtype)
    return rope(q, cos, sin), rope(k, cos, sin)


def _fp32_complex_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Local copy of the old CUDA helper; fused kernel still tracks this math."""
    paired = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(paired * freqs.unsqueeze(1)).flatten(3).to(x.dtype)


def _fused_kernel_reference(data: QwenImageQKInput) -> tuple[torch.Tensor, torch.Tensor]:
    q = data.norm_q(data.q)
    k = data.norm_k(data.k)
    return _fp32_complex_rotary(q, data.freqs), _fp32_complex_rotary(k, data.freqs)


def _run(data: QwenImageQKInput, *, use_fused: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    return _qwen_image_qk_norm_rope(
        data.q,
        data.k,
        data.norm_q,
        data.norm_k,
        data.freqs,
        RotaryEmbedding(is_neox_style=False),
        EPS,
        use_fused=use_fused,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_qk_norm_rope_cuda_fp32_fallback_matches_reference():
    data = _make_input(
        seq_len=7,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    actual_q, actual_k = _run(data, use_fused=False)
    expected_q, expected_k = _eager_rotary(data)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_qk_norm_rope_cuda_fp16_fallback_matches_reference():
    data = _make_input(
        seq_len=7,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    actual_q, actual_k = _run(data, use_fused=False)
    expected_q, expected_k = _eager_rotary(data)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1, 7, 257, 1024])
@pytest.mark.parametrize("packed_qkv_view", [False, True])
def test_qwen_image_fused_qk_norm_rope_cuda_matches_fp32_rope_reference(
    seq_len: int,
    packed_qkv_view: bool,
):
    data = _make_input(
        seq_len=seq_len,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=packed_qkv_view,
    )

    actual_q, actual_k = _run(data)
    expected_q, expected_k = _fused_kernel_reference(data)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_torch_compile_fullgraph_capture():
    data = _make_input(
        seq_len=257,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )
    rope = RotaryEmbedding(is_neox_style=False)

    def fn(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _qwen_image_qk_norm_rope(
            q,
            k,
            data.norm_q,
            data.norm_k,
            freqs,
            rope,
            EPS,
        )

    compiled_fn = torch.compile(fn, dynamic=True, fullgraph=True)
    expected_q, expected_k = fn(data.q, data.k, data.freqs)
    actual_q, actual_k = compiled_fn(data.q, data.k, data.freqs)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


# --- joint (text + image) launch ------------------------------------------


def _joint_reference(txt: QwenImageQKInput, img: QwenImageQKInput, *, use_fused: bool):
    """What the block does today: one `_qwen_image_qk_norm_rope` per stream,
    then three `torch.cat`s into the joint sequence attention consumes."""
    rope = RotaryEmbedding(is_neox_style=False)
    img_q, img_k = _qwen_image_qk_norm_rope(
        img.q, img.k, img.norm_q, img.norm_k, img.freqs, rope, EPS, use_fused=use_fused
    )
    txt_q, txt_k = _qwen_image_qk_norm_rope(
        txt.q, txt.k, txt.norm_q, txt.norm_k, txt.freqs, rope, EPS, use_fused=use_fused
    )
    return torch.cat([txt_q, img_q], dim=1), torch.cat([txt_k, img_k], dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("txt_len,img_len", [(512, 4096), (77, 1024)])
def test_joint_launch_matches_per_stream_then_cat(txt_len, img_len):
    """The joint op replaces two per-stream launches plus three cats; its Q/K
    must equal the per-stream fused path's, bitwise."""
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_joint_qkv_norm_rope
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import _packed_qk_norm_rope_table

    device = torch.device("cuda")
    txt = _make_input(seq_len=txt_len, dtype=torch.bfloat16, device=device, packed_qkv_view=True)
    img = _make_input(seq_len=img_len, dtype=torch.bfloat16, device=device, packed_qkv_view=True)
    txt_v = torch.randn_like(txt.q)
    img_v = torch.randn_like(img.q)

    table = _packed_qk_norm_rope_table(img.freqs, txt.freqs, BATCH, torch.bfloat16)
    assert table is not None and table.shape == (BATCH * (txt_len + img_len), HEAD_DIM)

    joint_q, joint_k, joint_v = fused_joint_qkv_norm_rope(
        txt.q,
        txt.k,
        txt_v,
        img.q,
        img.k,
        img_v,
        txt.norm_q.weight,
        txt.norm_k.weight,
        img.norm_q.weight,
        img.norm_k.weight,
        table,
        EPS,
    )
    exp_q, exp_k = _joint_reference(txt, img, use_fused=True)
    assert torch.equal(joint_q, exp_q)
    assert torch.equal(joint_k, exp_k)
    assert torch.equal(joint_v, torch.cat([txt_v, img_v], dim=1))


def test_packed_table_skipped_when_fused_path_unavailable(monkeypatch):
    """No table, no allocation, where the CUDA kernel cannot run."""
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import _packed_qk_norm_rope_table

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    vid = torch.polar(torch.ones(64, HEAD_DIM // 2), torch.randn(64, HEAD_DIM // 2))
    txt = torch.polar(torch.ones(16, HEAD_DIM // 2), torch.randn(16, HEAD_DIM // 2))
    assert _packed_qk_norm_rope_table(vid, txt, 1, torch.bfloat16) is None  # CPU
    if torch.cuda.is_available():
        vid, txt = vid.cuda(), txt.cuda()
        assert _packed_qk_norm_rope_table(vid, txt, 1, torch.float16) is None  # non-bf16 activations
        table = _packed_qk_norm_rope_table(vid, txt, 1, torch.bfloat16)
        assert table is not None and table.dtype == torch.float32  # same dtype as the per-stream table
        monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
        assert _packed_qk_norm_rope_table(vid, txt, 1, torch.bfloat16) is None
