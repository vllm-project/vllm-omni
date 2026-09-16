# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FLUX.1 attention on the fused QK RMSNorm + RoPE ops (double and single blocks)."""

import os

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEADS, _HEAD_DIM = 24, 128
_DIM = _HEADS * _HEAD_DIM


@pytest.fixture
def _dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    # vLLM 0.28+: parallel-state init, CustomOp construction and the linear
    # layers' forward all read the current vLLM config.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


def _rope(seq_len: int):
    torch.manual_seed(seq_len)
    angles = torch.randn(seq_len, _HEAD_DIM // 2, device="cuda")
    return torch.cos(angles), torch.sin(angles)


def test_table_skipped_on_cpu(monkeypatch):
    """No table (no allocation) where the fused kernel cannot run."""
    from vllm_omni.diffusion.models.flux.flux_transformer import _with_qk_norm_rope_table

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    cos, sin = torch.randn(40, 64), torch.randn(40, 64)
    x = torch.zeros(2, 40, _DIM, dtype=torch.bfloat16)
    assert _with_qk_norm_rope_table({"other": 1}, (cos, sin), x) == {"other": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_table_gate_and_kwargs(monkeypatch):
    from vllm_omni.diffusion.models.flux.flux_transformer import _QK_NORM_ROPE_TABLE_KEY, _with_qk_norm_rope_table

    cos, sin = torch.randn(2, 40, 64, device="cuda"), torch.randn(2, 40, 64, device="cuda")
    x = torch.zeros(2, 40, _DIM, dtype=torch.bfloat16, device="cuda")
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    kw = _with_qk_norm_rope_table({"other": 1}, (cos[0], sin[0]), x)
    assert kw["other"] == 1 and kw[_QK_NORM_ROPE_TABLE_KEY].shape == (80, _HEAD_DIM)
    assert kw[_QK_NORM_ROPE_TABLE_KEY].dtype == torch.bfloat16
    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1000000")
    assert _with_qk_norm_rope_table(None, (cos[0], sin[0]), x) is None
    assert _with_qk_norm_rope_table(None, (cos[0], sin[0]), x.half()) is None  # non-bf16 activations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("double_stream", [True, False])
def test_flux_attention_fused_matches_eager(_dist_env, double_stream):
    from vllm_omni.diffusion.models.flux.flux_transformer import (
        _QK_NORM_ROPE_TABLE_KEY,
        FluxAttention,
        pack_qk_norm_rope_table,
    )

    torch.manual_seed(3)
    with torch.device("cuda"):
        attn = FluxAttention(
            query_dim=_DIM,
            heads=_HEADS,
            dim_head=_HEAD_DIM,
            added_kv_proj_dim=_DIM if double_stream else None,
            eps=1e-6,
        )
    attn = attn.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_((0.02 * torch.randn(p.shape, device="cuda")).to(p.dtype) if p.dim() > 1 else p)
        for n in (attn.norm_q, attn.norm_k) + ((attn.norm_added_q, attn.norm_added_k) if double_stream else ()):
            n.weight.uniform_(0.5, 1.5)
    batch, txt, img = 2, 64, 256
    hidden = torch.randn(batch, img if double_stream else txt + img, _DIM, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(batch, txt, _DIM, device="cuda", dtype=torch.bfloat16) if double_stream else None
    cos, sin = _rope(txt + img)
    table = pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=0)

    with torch.no_grad():
        eager = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin))
        fused = attn(
            hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin), **{_QK_NORM_ROPE_TABLE_KEY: table}
        )
    for e, f in zip(eager if double_stream else (eager,), fused if double_stream else (fused,)):
        torch.testing.assert_close(f, e, atol=0.05, rtol=0.05)
