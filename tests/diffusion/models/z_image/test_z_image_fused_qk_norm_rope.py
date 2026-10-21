# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Z-Image attention on the fused single-stream QK RMSNorm + RoPE op."""

import os

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1), pytest.mark.diffusion]

_HEADS, _HEAD_DIM = 30, 128
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
    os.environ.setdefault("MASTER_PORT", "29519")
    config = VllmConfig()
    with (
        set_current_vllm_config(config),
        current_omni_platform.get_default_ir_op_priority(config).set_priority(),
    ):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


def test_packed_table_skipped_on_cpu():
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _packed_qk_norm_rope_table

    cos, sin = torch.randn(3, 20, _HEAD_DIM // 2), torch.randn(3, 20, _HEAD_DIM // 2)
    assert _packed_qk_norm_rope_table(cos, sin, torch.bfloat16) is None


@pytest.mark.skipif(not current_omni_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_packed_table_uses_row_zero_like_rotary_embedding():
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _packed_qk_norm_rope_table

    cos, sin = torch.randn(3, 20, _HEAD_DIM // 2, device="cuda"), torch.randn(3, 20, _HEAD_DIM // 2, device="cuda")
    table = _packed_qk_norm_rope_table(cos, sin, torch.bfloat16)
    assert table.shape == (60, _HEAD_DIM) and table.dtype == torch.bfloat16
    expected = torch.cat((cos[0], sin[0]), dim=-1).to(torch.bfloat16)
    assert torch.equal(table, expected.repeat(3, 1))
    assert _packed_qk_norm_rope_table(cos, sin, torch.float16) is None


@pytest.mark.skipif(not current_omni_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_packed_table_is_built_under_sequence_parallelism():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _packed_qk_norm_rope_table

    shard = 10
    cos = torch.randn(3, shard, _HEAD_DIM // 2, device="cuda")
    sin = torch.randn(3, shard, _HEAD_DIM // 2, device="cuda")

    od_config = OmniDiffusionConfig(model=None, enforce_eager=True, parallel_config={"ulysses_degree": 2})
    assert od_config.parallel_config.sequence_parallel_size == 2
    with set_forward_context(omni_diffusion_config=od_config):
        table = _packed_qk_norm_rope_table(cos, sin, torch.bfloat16)

    assert table is not None and table.shape == (3 * shard, _HEAD_DIM)
    expected = torch.cat((cos[0], sin[0]), dim=-1).to(torch.bfloat16)
    assert torch.equal(table, expected.repeat(3, 1))
    # Compiled execution packs the same table: the fused op also serves torch.compile.
    od_config.enforce_eager = False
    with set_forward_context(omni_diffusion_config=od_config):
        assert torch.equal(_packed_qk_norm_rope_table(cos, sin, torch.bfloat16), expected.repeat(3, 1))


@pytest.mark.skipif(not current_omni_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_z_image_attention_fused_matches_eager(_dist_env, mocker):
    """The eager fused path is bit-identical to norm_q/norm_k + RoPE."""
    from vllm_omni.diffusion.models.z_image import z_image_transformer as z_image

    torch.manual_seed(3)
    with torch.device("cuda"):
        attn = z_image.ZImageAttention(dim=_DIM, num_heads=_HEADS, num_kv_heads=_HEADS, eps=1e-6)
    attn = attn.to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_((0.02 * torch.randn(p.shape, device="cuda")).to(p.dtype) if p.dim() > 1 else p)
        attn.norm_q.weight.uniform_(0.5, 1.5)
        attn.norm_k.weight.uniform_(0.5, 1.5)
    batch, seq = 2, 320
    hidden = torch.randn(batch, seq, _DIM, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(batch, seq, _HEAD_DIM // 2, device="cuda")
    cos, sin = torch.cos(angles), torch.sin(angles)
    mask = torch.ones(batch, seq, dtype=torch.bool, device="cuda")
    table = z_image._packed_qk_norm_rope_table(cos, sin, torch.bfloat16)

    with torch.no_grad():
        eager = attn(hidden, mask, cos, sin)
        fused_op = mocker.spy(z_image, "fused_qk_norm_rope")
        fused = attn(hidden, mask, cos, sin, qk_norm_rope_table=table)
        assert torch.equal(fused, eager)
        assert fused_op.call_count == 1
