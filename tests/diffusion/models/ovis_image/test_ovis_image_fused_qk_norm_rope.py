# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ovis-Image attention on the fused QK RMSNorm + RoPE ops (double and single blocks)."""

import os
from unittest.mock import patch

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton required"),
]

_HEADS, _HEAD_DIM = 24, 128
_DIM = _HEADS * _HEAD_DIM
_MIN_TOKENS_ENV = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"


@pytest.fixture
def _dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.platforms import current_omni_platform

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29518")
    # vLLM 0.28+: parallel-state init, CustomOp construction and the linear
    # layers' forward all read the current vLLM config.
    config = VllmConfig()
    with set_current_vllm_config(config), current_omni_platform.get_default_ir_op_priority(config).set_priority():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        try:
            yield
        finally:
            cleanup_dist_env_and_memory()


def _rope(seq_len: int):
    torch.manual_seed(seq_len)
    angles = torch.randn(seq_len, _HEAD_DIM // 2, device="cuda")
    return torch.cos(angles), torch.sin(angles)


@pytest.mark.parametrize("double_stream", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("txt,img", [(77, 1023), (256, 1024), (256, 4096)])
def test_ovis_attention_fused_matches_eager(_dist_env, double_stream, batch, txt, img):
    from vllm import ir

    from vllm_omni.diffusion.models.ovis_image import ovis_image_transformer as model_module

    torch.manual_seed(3)
    with torch.device("cuda"):
        attn = model_module.OvisImageAttention(
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
    hidden = torch.randn(batch, img if double_stream else txt + img, _DIM, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(batch, txt, _DIM, device="cuda", dtype=torch.bfloat16) if double_stream else None
    cos, sin = _rope(txt + img)
    table = model_module.pack_qk_norm_rope_table(
        cos, sin, batch, dtype=torch.bfloat16, head_dim=_HEAD_DIM, cuda_rms_norm=True
    )
    assert table is not None
    probe = hidden.new_empty((batch, img, _HEADS, _HEAD_DIM))
    assert ir.ops.rms_norm.dispatch(probe, attn.norm_q.weight, 1e-6, None).provider == "vllm_c"

    op_name = "fused_joint_qkv_norm_rope" if double_stream else "fused_qk_norm_rope"
    with (
        torch.no_grad(),
        patch.object(model_module, op_name, wraps=getattr(model_module, op_name)) as fused_op,
        patch.object(attn.attn, "forward", wraps=attn.attn.forward) as attention,
    ):
        eager = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin))
        fused_op.assert_not_called()
        eager_qkv = attention.call_args.args
        fused = attn(
            hidden,
            encoder_hidden_states=encoder,
            image_rotary_emb=(cos, sin),
            **{model_module.QK_NORM_ROPE_TABLE_KEY: table},
        )
        fused_op.assert_called_once()
        assert fused_op.call_args.kwargs["cuda_rms_norm"] is True
        for expected, actual in zip(eager_qkv, attention.call_args.args):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    for e, f in zip(eager if double_stream else (eager,), fused if double_stream else (fused,)):
        assert torch.equal(f.view(torch.int16), e.view(torch.int16))


@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("threshold", [None, "48", "49"])
def test_ovis_forward_fusion_gate(_dist_env, monkeypatch, head_dim, threshold):
    """Cover the real pack call site, the B*(text+image) boundary and partial RoPE."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.models.ovis_image import ovis_image_transformer as model_module

    torch.manual_seed(4)
    config = OmniDiffusionConfig(model=None, enforce_eager=True)
    config.set_tf_model_config(TransformerConfig.from_dict({"num_layers": 1}))
    with torch.device("cuda"):
        model = (
            model_module.OvisImageTransformer2DModel(
                od_config=config,
                num_single_layers=1,
                attention_head_dim=head_dim,
                num_attention_heads=4,
                joint_attention_dim=32,
            )
            .to(torch.bfloat16)
            .eval()
        )
    with torch.no_grad():
        # vLLM parallel linear layers allocate uninitialized parameters.
        for name, p in model.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.02)
            elif name.endswith("weight") and "norm" in name:
                p.uniform_(0.5, 1.5)
            else:
                p.zero_()
    batch, txt, img = 2, 7, 17  # 48 joint positions; 128 rotary dims for both head sizes.
    img_ids = torch.zeros(img, 3, device="cuda")
    img_ids[:, 1] = torch.arange(img, device="cuda")
    txt_ids = torch.zeros(txt, 3, device="cuda")
    txt_ids[:, 0] = torch.arange(txt, device="cuda")
    inputs = dict(
        hidden_states=torch.randn(batch, img, 64, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(batch, txt, 32, device="cuda", dtype=torch.bfloat16),
        timestep=torch.full((batch,), 0.5, device="cuda"),
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )
    with (
        torch.no_grad(),
        patch.object(model_module, "pack_qk_norm_rope_table", wraps=model_module.pack_qk_norm_rope_table) as pack,
        patch.object(model_module, "fused_joint_qkv_norm_rope", wraps=model_module.fused_joint_qkv_norm_rope) as joint,
        patch.object(model_module, "fused_qk_norm_rope", wraps=model_module.fused_qk_norm_rope) as single,
    ):
        monkeypatch.setenv(_MIN_TOKENS_ENV, str(2**63 - 1))
        eager = model(**inputs)[0]
        pack.assert_not_called()
        joint.assert_not_called()
        single.assert_not_called()
        if threshold is None:
            monkeypatch.delenv(_MIN_TOKENS_ENV)
        else:
            monkeypatch.setenv(_MIN_TOKENS_ENV, threshold)
        actual = model(**inputs)[0]
        should_pack = threshold != "49"
        assert pack.call_count == int(should_pack)
        if should_pack:
            assert pack.call_args.kwargs["head_dim"] == head_dim
            assert pack.call_args.kwargs["cuda_rms_norm"] is True
        should_fuse = should_pack and head_dim == 128
        assert joint.call_count == single.call_count == int(should_fuse)
    assert torch.isfinite(eager).all()
    assert torch.equal(actual.view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("double_stream", [True, False])
def test_ovis_attention_rejects_partial_rope_table(_dist_env, double_stream):
    """An externally supplied table must also respect the CUDA RMSNorm contract."""
    from vllm_omni.diffusion.models.ovis_image import ovis_image_transformer as model_module

    with torch.device("cuda"):
        attn = (
            model_module.OvisImageAttention(
                query_dim=192, heads=1, dim_head=192, added_kv_proj_dim=192 if double_stream else None
            )
            .to(torch.bfloat16)
            .eval()
        )
    with torch.no_grad():
        for p in attn.parameters():
            p.fill_(0.02)
    hidden = torch.randn(1, 5, 192, device="cuda", dtype=torch.bfloat16)
    encoder = hidden[:, :2] if double_stream else None
    cos, sin = _rope(7 if double_stream else 5)
    table = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
    op_name = "fused_joint_qkv_norm_rope" if double_stream else "fused_qk_norm_rope"
    with torch.no_grad(), patch.object(model_module, op_name, wraps=getattr(model_module, op_name)) as fused_op:
        expected = attn(hidden, encoder_hidden_states=encoder, image_rotary_emb=(cos, sin))
        actual = attn(
            hidden,
            encoder_hidden_states=encoder,
            image_rotary_emb=(cos, sin),
            **{model_module.QK_NORM_ROPE_TABLE_KEY: table},
        )
        fused_op.assert_not_called()
    for e, a in zip(expected if double_stream else (expected,), actual if double_stream else (actual,)):
        assert torch.equal(a.view(torch.int16), e.view(torch.int16))
