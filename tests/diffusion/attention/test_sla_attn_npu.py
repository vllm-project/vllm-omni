# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free real MindIE-SD SLA smoke test; requires an Ascend runtime."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.sla_attn import SLAAttentionImpl
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A5"}, num_cards=1)]


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_real_mindiesd_sla_projection_and_padding(precision):
    if not current_omni_platform.is_npu():
        pytest.skip("Ascend NPU is required")
    impl = SLAAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="blocks.0.attn",
        role="self",
        qkv_layout="BSND",
        backend_kwargs={"sla": {"precision": precision}},
    )
    impl.sla_module.to("npu")
    q, k, v = (torch.randn(1, 1536, 2, 128, device="npu", dtype=torch.bfloat16) for _ in range(3))
    cu = torch.tensor([0, 1280], dtype=torch.int32, device="npu")
    md = AttentionMetadata(packed_padding=PackedPaddingMetadata(1280, 1280, cu, cu))
    calls = []
    handle = impl.sla_module.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
    try:
        with torch.no_grad():
            impl.sla_module.proj_l.weight.zero_()
            impl.sla_module.proj_l.bias.zero_()
            baseline = impl.forward(q, k, v, md)
            impl.sla_module.proj_l.bias.fill_(1)
            projected = impl.forward(q, k, v, md)
    finally:
        handle.remove()
    assert calls == [torch.Size([1, 2, 1280, 128])] * 2
    assert torch.count_nonzero(baseline[:, :1280]) > 0
    assert torch.isfinite(projected).all()
    assert torch.count_nonzero(projected[:, 1280:]) == 0
    torch.testing.assert_close(projected[:, :1280], baseline[:, :1280] + 1, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_h3_loader_and_shared_dispatch_reach_real_sla(monkeypatch, precision):
    if not current_omni_platform.is_npu():
        pytest.skip("Ascend NPU is required")
    from torch import nn

    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _FakeLinear
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    # Only projection constructors are replaced to avoid a TP process group.
    # H3 construction, loading, packing, shared dispatch and MindIE-SD are real.
    monkeypatch.setattr(blocks, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "RowParallelLinear", _FakeLinear)
    config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(
            default={"backend": "FLASH_ATTN"},
            per_role={"self": {"backend": "SLA_ATTN", "sla": {"precision": precision}}},
        )
    )
    with set_current_diffusion_config(config):
        attn = blocks.MiniMaxH3Attention(
            blocks.MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=128, hidden_size=256),
            None,
            prefix="blocks.0.attn",
        ).to("npu")
    assert attn.attention.attn_backend.get_name() == "SLA_ATTN"
    assert attn.sla_module is attn.attention.attention.sla_module
    dit = h3.MiniMaxH3DiTModel.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(dit)
    block = nn.Module()
    block.attn = attn
    dit.blocks = nn.ModuleList([block])
    prefix = "blocks.0.attn.sla_module.proj_l."
    weight = torch.zeros(128, 128)
    bias = torch.zeros(128)
    assert dit.load_weights([(prefix + "weight", weight), (prefix + "bias", bias)]) == {
        prefix + "weight",
        prefix + "bias",
    }
    q, k, v = (torch.randn(1536, 2, 128, device="npu", dtype=torch.bfloat16) for _ in range(3))
    cu = torch.tensor([0, 1280, 1536], dtype=torch.int32, device="npu")
    calls = []
    handle = attn.sla_module.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
    try:
        with torch.no_grad():
            baseline = attn._run_packed_attention(q, k, v, cu_seqlens=cu, max_seqlen=1280, packed_total=1536)
            dit.load_weights([(prefix + "weight", weight), (prefix + "bias", bias + 1)])
            projected = attn._run_packed_attention(q, k, v, cu_seqlens=cu, max_seqlen=1280, packed_total=1536)
    finally:
        handle.remove()
    assert calls == [torch.Size([1, 2, 1280, 128])] * 2
    assert torch.count_nonzero(baseline[:1280]) > 0
    assert torch.isfinite(projected).all()
    assert torch.count_nonzero(projected[1280:]) == 0
    torch.testing.assert_close(projected[:1280], baseline[:1280] + 1, atol=0.02, rtol=0.02)
