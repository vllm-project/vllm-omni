# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.diffusion.attention.test_sla_attn import RecordingSLA
from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _FakeLinear
from vllm_omni.diffusion.attention.backends import sla_attn
from vllm_omni.diffusion.attention.backends.sla_attn import SLAAttentionBackend
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("zero_init", [False, True])
def test_h3_registers_provider_projection_and_executes_sla(monkeypatch, zero_init):
    class CPUAttention(nn.Module):
        def __init__(self, **options):
            super().__init__()
            self.attn_backend = SLAAttentionBackend
            self.attention = sla_attn.SLAAttentionImpl(
                **options, backend_kwargs={"sla": {"zero_init": zero_init, "topk": 1.0}}
            )
            self.use_ring = False
            self.parallel_strategy = Mock(name="strategy")
            self.parallel_strategy.name = "none"

        def add_schedule_candidate_check(self, check):
            pass

        def effective_attention(self):
            return self.attention, self.attn_backend, None

        def forward(self, q, k, v, metadata):
            return self.attention.forward_npu(q, k, v, metadata)

    monkeypatch.setattr(sla_attn, "_mindiesd_sla_cls", lambda: RecordingSLA)
    monkeypatch.setattr(blocks, "Attention", CPUAttention)
    monkeypatch.setattr(blocks, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "RowParallelLinear", _FakeLinear)
    attn = blocks.MiniMaxH3Attention(
        blocks.MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=16),
        None,
        prefix="blocks.0.attn",
    )
    assert attn.sla_module is attn.attention.attention.sla_module
    params = dict(attn.named_parameters())
    assert "sla_module.proj_l.weight" in params
    assert params["sla_module.proj_l.weight"].is_checkpoint_optional == zero_init
    q = torch.randn(8, 2, 16, dtype=torch.bfloat16)
    out = attn._run_packed_attention(q, q, q, cu_seqlens=torch.tensor([0, 5, 8]), max_seqlen=5, packed_total=8)
    assert len(attn.sla_module.calls) == 1
    torch.testing.assert_close(out[:5], q[:5])
    assert torch.count_nonzero(out[5:]) == 0


def model(*, zero_init=False, enabled=True):
    # Construct just the real module tree consumed by the production loader.
    dit = h3.MiniMaxH3DiTModel.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(dit)
    attention = nn.Module()
    if enabled:
        attention.sla_module = nn.Module()
        attention.sla_module.proj_l = nn.Linear(16, 16)
        for param in attention.sla_module.parameters():
            nn.init.zeros_(param)
            param.is_checkpoint_optional = zero_init
            param.is_checkpoint_required = not zero_init
    block = nn.Module()
    block.attn = attention
    dit.blocks = nn.ModuleList([block])
    dit._mark_missing_params_required()
    return dit


@pytest.mark.parametrize("diffusers", [False, True])
def test_load_projection_weights(diffusers):
    dit = model()
    dit._diffusers_weights = diffusers
    # A projection-only stream deliberately exercises the native SLA name
    # before Diffusers translation; no base weights are needed for this check.
    if diffusers:
        dit._serialized_weight_map = {}
        dit.arch = Mock(rope_inv_freq_len=1)
        dit._rope_theta = 10000
        dit.rope = nn.Module()
        dit.rope.register_buffer("inv_freq", torch.empty(1))
    tensors = {name: torch.full_like(p, 0.25) for name, p in dit.named_parameters()}
    loaded = dit.load_weights(tensors.items())
    assert tensors.keys() <= loaded
    for name, param in dit.named_parameters():
        torch.testing.assert_close(param, tensors[name])
        assert param.missing_param_init == "error"


def test_missing_projection_fails():
    dit = model()
    name, param = next(dit.named_parameters())
    with pytest.raises(ValueError, match="missing projection weights"):
        dit.load_weights([(name, torch.zeros_like(param))])


def test_zero_init_explicitly_allows_base_checkpoint():
    dit = model(zero_init=True)
    with torch.no_grad():
        for param in dit.parameters():
            param.fill_(10)
    assert dit.load_weights([]) == set()
    for param in dit.parameters():
        assert param.missing_param_init == "zeros"
        assert param.is_checkpoint_optional
        assert torch.count_nonzero(param) == 0


def test_duplicate_projection_fails():
    dit = model()
    name, param = next(dit.named_parameters())
    with pytest.raises(ValueError, match="duplicate MiniMax-H3 SLA"):
        dit.load_weights([(name, torch.zeros_like(param)), (name, torch.zeros_like(param))])


def test_sla_checkpoint_with_dense_backend_fails():
    dit = model(enabled=False)
    with pytest.raises(ValueError, match="enable SLA_ATTN"):
        dit.load_weights([("blocks.0.attn.sla_module.proj_l.weight", torch.zeros(16, 16))])


def test_h3_packed_forward_rejects_reference_cache():
    attn = blocks.MiniMaxH3Attention.__new__(blocks.MiniMaxH3Attention)
    nn.Module.__init__(attn)
    attn.sla_module = nn.Module()
    attn.attention = Mock(use_ring=False)
    attn.attention.effective_attention.return_value = (None, SLAAttentionBackend, None)
    q = torch.randn(8, 2, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="reference-KV"):
        attn._run_packed_attention(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 5, 8]),
            max_seqlen=5,
            packed_total=8,
            reference_kv_state=Mock(),
            reference_kv_layer_index=0,
        )


def test_sla_disallows_scheduled_backend():
    attn = blocks.MiniMaxH3Attention.__new__(blocks.MiniMaxH3Attention)
    nn.Module.__init__(attn)
    attn.sla_module = nn.Module()
    assert "schedules" in attn._sla_candidate_check(Mock())
