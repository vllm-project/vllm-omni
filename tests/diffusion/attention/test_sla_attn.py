# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends import sla_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.sla_config import SLASpec
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class RecordingSLA(nn.Module):
    """CPU provider double: records layout and exercises a learned projection."""

    def __init__(self, head_dim, **options):
        super().__init__()
        self.options = options
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)
        nn.init.zeros_(self.proj_l.weight)
        nn.init.zeros_(self.proj_l.bias)
        self.calls = []

    def forward(self, q, k, v):
        self.calls.append((q, k, v))
        return v + self.proj_l(v.float()).to(v.dtype)


@pytest.fixture
def impl(monkeypatch):
    monkeypatch.setattr(sla_attn, "_mindiesd_sla_cls", lambda: RecordingSLA)
    return make_impl()


def make_impl(**updates):
    options = dict(
        num_heads=2,
        head_size=16,
        softmax_scale=0.25,
        qkv_layout="BSND",
        prefix="blocks.0.attn",
        role="self",
        backend_kwargs={"sla": {"topk": 1.0}},
    )
    return sla_attn.SLAAttentionImpl(**(options | updates))


def metadata(used=5):
    cu = torch.tensor([0, used], dtype=torch.int32)
    return AttentionMetadata(packed_padding=PackedPaddingMetadata(used, used, cu, cu))


def test_layout_padding_and_projection(impl):
    q = torch.randn(1, 8, 2, 16, dtype=torch.bfloat16)
    out = impl.forward_npu(q, q, q, metadata())
    call = impl.sla_module.calls[-1]
    assert call[0].shape == (1, 2, 5, 16)
    assert all(x.is_contiguous() for x in call)
    torch.testing.assert_close(out[:, :5], q[:, :5])
    assert torch.count_nonzero(out[:, 5:]) == 0
    with torch.no_grad():
        impl.sla_module.proj_l.bias.fill_(1)
    changed = impl.forward_npu(q, q, q, metadata())
    torch.testing.assert_close(changed[:, :5], q[:, :5] + 1)
    assert torch.count_nonzero(changed[:, 5:]) == 0
    assert impl.sla_module.options["topk"] == 1.0


@pytest.mark.parametrize("used", [64, 192, 1216])
def test_zero_selected_blocks_rejected_before_provider(impl, used):
    short = make_impl(backend_kwargs={"sla": {"topk": 0.05}})
    q = torch.zeros(1, used, 2, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="zero KV blocks"):
        short.forward_npu(q, q, q, metadata(used))
    assert not short.sla_module.calls


@pytest.mark.parametrize("topk", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.05"])
def test_invalid_topk(topk):
    with pytest.raises(ValueError, match="sla.topk"):
        SLASpec(topk=topk)


@pytest.mark.parametrize("options", [{"blkq": 32}, {"blkk": True}, {"feature_map": "unknown"}, {"zero_init": "false"}])
def test_invalid_sla_options(options):
    with pytest.raises(ValueError, match="sla."):
        SLASpec(**options)


def test_attention_spec():
    assert AttentionSpec("SLA_ATTN").backend_kwargs() == {"sla": vars(SLASpec())}
    assert AttentionSpec("SLA_ATTN", sla={"topk": 0.1, "zero_init": True}).sla.zero_init is True
    with pytest.raises(ValueError, match="SLA_ATTN"):
        AttentionSpec("FLASH_ATTN", sla={"topk": 0.05})


def test_per_role_config_keeps_refiner_dense():
    config = AttentionConfig(
        default={"backend": "FLASH_ATTN"},
        per_role={
            "self": {"backend": "SLA_ATTN", "sla": {"topk": 0.05, "zero_init": True}},
            "minimax_h3.token_refiner": {"backend": "FLASH_ATTN"},
        },
    )
    assert config.resolve_with_source("self")[0].sla.zero_init
    assert config.resolve_with_source("minimax_h3.token_refiner", "self")[0].backend == "FLASH_ATTN"


def test_missing_provider_is_actionable(monkeypatch):
    import sys
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    monkeypatch.setitem(sys.modules, "mindiesd.layers", ModuleType("mindiesd.layers"))
    with pytest.raises(ImportError, match="requires MindIE-SD"):
        sla_attn.SLAAttentionBackend.validate_available()


@pytest.mark.parametrize(
    "options",
    [
        {"role": "minimax_h3.token_refiner"},
        {"prefix": "token_refiner.blocks.0.attn"},
        {"causal": True},
        {"qkv_layout": "BHSD"},
        {"num_kv_heads": 1},
        {"softmax_scale": 1.0},
    ],
)
def test_unsupported_attention_contract(impl, options):
    with pytest.raises(ValueError, match="SLA|MindIE"):
        make_impl(**options)


@pytest.mark.parametrize("case", ["mask", "ranges", "unequal", "empty", "packed", "batch", "dtype"])
def test_reject_unsupported_metadata(impl, case):
    q = torch.randn(1, 8, 2, 16, dtype=torch.bfloat16)
    md = metadata()
    if case == "mask":
        md.attn_mask = torch.ones(1, 8, dtype=torch.bool)
    elif case == "ranges":
        md.query_ranges = ()
    elif case == "unequal":
        md.packed_padding = PackedPaddingMetadata(5, 6, torch.tensor([0, 5]), torch.tensor([0, 6]))
    elif case == "empty":
        md = metadata(0)
    elif case == "packed":
        md = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 4, 8])})
    elif case == "batch":
        q = q.expand(2, -1, -1, -1)
    elif case == "dtype":
        q = q.float()
    with pytest.raises(ValueError, match="SLA|H3"):
        impl.forward_npu(q, q, q, md)
    assert not impl.sla_module.calls


def test_provider_error_is_not_dense_fallback(impl, monkeypatch):
    def fail(*args):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(impl.sla_module, "forward", fail)
    q = torch.randn(1, 8, 2, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="provider failed"):
        impl.forward_npu(q, q, q, metadata())
