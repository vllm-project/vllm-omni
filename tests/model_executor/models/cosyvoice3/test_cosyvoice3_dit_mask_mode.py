# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.models.cosyvoice3_audio import cosyvoice3_dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPTH = 2


class _HostReads(TorchDispatchMode):
    """Count tensor-to-Python reads, each of which blocks on the device."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.is_nonzero.default, torch.ops.aten._local_scalar_dense.default):
            self.count += 1
        return func(*args, **(kwargs or {}))


class _MaskInspectingAttention(nn.Module):
    """FLASH_ATTN's contract: an unpublished mode is resolved by reading the mask."""

    def __init__(self):
        super().__init__()
        self.modes = []

    def forward(self, query, key, value, attn_metadata=None):
        mask = None if attn_metadata is None else attn_metadata.attn_mask
        mode = None if attn_metadata is None else attn_metadata.extra.get("attention_mask_mode")
        self.modes.append(mode)
        masked = mask is not None and (bool(torch.any(~mask)) if mode is None else mode != "none")
        q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
        attn_mask = mask[:, None, None, :] if mask is not None and masked else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)


def _dit():
    torch.manual_seed(0)
    dit = cosyvoice3_dit.DiT(
        dim=32, depth=DEPTH, heads=2, dim_head=16, dropout=0.0, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80
    ).eval()
    attention = _MaskInspectingAttention()
    for block in dit.transformer_blocks:
        block.attn.attn = attention
    return dit, attention


def _dit_inputs(mask):
    generator = torch.Generator().manual_seed(1)
    batch, _, width = mask.shape
    x, mu, cond = (torch.randn(batch, 80, width, generator=generator) for _ in range(3))
    return x, mask, mu, torch.full((batch,), 0.5), torch.randn(batch, 80, generator=generator), cond


def _run(dit, mask, monkeypatch, reads_unpublished):
    monkeypatch.setattr(cosyvoice3_dit.DiT, "_reads_unpublished_masks", lambda self, device_type: reads_unpublished)
    x, mask, mu, t, spks, cond = _dit_inputs(mask)
    with torch.inference_mode(), _HostReads() as counter:
        output = dit(x, mask, mu, t, spks=spks, cond=cond)
    return output, counter.count


def test_dit_reads_a_full_mask_once_per_forward_for_flash_attn(monkeypatch):
    dit, attention = _dit()
    mask = torch.ones(2, 1, 12)
    expected, layer_reads = _run(dit, mask, monkeypatch, reads_unpublished=False)
    assert layer_reads == DEPTH and attention.modes == [None] * DEPTH
    attention.modes.clear()
    output, reads = _run(dit, mask, monkeypatch, reads_unpublished=True)
    assert reads == 1
    assert attention.modes == ["none"] * DEPTH
    assert torch.equal(output, expected)


def test_dit_keeps_inspecting_a_padded_mask(monkeypatch):
    dit, attention = _dit()
    mask = torch.ones(2, 1, 12)
    mask[1, :, 9:] = 0
    expected, _ = _run(dit, mask, monkeypatch, reads_unpublished=False)
    attention.modes.clear()
    output, reads = _run(dit, mask, monkeypatch, reads_unpublished=True)
    assert reads == 1 + DEPTH
    assert attention.modes == [None] * DEPTH
    assert torch.equal(output, expected)


def test_attention_skips_the_identity_fill_only_for_mode_none():
    torch.manual_seed(0)
    layer = cosyvoice3_dit.DiTAttention(dim=16, heads=2, dim_head=8, dropout=0.0).eval()
    layer.attn = _MaskInspectingAttention()
    x = torch.randn(2, 5, 16)
    full = torch.ones(2, 5, dtype=torch.bool)
    padded = full.clone()
    padded[0, 3:] = False
    with torch.inference_mode():
        assert torch.equal(layer(x, mask=full, attention_mask_mode="none"), layer(x, mask=full))
        out = layer(x, mask=padded)
        # Mode "none" asserts a full mask, so a padded one shows whether the fill ran.
        unfilled = layer(x, mask=padded, attention_mask_mode="none")
    assert layer.attn.modes == ["none", None, None, "none"]
    assert torch.all(out[0, 3:] == 0)
    assert not torch.all(unfilled[0, 3:] == 0)


@pytest.mark.parametrize(
    ("backend", "device_type", "npu_graph", "expected"),
    [
        ("FLASH_ATTN", "cuda", False, True),
        ("TORCH_SDPA", "cuda", False, False),
        ("CUDNN_ATTN", "cuda", False, False),
        (None, "cuda", False, False),
        ("FLASH_ATTN", "xpu", False, False),
        ("FLASH_ATTN", "cpu", False, False),
        ("FLASH_ATTN", "cuda", True, False),
    ],
)
def test_full_mask_is_published_only_where_flash_attn_honors_it(backend, device_type, npu_graph, expected):
    dit, _ = _dit()
    layer = cosyvoice3_dit.DiTAttention(dim=32, heads=2, dim_head=16, dropout=0.0)
    layer.attn.attn_backend = None if backend is None else DiffusionAttentionBackendEnum[backend].get_class()
    dit.transformer_blocks[0].attn = layer
    if npu_graph:
        dit._use_npu_graph_attention = True
    assert dit._reads_unpublished_masks(device_type) is expected


def test_cpu_forward_publishes_nothing():
    dit, attention = _dit()
    x, mask, mu, t, spks, cond = _dit_inputs(torch.ones(2, 1, 12))
    with torch.inference_mode():
        dit(x, mask, mu, t, spks=spks, cond=cond)
    assert attention.modes == [None] * DEPTH


def test_dit_attention_layer_exposes_its_backend():
    # _reads_unpublished_masks reads this attribute chain on the real layer.
    dit = cosyvoice3_dit.DiT(
        dim=32, depth=DEPTH, heads=2, dim_head=16, dropout=0.0, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80
    )
    backend = dit.transformer_blocks[0].attn.attn.attn_backend
    assert backend is not None and isinstance(backend.get_name(), str)
