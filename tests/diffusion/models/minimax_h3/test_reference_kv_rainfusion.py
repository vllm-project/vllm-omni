# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference-KV/RainFusion contracts: padding, compact geometry, and warmup."""

import sys
import types
from unittest import mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends import rainfusion_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout, VideoTokenSpan
from vllm_omni.diffusion.attention.backends.rainfusion_attn import RainFusionAttentionBackend, RainFusionAttentionImpl
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3Attention,
    _compact_reference_video_layout,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("supported", [False, True])
def test_rainfusion_packed_padding_capability_matches_flash(monkeypatch, supported):
    monkeypatch.setattr(
        rainfusion_attn.FlashAttentionBackend, "supports_packed_mask_free", classmethod(lambda cls: supported)
    )
    assert RainFusionAttentionBackend.supports_packed_mask_free() is supported


def test_compact_ref2va_video_offsets_drop_only_reference_rows():
    # Reference image tokens may be interleaved with text/audio context.
    refs = torch.tensor([2, 3, 7, 8])
    layout = VideoTokenLayout(
        used_len=24,
        video_spans=(
            VideoTokenSpan(start=2, latent_grid=(1, 1, 2), role="reference"),
            VideoTokenSpan(start=12, latent_grid=(3, 2, 2), role="target"),
        ),
    )
    compact = _compact_reference_video_layout(layout, refs, 20)
    assert compact.used_len == 20
    assert compact.video_spans == (VideoTokenSpan(start=8, latent_grid=(3, 2, 2), role="target"),)


def test_compact_fl2va_removes_whole_boundary_frames():
    layout = VideoTokenLayout(prefix_len=2, latent_grid=(4, 2, 2))
    compact = _compact_reference_video_layout(layout, torch.arange(2, 6), 14)
    assert compact.video_spans == (VideoTokenSpan(start=2, latent_grid=(3, 2, 2), role="target"),)


@pytest.mark.parametrize("refs", [[3], [6, 7, 8, 9]])
def test_partial_or_disjoint_target_geometry_falls_back(refs):
    layout = VideoTokenLayout(prefix_len=2, latent_grid=(4, 2, 2))
    assert _compact_reference_video_layout(layout, torch.tensor(refs), 18 - len(refs)) is None


@pytest.mark.parametrize("compact", [False, True])
def test_h3_padding_metadata_stays_mask_free_with_rainfusion(monkeypatch, compact):
    monkeypatch.setattr(RainFusionAttentionBackend, "supports_packed_mask_free", classmethod(lambda cls: True))

    class Capture(torch.nn.Module):
        use_ring = False
        attn_backend = RainFusionAttentionBackend

        def forward(self, q, k, v, metadata):
            self.metadata = metadata
            return q

    attn = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attn)
    attn.attention = Capture()
    state = types.SimpleNamespace(post_parallel_cache=True, global_reference_rows=2)
    q = torch.randn(8, 2, 4)
    attn._run_packed_attention(
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        max_seqlen=5,
        packed_total=8,
        reference_kv_tier1_state=state,
        reference_kv_layer_index=0,
        reference_kv_compact=compact,
    )
    metadata = attn.attention.metadata
    assert metadata.attn_mask is None
    assert metadata.packed_padding.q_length == 5
    assert metadata.packed_padding.kv_length == (7 if compact else 5)
    if compact:
        assert metadata.extra["minimax_h3_compact_reference_rows"] == 2


def _make_reuse(monkeypatch):
    monkeypatch.setattr(rainfusion_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(rainfusion_attn, "_MIN_VIDEO_BLOCKS", 1)
    impl = RainFusionAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="blocks.0.attn",
        qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8},
    )
    metadata = AttentionMetadata(
        extra={"minimax_h3_compact_reference_rows": 3, "max_seqlen_q": 261, "max_seqlen_k": 264},
        video_layout=VideoTokenLayout(
            used_len=261,
            video_spans=(VideoTokenSpan(start=5, latent_grid=(2, 8, 16), role="target"),),
        ),
    )
    q = torch.randn(1, 264, 2, 128)
    k = torch.randn(1, 267, 2, 128)
    v = torch.randn_like(k)
    return impl, metadata, q, k, v


def test_reuse_sparse_keeps_reference_keys_and_excludes_padding(monkeypatch):
    impl, metadata, q, k, v = _make_reuse(monkeypatch)
    seen = {}

    def sparse(q_full, key, value, *, video_spans=None, **kwargs):
        seen.update(q=q_full, k=key, v=value, kwargs={**kwargs, "video_spans": video_spans})
        return q_full + 1

    monkeypatch.setitem(sys.modules, "mindiesd", types.SimpleNamespace(sparse_attention=sparse))
    out = impl.forward_npu(q, k, v, metadata)
    assert seen["q"].shape[1] == seen["k"].shape[1] == 264
    assert torch.count_nonzero(seen["q"][:, :3]) == 0
    torch.testing.assert_close(seen["q"][:, 3:], q[:, :261])
    torch.testing.assert_close(seen["k"], k[:, :264])
    assert seen["kwargs"]["video_spans"] == [{"start": 8, "latent_shape": [2, 8, 16]}]
    torch.testing.assert_close(out[:, :261], q[:, :261] + 1)
    assert out.shape == q.shape
    assert torch.count_nonzero(out[:, 261:]) == 0
    assert metadata.video_layout.video_spans[0].start == 5


def test_reuse_warmup_preserves_rectangular_dense_call(monkeypatch):
    impl, metadata, q, k, v = _make_reuse(monkeypatch)
    impl.dense_fallback.forward_npu = mock.Mock(return_value=q)
    monkeypatch.setattr(impl, "_resolve_plan", lambda metadata: None)
    assert impl.forward_npu(q, k, v, metadata) is q
    impl.dense_fallback.forward_npu.assert_called_once_with(q, k, v, metadata)


def test_invalid_reuse_prefix_is_rejected(monkeypatch):
    impl, metadata, q, k, v = _make_reuse(monkeypatch)
    with pytest.raises(ValueError, match="reference prefix"):
        impl.forward_npu(q, k[:, :-1], v[:, :-1], metadata)


def test_unmarked_rectangular_attention_stays_dense(monkeypatch):
    impl, metadata, q, k, v = _make_reuse(monkeypatch)
    metadata.extra.pop("minimax_h3_compact_reference_rows")
    impl.dense_fallback.forward_npu = mock.Mock(return_value=q)
    assert impl.forward_npu(q, k, v, metadata) is q
    impl.dense_fallback.forward_npu.assert_called_once_with(q, k, v, metadata)
