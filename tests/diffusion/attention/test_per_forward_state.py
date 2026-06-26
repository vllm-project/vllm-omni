# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for per-forward attention state.

Covers:
- ForwardContext.total_denoise_steps + geometry fields + setters + plumbing
- PerForwardState dataclass + AttentionMetadata.per_forward field
- Attention._with_per_forward_state bridge (step state + geometry)
"""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PerForwardState,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    create_forward_context,
    get_forward_context,
    override_forward_context,
    set_forward_context_denoise_step_idx,
    set_forward_context_total_denoise_steps,
    set_forward_context_video_geometry,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestForwardContextTotalDenoiseSteps:
    """total_denoise_steps is the denominator for step-aware schedules."""

    def test_defaults_none(self):
        ctx = ForwardContext()
        assert ctx.total_denoise_steps is None
        assert ctx.denoise_step_idx is None

    def test_create_forward_context_plumbs_total(self):
        ctx = create_forward_context(denoise_step_idx=5, total_denoise_steps=40)
        assert ctx.denoise_step_idx == 5
        assert ctx.total_denoise_steps == 40

    def test_setter_updates_active_context(self):
        with override_forward_context(ForwardContext()):
            set_forward_context_total_denoise_steps(40)
            set_forward_context_denoise_step_idx(7)
            ctx = get_forward_context()
            assert ctx.total_denoise_steps == 40
            assert ctx.denoise_step_idx == 7

    def test_setter_is_noop_without_active_context(self):
        # Mirrors set_forward_context_denoise_step_idx: a silent no-op when no
        # ForwardContext is active (must not raise).
        with override_forward_context(None):
            set_forward_context_total_denoise_steps(40)
            set_forward_context_video_geometry(latent_shape=(21, 60, 104), patch_size=(1, 2, 2))

    def test_geometry_setter_updates_active_context(self):
        with override_forward_context(ForwardContext()):
            set_forward_context_video_geometry(latent_shape=(21, 60, 104), patch_size=(1, 2, 2))
            ctx = get_forward_context()
            assert ctx.latent_shape == (21, 60, 104)
            assert ctx.patch_size == (1, 2, 2)


class TestPerForwardState:
    """Frozen, per-sample tensor carrier with a shallow to_dict()."""

    def test_defaults_none(self):
        st = PerForwardState()
        assert st.denoise_step_idx is None
        assert st.total_denoise_steps is None

    def test_frozen(self):
        st = PerForwardState()
        with pytest.raises(FrozenInstanceError):
            st.denoise_step_idx = torch.tensor([0])  # type: ignore[misc]

    def test_to_dict_includes_none_by_default(self):
        st = PerForwardState(denoise_step_idx=torch.tensor([5, 5]))
        d = st.to_dict()
        assert set(d) == {
            "denoise_step_idx",
            "total_denoise_steps",
            "latent_shape",
            "patch_size",
            "total_latent_frames",
            "patches_per_frame",
        }
        assert d["total_denoise_steps"] is None

    def test_to_dict_exclude_none_drops_unset(self):
        st = PerForwardState(denoise_step_idx=torch.tensor([5, 5]))
        assert set(st.to_dict(exclude_none=True)) == {"denoise_step_idx"}

    def test_to_dict_is_shallow_no_tensor_copy(self):
        # asdict() would deep-copy tensors; to_dict() must return the same object.
        t = torch.tensor([5, 5])
        assert PerForwardState(denoise_step_idx=t).to_dict()["denoise_step_idx"] is t


class TestAttentionMetadataPerForward:
    """AttentionMetadata carries per_forward by composition (one field)."""

    def test_default_none_and_extra_intact(self):
        md = AttentionMetadata()
        assert md.per_forward is None
        assert md.extra == {}  # Tier-2 opaque channel still present

    def test_replace_sets_per_forward(self):
        st = PerForwardState(denoise_step_idx=torch.tensor([7]))
        md = replace(AttentionMetadata(), per_forward=st)
        assert md.per_forward is st


class TestRingSparseGuard:
    """Attention._assert_ring_backend_supported fails fast on SPARSE_ATTN + ring.

    A staticmethod, so it is exercised directly without constructing an Attention
    layer (which would require a GPU platform).
    """

    def test_raises_for_sparse_under_ring(self):
        with pytest.raises(ValueError, match="ring sequence parallelism"):
            Attention._assert_ring_backend_supported("SPARSE_ATTN", use_ring=True, role="self")

    def test_noop_for_sparse_without_ring(self):
        Attention._assert_ring_backend_supported("SPARSE_ATTN", use_ring=False, role="self")

    def test_noop_for_dense_backend_under_ring(self):
        Attention._assert_ring_backend_supported("SDPA", use_ring=True, role="self")


class TestFp32SparseGuard:
    """The fp32 SDPA override must not silently bypass a configured sparse plugin.

    ``_run_local_attention`` is exercised on a stand-in ``self`` (constructing an
    Attention layer would require a GPU platform): fp32 raises for SPARSE_ATTN
    instead of quietly rerouting to dense SDPA; dense backends keep the fallback.
    """

    def test_fp32_raises_for_sparse(self):
        from vllm_omni.diffusion.attention.backends.sparse_attn import SparseAttentionBackend

        fake_self = SimpleNamespace(attn_backend=SparseAttentionBackend, role="self")
        q = torch.randn(1, 4, 2, 8, dtype=torch.float32)
        with pytest.raises(ValueError, match="float32"):
            Attention._run_local_attention(fake_self, q, q, q, None)

    def test_fp32_falls_back_to_sdpa_for_dense(self):
        sentinel = torch.zeros(1)
        fake_self = SimpleNamespace(
            attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
            role="self",
            backend_pref="FLASH_ATTN",
            attention=None,
            sdpa_fallback=SimpleNamespace(forward=lambda q, k, v, md: sentinel),
        )
        q = torch.randn(1, 4, 2, 8, dtype=torch.float32)
        assert Attention._run_local_attention(fake_self, q, q, q, None) is sentinel


class TestPerForwardBridge:
    """Attention._with_per_forward_state surfaces ForwardContext state.

    The bridge is a staticmethod, so it is exercised directly without
    constructing an Attention layer (which would require a GPU platform).
    """

    Q = staticmethod(lambda n: torch.zeros(n, 4, 8, 16))  # BSND: n samples

    def test_noop_without_forward_context(self):
        md = AttentionMetadata()
        with override_forward_context(None):
            out = Attention._with_per_forward_state(md, self.Q(2))
        assert out is md

    def test_noop_when_step_state_unset(self):
        md = AttentionMetadata()
        with override_forward_context(ForwardContext()):
            out = Attention._with_per_forward_state(md, self.Q(2))
        assert out is md
        assert out.per_forward is None

    def test_fills_uniform_per_sample_tensors(self):
        q = self.Q(3)
        with override_forward_context(ForwardContext(denoise_step_idx=5, total_denoise_steps=40)):
            out = Attention._with_per_forward_state(AttentionMetadata(), q)
        pf = out.per_forward
        assert pf is not None
        assert pf.denoise_step_idx.tolist() == [5, 5, 5]
        assert pf.total_denoise_steps.tolist() == [40, 40, 40]
        assert pf.denoise_step_idx.device == q.device

    def test_creates_metadata_when_none(self):
        with override_forward_context(ForwardContext(denoise_step_idx=1, total_denoise_steps=10)):
            out = Attention._with_per_forward_state(None, self.Q(2))
        assert isinstance(out, AttentionMetadata)
        assert out.per_forward.denoise_step_idx.tolist() == [1, 1]

    def test_preserves_existing_fields(self):
        md = AttentionMetadata(extra={"kv_cache_dtype": "fp8"})
        with override_forward_context(ForwardContext(denoise_step_idx=0, total_denoise_steps=10)):
            out = Attention._with_per_forward_state(md, self.Q(2))
        assert out.extra == {"kv_cache_dtype": "fp8"}  # preserved via replace()
        assert out.per_forward is not None

    def test_partial_only_step_idx(self):
        with override_forward_context(ForwardContext(denoise_step_idx=3)):
            out = Attention._with_per_forward_state(AttentionMetadata(), self.Q(2))
        assert out.per_forward.denoise_step_idx.tolist() == [3, 3]
        assert out.per_forward.total_denoise_steps is None

    def test_fills_geometry_per_sample(self):
        q = self.Q(2)
        # Raw geometry primitives alone (no step state) must still populate
        # per_forward, as per-sample [num_samples, 3] tensors.
        with override_forward_context(ForwardContext(latent_shape=(21, 60, 104), patch_size=(1, 2, 2))):
            out = Attention._with_per_forward_state(AttentionMetadata(), q)
        pf = out.per_forward
        assert pf is not None
        assert pf.latent_shape.tolist() == [[21, 60, 104], [21, 60, 104]]
        assert pf.patch_size.tolist() == [[1, 2, 2], [1, 2, 2]]
        # Derived fields are NOT produced by the bridge anymore (the dispatcher's
        # resolver fills them); only raw primitives flow through here.
        assert pf.total_latent_frames is None
        assert pf.patches_per_frame is None
        assert pf.denoise_step_idx is None  # step fields stay None
