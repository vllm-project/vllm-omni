# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for per-forward attention state.

Covers:
- ForwardContext.total_denoise_steps + geometry fields + setters + plumbing
- PerForwardState dataclass + AttentionMetadata.per_forward field
- Attention._with_per_forward_state bridge (step state + geometry)
"""

from dataclasses import FrozenInstanceError, replace

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
    set_forward_context_geometry,
    set_forward_context_total_denoise_steps,
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
            set_forward_context_geometry(total_latent_frames=21, patches_per_frame=1560)

    def test_geometry_setter_updates_active_context(self):
        with override_forward_context(ForwardContext()):
            set_forward_context_geometry(total_latent_frames=21, patches_per_frame=1560)
            ctx = get_forward_context()
            assert ctx.total_latent_frames == 21
            assert ctx.patches_per_frame == 1560


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
        assert set(d) == {"denoise_step_idx", "total_denoise_steps", "total_latent_frames", "patches_per_frame"}
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
        # Geometry alone (no step state) must still populate per_forward.
        with override_forward_context(ForwardContext(total_latent_frames=21, patches_per_frame=1560)):
            out = Attention._with_per_forward_state(AttentionMetadata(), q)
        pf = out.per_forward
        assert pf is not None
        assert pf.total_latent_frames.tolist() == [21, 21]
        assert pf.patches_per_frame.tolist() == [1560, 1560]
        assert pf.denoise_step_idx is None  # step fields stay None
