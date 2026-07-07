# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for generic, model-agnostic per-forward video geometry publication.

Covers the transformer forward pre-hook that replaces the old per-model
``set_forward_context_geometry`` calls:
- patch-size normalization across the two in-tree conventions
- the pre-hook publishing raw primitives onto the active ForwardContext
- best-effort no-ops (no patch size / non-5D input / no active context)
- idempotent registration that fires on a real forward
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.video_geometry import (
    _resolve_patch_size,
    _video_geometry_pre_hook,
    register_video_geometry_hooks,
)
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestResolvePatchSize:
    """Normalize patch size to (p_t, p_h, p_w) across model conventions."""

    def test_tuple_from_config(self):
        # Wan: config.patch_size is a 3-tuple.
        module = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))
        assert _resolve_patch_size(module) == (1, 2, 2)

    def test_scalar_plus_temporal_from_attrs(self):
        # HunyuanVideo: scalar patch_size + separate patch_size_t, no .config.
        module = SimpleNamespace(patch_size=2, patch_size_t=4)
        assert _resolve_patch_size(module) == (4, 2, 2)

    def test_scalar_without_temporal_defaults_unpatched_time(self):
        # Most video DiTs do not patch time: a scalar patch_size with no
        # patch_size_t must default to p_t = 1, not an isotropic (2, 2, 2)
        # (which would silently halve the published frame count).
        module = SimpleNamespace(patch_size=2)
        assert _resolve_patch_size(module) == (1, 2, 2)

    def test_config_takes_precedence_over_attr(self):
        module = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)), patch_size=8)
        assert _resolve_patch_size(module) == (1, 2, 2)

    def test_none_when_absent(self):
        assert _resolve_patch_size(SimpleNamespace()) is None
        assert _resolve_patch_size(SimpleNamespace(config=SimpleNamespace())) is None


class TestPreHook:
    """The pre-hook publishes raw primitives; best-effort otherwise."""

    @staticmethod
    def _module():
        return SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))

    def test_publishes_from_kwargs(self):
        hs = torch.zeros(2, 16, 21, 60, 104)  # (B, C, T, H, W)
        with override_forward_context(ForwardContext()):
            _video_geometry_pre_hook(self._module(), (), {"hidden_states": hs})
            ctx = get_forward_context()
            assert ctx.latent_shape == (21, 60, 104)
            assert ctx.patch_size == (1, 2, 2)

    def test_publishes_from_positional_args(self):
        hs = torch.zeros(1, 16, 13, 32, 32)
        with override_forward_context(ForwardContext()):
            _video_geometry_pre_hook(self._module(), (hs,), {})
            assert get_forward_context().latent_shape == (13, 32, 32)

    def test_noop_for_non_5d_input(self):
        hs = torch.zeros(1, 256, 16)  # image/audio-like, not a video grid
        with override_forward_context(ForwardContext()):
            _video_geometry_pre_hook(self._module(), (), {"hidden_states": hs})
            assert get_forward_context().latent_shape is None

    def test_noop_when_patch_size_absent(self):
        hs = torch.zeros(1, 16, 13, 32, 32)
        with override_forward_context(ForwardContext()):
            _video_geometry_pre_hook(SimpleNamespace(), (), {"hidden_states": hs})
            assert get_forward_context().patch_size is None

    def test_noop_without_active_context(self):
        # Must not raise when no ForwardContext is active.
        hs = torch.zeros(1, 16, 13, 32, 32)
        with override_forward_context(None):
            _video_geometry_pre_hook(self._module(), (), {"hidden_states": hs})


class _TinyTransformer(torch.nn.Module):
    config = SimpleNamespace(patch_size=(1, 2, 2))

    def forward(self, hidden_states):
        return hidden_states


class TestRegistration:
    """register_video_geometry_hooks is idempotent and fires on forward."""

    def test_hook_fires_on_forward(self):
        pipeline = SimpleNamespace(transformer=_TinyTransformer(), transformer_2=None)
        register_video_geometry_hooks(pipeline)
        hs = torch.zeros(1, 16, 21, 60, 104)
        with override_forward_context(ForwardContext()):
            pipeline.transformer(hidden_states=hs)
            assert get_forward_context().latent_shape == (21, 60, 104)

    def test_idempotent_registration(self):
        module = _TinyTransformer()
        pipeline = SimpleNamespace(transformer=module, transformer_2=None)
        register_video_geometry_hooks(pipeline)
        register_video_geometry_hooks(pipeline)  # second call must not double-register
        assert len(module._forward_pre_hooks) == 1

    def test_no_transformer_is_noop(self):
        register_video_geometry_hooks(SimpleNamespace())  # must not raise
        register_video_geometry_hooks(None)
