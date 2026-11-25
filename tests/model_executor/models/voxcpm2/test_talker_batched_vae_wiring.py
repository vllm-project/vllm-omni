# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The talker resolves the legacy batched VAE decoder against the runtime graph."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker
from vllm_omni.model_executor.models.voxcpm2.runtime_config import _VoxCPM2RuntimeConfig
from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import VoxCPM2TalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _talker(
    config: _VoxCPM2RuntimeConfig,
    *,
    delayed_audio_copy: bool = False,
    audio_emit_every: int = 1,
    vae_cuda_graph: bool = False,
) -> VoxCPM2TalkerForConditionalGeneration:
    # ``__new__`` keeps the test off weight loading: the resolver only reads the
    # runtime config and the three incompatibility flags.
    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    model._runtime_config = config
    model._enable_delayed_audio_copy = delayed_audio_copy
    model._audio_emit_every = audio_emit_every
    model._enable_vae_cuda_graph = vae_cuda_graph
    return model


def test_legacy_path_is_off_when_the_unified_graph_runs():
    """#9738: the unified graph batches VAE decode itself, so the pair must not run."""
    config = _VoxCPM2RuntimeConfig(enable_batched_vae_decode=True, enable_unified_decode_graph=True)
    assert _talker(config)._resolve_batched_vae_decode(use_cuda_graph=True) is False


def test_legacy_path_survives_without_the_unified_graph():
    config = _VoxCPM2RuntimeConfig(enable_batched_vae_decode=True, enable_unified_decode_graph=True)
    assert _talker(config)._resolve_batched_vae_decode(use_cuda_graph=False) is True


def test_incompatible_features_disable_a_runnable_legacy_path(monkeypatch):
    warnings = []
    monkeypatch.setattr(voxcpm2_talker.logger, "warning", lambda *args: warnings.append(args))
    config = _VoxCPM2RuntimeConfig(enable_batched_vae_decode=True, enable_unified_decode_graph=False)
    assert _talker(config, delayed_audio_copy=True)._resolve_batched_vae_decode(use_cuda_graph=True) is False
    assert warnings


def test_unreachable_legacy_path_does_not_warn(monkeypatch):
    """The default CUDA configuration must not warn about a decoder it never runs."""
    warnings = []
    monkeypatch.setattr(voxcpm2_talker.logger, "warning", lambda *args: warnings.append(args))
    config = _VoxCPM2RuntimeConfig(enable_batched_vae_decode=True, enable_unified_decode_graph=True)
    talker = _talker(config, delayed_audio_copy=True, audio_emit_every=3, vae_cuda_graph=True)
    assert talker._resolve_batched_vae_decode(use_cuda_graph=True) is False
    assert not warnings
