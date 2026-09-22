# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for in-place decode revert and stage-boundary cache release.

``_revert_decoded_inplace`` must be numerically identical to the checkpoint's
``revert_tensor`` (denormalize → clamp → rearrange) with zero whole-video
copies, and ``_release_stage_cache`` must force the allocator-cache release
(or fall back to a plain empty_cache when no component cache exists).
"""

import pytest
import torch

import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

DENORM_MEAN = (-2.11628412, -2.03571429, -1.80444444)  # -imagenet_mean/std
DENORM_STD = (4.36681223, 4.46428571, 4.44444444)  # 1/imagenet_std


class _FakeDenormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _FakeProcessor:
    def __init__(self):
        self.transform_rev = _FakeDenormalize(DENORM_MEAN, DENORM_STD)
        self.use_3d_conv = True
        self.revert_calls = 0

    def revert_tensor(self, tensor):
        self.revert_calls += 1
        return _legacy_revert(tensor)


class _FakeCheckpointModel:
    def __init__(self):
        self.processor = _FakeProcessor()


def _vae(model=None):
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = model if model is not None else _FakeCheckpointModel()
    return vae


def _legacy_revert(tensor: torch.Tensor) -> torch.Tensor:
    """The checkpoint's revert_tensor, op for op, on the CPU (B == 1)."""
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(2)
    assert tensor.ndim == 5
    # Normalize runs on the (b t) c h w rearrangement; at B == 1 that is a
    # view of (B, C, T, H, W), so the per-channel broadcast is equivalent.
    mean = torch.tensor(DENORM_MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(DENORM_STD).view(1, 3, 1, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.clamp(0, 1)


def test_revert_inplace_matches_legacy_bitwise():
    decoded = torch.randn(1, 3, 7, 6, 8).clamp_(-3, 3)
    expected = _legacy_revert(decoded.clone())
    out = _vae()._revert_decoded_inplace(decoded.clone())
    assert out.shape == expected.shape == (1, 3, 7, 6, 8)
    assert torch.equal(out, expected)


def test_revert_inplace_mutates_without_extra_copies():
    decoded = torch.randn(1, 3, 5, 4, 4)
    out = _vae()._revert_decoded_inplace(decoded)
    assert out is decoded  # in-place: no whole-video copy materializes
    assert out.is_contiguous()


def test_revert_inplace_unsqueezes_4d_input():
    # 4-D decoded tensors follow the (B, C, H, W) image contract: unsqueeze(2)
    # lands T=1, matching the checkpoint's revert_tensor branch.
    decoded = torch.randn(1, 3, 6, 8)
    expected = _legacy_revert(decoded.clone())
    out = _vae()._revert_decoded_inplace(decoded.clone())
    assert torch.equal(out, expected)


def test_revert_inplace_falls_back_without_contract():
    vae = _vae()
    vae.model.processor.transform_rev = type("Empty", (), {})()
    decoded = torch.randn(1, 3, 5, 4, 4)
    out = vae._revert_decoded_inplace(decoded)
    assert vae.model.processor.revert_calls == 1
    assert torch.equal(out, _legacy_revert(decoded))


def test_revert_inplace_falls_back_on_bad_arity():
    vae = _vae()
    vae.model.processor.transform_rev = _FakeDenormalize((0.5, 0.5), (1.0, 1.0, 1.0))
    vae._revert_decoded_inplace(torch.randn(1, 3, 5, 4, 4))
    assert vae.model.processor.revert_calls == 1


class _RecordingCache:
    def __init__(self):
        self.calls = []

    def release_if_needed(self, *, force=False):
        self.calls.append(force)
        return True


def _pipeline():
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._dlo_component_cache = None
    return pipeline


def test_stage_cache_release_forces_component_cache():
    pipeline = _pipeline()
    cache = _RecordingCache()
    pipeline._dlo_component_cache = cache
    pipeline._release_stage_cache()
    assert cache.calls == [True]


def test_stage_cache_release_falls_back_to_empty_cache(monkeypatch):
    released = []
    monkeypatch.setattr(
        pipeline_module.current_omni_platform,
        "empty_cache",
        lambda: released.append(True),
        raising=True,
    )
    pipeline = _pipeline()
    pipeline._release_stage_cache()
    assert released == [True]
