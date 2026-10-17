# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full frame coverage and bounded activations for native background CLIP."""

from __future__ import annotations

import gc
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks.lingbot_world.encoder_inference import install_vbench_background_inference, verify_background_coverage

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def native_background(monkeypatch):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0, 5.0]))
            self.calls = []

        def encode_image(self, images):
            self.calls.append((len(images), torch.is_grad_enabled()))
            return images.mean(dim=(2, 3)) * self.weight

    module = ModuleType("vbench.background_consistency")
    native_clip = SimpleNamespace(load=lambda: (Encoder().eval(), "native-preprocess"))
    setattr(module, "clip", native_clip)
    monkeypatch.setitem(sys.modules, "vbench.background_consistency", module)
    return module, native_clip


@pytest.mark.parametrize("frames", [1, 32, 33, 7197])
def test_background_encoding_preserves_every_image_and_order_without_gradients(native_background, frames):
    module, native_clip = native_background
    native, _ = native_clip.load()
    images = torch.arange(frames * 12, dtype=torch.float32).reshape(frames, 3, 2, 2)
    expected = native.encode_image(images).detach()
    stats = install_vbench_background_inference()
    bounded, preprocess = module.clip.load()
    actual = bounded.encode_image(images)
    assert torch.equal(actual, expected)
    assert not actual.requires_grad
    assert preprocess == "native-preprocess"
    assert module.clip is not native_clip
    assert all(count <= 32 and not grad for count, grad in bounded.calls)
    assert stats["encoded_images"] == frames
    assert stats["forward_batches"] == (frames + 31) // 32
    assert stats["max_forward_batch"] == min(frames, 32)
    verify_background_coverage(stats, frames)
    assert install_vbench_background_inference() is stats
    assert native.encode_image(images).requires_grad


def test_background_loader_does_not_keep_released_models_alive(native_background):
    module, _ = native_background
    install_vbench_background_inference()
    model, _ = module.clip.load()
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("shape", [(0, 3, 2, 2), (3, 2, 2)])
def test_background_encoder_rejects_invalid_frame_populations(native_background, shape):
    module, _ = native_background
    stats = install_vbench_background_inference()
    model, _ = module.clip.load()
    with pytest.raises(ValueError, match="nonempty NCHW"):
        model.encode_image(torch.zeros(shape))
    assert stats["encoded_images"] == 0


def test_background_encoding_propagates_a_failure_after_two_complete_batches(native_background):
    module, native_clip = native_background

    class FailingEncoder(torch.nn.Module):
        def encode_image(self, images):
            if (images >= 64).any():
                raise RuntimeError("bad late frame")
            return images.mean(dim=(2, 3))

    native_clip.load = lambda: (FailingEncoder().eval(), "native-preprocess")
    stats = install_vbench_background_inference()
    model, _ = module.clip.load()
    with pytest.raises(RuntimeError, match="bad late frame"):
        model.encode_image(torch.arange(65, dtype=torch.float32).reshape(65, 1, 1, 1))
    assert stats["encoded_images"] == 64
    with pytest.raises(ValueError, match="every frame"):
        verify_background_coverage(stats, 65)


def test_background_encoder_cannot_return_a_partial_feature_population(native_background):
    module, native_clip = native_background

    class TruncatedEncoder(torch.nn.Module):
        def encode_image(self, images):
            return images[:-1].mean(dim=(2, 3))

    native_clip.load = lambda: (TruncatedEncoder().eval(), "native-preprocess")
    install_vbench_background_inference()
    model, _ = module.clip.load()
    with pytest.raises(ValueError, match="changed the image population"):
        model.encode_image(torch.zeros(33, 3, 2, 2))


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 64),
        ("gradients", True),
        ("encoded_images", 7196),
        ("encoded_images", True),
        ("forward_batches", 224),
        ("forward_batches", True),
        ("max_forward_batch", 33),
    ],
)
def test_background_coverage_rejects_missing_frames_or_changed_protocol(field, value):
    stats = {
        "batch_size": 32,
        "gradients": False,
        "encoded_images": 7197,
        "forward_batches": 225,
        "max_forward_batch": 32,
    }
    stats[field] = value
    with pytest.raises(ValueError, match="every frame"):
        verify_background_coverage(stats, 7197)
