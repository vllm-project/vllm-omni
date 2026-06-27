# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OOT GGUF integration via vllm-gguf-plugin monkey-patching.

These tests verify that the plugin's monkey-patch correctly injects GGUF
behavior into DiffusersPipelineLoader without any GGUF code in omni.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=True)
        self.vae = nn.Linear(2, 2, bias=False)
        self.register_buffer("transformer_buffer", torch.ones(1))
        self.calls: list[list[str]] = []
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="vae",
                revision=None,
                prefix="vae.",
                fall_back_to_pt=True,
            ),
        ]

    def load_weights(self, weights):
        loadable = dict(self.named_parameters())
        loadable.update(dict(self.named_buffers()))
        seen: list[str] = []
        loaded: set[str] = set()
        for name, tensor in weights:
            seen.append(name)
            if name in loadable:
                target = loadable[name]
                target.data.copy_(tensor.to(dtype=target.dtype))
                loaded.add(name)
        self.calls.append(seen)
        return loaded


def _make_loader() -> DiffusersPipelineLoader:
    loader = object.__new__(DiffusersPipelineLoader)
    loader.load_config = SimpleNamespace(
        download_dir="cache-dir",
        ignore_patterns=["*.tmp"],
    )
    loader.od_config = SimpleNamespace(
        revision="main",
        model_class_name="QwenImagePipeline",
        tf_model_config={"model_type": "qwen_image"},
    )
    loader.quant_config = None
    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0
    loader.parallel_config = SimpleNamespace(use_hsdp=False)
    return loader


def test_patch_load_weights_calls_plugin_for_gguf():
    """When GGUF is active, the patched load_weights calls the plugin."""
    from vllm_gguf_plugin.weights_adapter.diffusion.integration import _patch_diffusers_loader

    # Apply patch
    _patch_diffusers_loader()

    try:
        loader = _make_loader()
        loader.quant_config = SimpleNamespace(
            get_name=lambda: "gguf",
            gguf_model="weights.gguf",
        )
        model = _DummyModel()

        # Monkeypatch plugin's load_diffusion_gguf_weights inside the patched loader
        captured: dict = {}

        def _fake_load(**kw):
            captured.update(locals())
            return {"transformer.weight", "transformer.bias", "vae.weight"}

        import vllm_gguf_plugin.weights_adapter.diffusion.loader as _mod

        orig = _mod.load_diffusion_gguf_weights
        _mod.load_diffusion_gguf_weights = _fake_load
        try:
            loader.load_weights(model)
        finally:
            _mod.load_diffusion_gguf_weights = orig

        assert captured["gguf_model"] == "weights.gguf"
        assert captured["model_class_name"] == "QwenImagePipeline"
    finally:
        # Restore original methods
        DiffusersPipelineLoader._gguf_load_weights_patched = False


def test_patch_load_weights_passes_through_for_non_gguf():
    """When quant_config is not GGUF, the patched load_weights delegates to original."""
    from vllm_gguf_plugin.weights_adapter.diffusion.integration import _patch_diffusers_loader

    _patch_diffusers_loader()

    try:
        loader = _make_loader()
        loader.quant_config = SimpleNamespace(get_name=lambda: "fp8")
        model = _DummyModel()

        # Set up original load_weights to return successfully
        original_load_weights = DiffusersPipelineLoader.load_weights

        def _fake_original(self, model):
            # Simulate successful HF loading
            model.calls.append(["fake-hf"])
            for name, param in model.named_parameters():
                param.data.fill_(1.0)
            model.named_buffers()["transformer_buffer"].fill_(1.0)

        DiffusersPipelineLoader.load_weights = _fake_original
        try:
            loader.load_weights(model)
            assert model.calls[-1] == ["fake-hf"]
        finally:
            DiffusersPipelineLoader.load_weights = original_load_weights
    finally:
        DiffusersPipelineLoader._gguf_load_weights_patched = False
