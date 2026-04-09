# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GGUF-specific DiffusersPipelineLoader behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.model_loader.diffusers_loader as loader_module
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
    loader.od_config = None
    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0
    return loader


def test_is_gguf_quantization_accepts_dict_config():
    loader = _make_loader()
    od_config = SimpleNamespace(
        quantization_config={
            "method": "gguf",
            "gguf_model": "weights.gguf",
        }
    )

    assert loader._is_gguf_quantization(od_config) is True


def test_is_gguf_quantization_rejects_non_gguf_dict_config():
    loader = _make_loader()
    od_config = SimpleNamespace(
        quantization_config={
            "method": "fp8",
            "gguf_model": "weights.gguf",
        }
    )

    assert loader._is_gguf_quantization(od_config) is False


def test_is_gguf_quantization_requires_gguf_model_for_dict_config():
    loader = _make_loader()
    od_config = SimpleNamespace(quantization_config={"method": "gguf"})

    with pytest.raises(ValueError, match="gguf_model"):
        loader._is_gguf_quantization(od_config)


def test_is_gguf_quantization_accepts_object_config():
    loader = _make_loader()
    od_config = SimpleNamespace(
        quantization_config=SimpleNamespace(
            get_name=lambda: "gguf",
            gguf_model="weights.gguf",
        )
    )

    assert loader._is_gguf_quantization(od_config) is True


def test_is_gguf_quantization_uses_fallback_object_without_get_name():
    loader = _make_loader()
    od_config = SimpleNamespace(quantization_config=SimpleNamespace(gguf_model="weights.gguf"))

    assert loader._is_gguf_quantization(od_config) is True


def test_get_model_loadable_names_collects_parameters_and_buffers():
    loader = _make_loader()
    model = _DummyModel()

    names = loader._get_model_loadable_names(model)

    assert "transformer.weight" in names
    assert "transformer.bias" in names
    assert "transformer_buffer" in names


def test_resolve_gguf_model_path_returns_local_file(tmp_path):
    loader = _make_loader()
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"gguf")

    resolved = loader._resolve_gguf_model_path(str(gguf_file), revision=None)

    assert resolved == str(gguf_file)


def test_resolve_gguf_model_path_downloads_explicit_gguf_filename(monkeypatch: pytest.MonkeyPatch):
    loader = _make_loader()

    monkeypatch.setattr(loader_module.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(
        loader_module,
        "hf_hub_download",
        lambda repo_id, filename, revision, cache_dir: f"{cache_dir}/{repo_id}/{filename}",
    )

    resolved = loader._resolve_gguf_model_path("owner/repo/model-Q4_0.gguf", revision="main")

    assert resolved == "cache-dir/owner/repo/model-Q4_0.gguf"


def test_resolve_gguf_model_path_downloads_by_quant_type(monkeypatch: pytest.MonkeyPatch):
    loader = _make_loader()

    monkeypatch.setattr(loader_module.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(
        loader_module,
        "download_gguf",
        lambda repo_id, quant_type, cache_dir, revision, ignore_patterns: f"{cache_dir}/{repo_id}/{quant_type}.gguf",
    )

    resolved = loader._resolve_gguf_model_path("owner/repo:Q4_0", revision="main")

    assert resolved == "cache-dir/owner/repo/Q4_0.gguf"


def test_get_gguf_weights_iterator_prefixes_source_names(monkeypatch: pytest.MonkeyPatch):
    loader = _make_loader()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )
    od_config = SimpleNamespace(
        revision="main",
        quantization_config=SimpleNamespace(gguf_model="weights.gguf"),
    )

    class _Adapter:
        def weights_iterator(self):
            yield "weight", torch.ones((2, 2))
            yield "bias", torch.zeros(2)

    monkeypatch.setattr(loader, "_resolve_gguf_model_path", lambda gguf_model, revision: "resolved.gguf")
    monkeypatch.setattr(loader_module, "get_gguf_adapter", lambda gguf_file, model, source, od_config: _Adapter())

    weights = list(loader._get_gguf_weights_iterator(source, object(), od_config))

    assert weights[0][0] == "transformer.weight"
    assert torch.equal(weights[0][1], torch.ones((2, 2)))
    assert weights[1][0] == "transformer.bias"
    assert torch.equal(weights[1][1], torch.zeros(2))


def test_load_weights_with_gguf_falls_back_only_for_missing_transformer_weights(monkeypatch: pytest.MonkeyPatch):
    loader = _make_loader()
    model = _DummyModel()
    od_config = SimpleNamespace()

    monkeypatch.setattr(loader, "_get_weight_sources", lambda _model: tuple(model.weights_sources))
    monkeypatch.setattr(
        loader,
        "_get_gguf_weights_iterator",
        lambda source, model, od_config: iter([("transformer.weight", torch.ones((2, 2), dtype=torch.float32))]),
    )

    def _hf_iter(source):
        if source.subfolder == "transformer":
            return iter([("transformer.bias", torch.zeros(2, dtype=torch.float32))])
        return iter([("vae.weight", torch.full((2, 2), 2.0, dtype=torch.float32))])

    monkeypatch.setattr(loader, "_get_weights_iterator", _hf_iter)
    monkeypatch.setattr(
        loader,
        "_get_expected_parameter_names",
        lambda _model: {"transformer.weight", "transformer.bias", "vae.weight"},
    )

    loaded = loader._load_weights_with_gguf(model, od_config)

    assert loaded == {"transformer.weight", "transformer.bias", "vae.weight"}
    assert model.calls[0] == ["transformer.weight"]
    assert model.calls[1] == ["transformer.bias"]
    assert model.calls[2] == ["vae.weight"]


def test_load_weights_with_gguf_skips_transformer_fallback_when_gguf_is_complete(monkeypatch: pytest.MonkeyPatch):
    loader = _make_loader()
    model = _DummyModel()
    od_config = SimpleNamespace()

    monkeypatch.setattr(loader, "_get_weight_sources", lambda _model: tuple(model.weights_sources))
    monkeypatch.setattr(
        loader,
        "_get_gguf_weights_iterator",
        lambda source, model, od_config: iter(
            [
                ("transformer.weight", torch.ones((2, 2), dtype=torch.float32)),
                ("transformer.bias", torch.zeros(2, dtype=torch.float32)),
            ]
        ),
    )

    hf_calls: list[str] = []

    def _hf_iter(source):
        hf_calls.append(source.subfolder or "")
        if source.subfolder == "transformer":
            return iter([("transformer.bias", torch.zeros(2, dtype=torch.float32))])
        return iter([("vae.weight", torch.full((2, 2), 2.0, dtype=torch.float32))])

    monkeypatch.setattr(loader, "_get_weights_iterator", _hf_iter)
    monkeypatch.setattr(
        loader,
        "_get_expected_parameter_names",
        lambda _model: {"transformer.weight", "transformer.bias", "vae.weight"},
    )

    loaded = loader._load_weights_with_gguf(model, od_config)

    assert loaded == {"transformer.weight", "transformer.bias", "vae.weight"}
    assert hf_calls == ["vae"]
