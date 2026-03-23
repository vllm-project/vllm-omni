# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Z-Image GGUF adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
from vllm_omni.diffusion.model_loader.gguf_adapters.z_image import (
    ZImageGGUFAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_od_config(
    model_class_name: str = "ZImagePipeline",
    model_type: str = "z_image",
):
    return SimpleNamespace(
        model_class_name=model_class_name,
        tf_model_config={"model_type": model_type},
    )


def _make_source(prefix: str = "", subfolder: str = "transformer"):
    return SimpleNamespace(prefix=prefix, subfolder=subfolder)


def test_z_image_adapter_selected_for_z_image_family():
    adapter = get_gguf_adapter(
        "dummy.gguf",
        object(),
        _make_source(),
        _make_od_config(),
    )

    assert isinstance(adapter, ZImageGGUFAdapter)


def test_z_image_adapter_matches_model_type_variants():
    for model_type in ("z_image", "zimage", "z-image"):
        assert ZImageGGUFAdapter.is_compatible(
            _make_od_config(model_class_name="OtherPipeline", model_type=model_type),
            object(),
            _make_source(),
        )


def test_z_image_adapter_renames_known_gguf_tensor_paths(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.model_loader.gguf_adapters.z_image as z_image_module

    monkeypatch.setattr(
        z_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("model.diffusion_model.final_layer.qweight", torch.ones((1, 1))),
                ("model.diffusion_model.x_embedder.qweight_type", torch.tensor(1)),
                ("transformer_blocks.0.attention.out.weight", torch.full((1, 1), 2.0)),
                ("transformer_blocks.0.attention.qkv.qweight", torch.full((1, 1), 3.0)),
            ]
        ),
    )

    adapter = ZImageGGUFAdapter(
        "dummy.gguf",
        object(),
        _make_source(),
        _make_od_config(),
    )

    weights = list(adapter.weights_iterator())
    names = [name for name, _ in weights]

    assert "all_final_layer.2-1.qweight" in names
    assert "all_x_embedder.2-1.qweight_type" in names
    assert "transformer_blocks.0.attention.to_out.0.weight" in names
    assert "transformer_blocks.0.attention.to_qkv.qweight" in names
