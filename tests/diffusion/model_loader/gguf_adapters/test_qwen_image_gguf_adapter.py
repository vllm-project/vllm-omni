# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
from vllm_omni.diffusion.model_loader.gguf_adapters.qwen_image import (
    QwenImageGGUFAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_od_config(
    model_class_name: str = "QwenImagePipeline",
    model_type: str = "qwen_image",
):
    return SimpleNamespace(
        model_class_name=model_class_name,
        tf_model_config={"model_type": model_type},
    )


def _make_source(prefix: str = "", subfolder: str = "transformer"):
    return SimpleNamespace(prefix=prefix, subfolder=subfolder)


class _FakeTransformer:
    def __init__(self) -> None:
        self._params = {
            "img_in.qweight": 1,
            "img_in.qweight_type": 1,
            "txt_in.qweight": 1,
            "txt_in.qweight_type": 1,
            "proj_out.qweight": 1,
            "proj_out.qweight_type": 1,
            "time_text_embed.timestep_embedder.linear_1.qweight": 1,
            "time_text_embed.timestep_embedder.linear_1.qweight_type": 1,
            "time_text_embed.timestep_embedder.linear_2.qweight": 1,
            "time_text_embed.timestep_embedder.linear_2.qweight_type": 1,
            "transformer_blocks.0.img_mod.1.qweight": 1,
            "transformer_blocks.0.img_mod.1.qweight_type": 1,
            "transformer_blocks.0.txt_mod.1.qweight": 1,
            "transformer_blocks.0.txt_mod.1.qweight_type": 1,
            "transformer_blocks.0.attn.to_qkv.qweight": 1,
            "transformer_blocks.0.attn.to_qkv.qweight_type": 1,
            "transformer_blocks.0.attn.add_kv_proj.qweight": 1,
            "transformer_blocks.0.attn.add_kv_proj.qweight_type": 1,
            "transformer_blocks.0.attn.to_out.weight": 1,
            "transformer_blocks.0.img_mlp.net.0.proj.qweight": 1,
            "transformer_blocks.0.img_mlp.net.0.proj.qweight_type": 1,
        }

    def named_parameters(self):
        return list(self._params.items())

    def named_buffers(self):
        return []

    def get_submodule(self, _prefix: str):
        return self


def test_qwen_adapter_selected_for_qwen_image_family():
    adapter = get_gguf_adapter(
        "dummy.gguf",
        _FakeTransformer(),
        _make_source(),
        _make_od_config(),
    )
    assert isinstance(adapter, QwenImageGGUFAdapter)


def test_qwen_adapter_matches_multiple_pipeline_variants():
    for model_class_name in (
        "QwenImagePipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageLayeredPipeline",
    ):
        assert QwenImageGGUFAdapter.is_compatible(
            _make_od_config(model_class_name=model_class_name),
            _FakeTransformer(),
            _make_source(),
        )


def test_qwen_adapter_preserves_split_projection_names(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.model_loader.gguf_adapters.qwen_image as qwen_image_module

    monkeypatch.setattr(
        qwen_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("transformer_blocks.0.attn.to_q.qweight_type", 1),
                ("transformer_blocks.0.attn.to_q.qweight", 2),
                ("transformer_blocks.0.attn.to_k.qweight_type", 3),
                ("transformer_blocks.0.attn.to_k.qweight", 4),
                ("transformer_blocks.0.attn.to_out.0.qweight_type", 5),
                ("transformer_blocks.0.attn.to_out.0.qweight", 6),
            ]
        ),
    )

    adapter = QwenImageGGUFAdapter(
        "dummy.gguf",
        _FakeTransformer(),
        _make_source(),
        _make_od_config(),
    )

    weights = list(adapter.weights_iterator())

    assert ("transformer_blocks.0.attn.to_q.qweight_type", 1) in weights
    assert ("transformer_blocks.0.attn.to_q.qweight", 2) in weights
    assert ("transformer_blocks.0.attn.to_k.qweight_type", 3) in weights
    assert ("transformer_blocks.0.attn.to_k.qweight", 4) in weights
    assert ("transformer_blocks.0.attn.to_out.0.qweight_type", 5) in weights
    assert ("transformer_blocks.0.attn.to_out.0.qweight", 6) in weights


def test_qwen_adapter_keeps_top_level_quantized_weights(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.model_loader.gguf_adapters.qwen_image as qwen_image_module

    monkeypatch.setattr(
        qwen_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("img_in.qweight_type", 1),
                ("img_in.qweight", 2),
                ("time_text_embed.timestep_embedder.linear_1.qweight_type", 3),
                ("time_text_embed.timestep_embedder.linear_1.qweight", 4),
            ]
        ),
    )

    adapter = QwenImageGGUFAdapter(
        "dummy.gguf",
        _FakeTransformer(),
        _make_source(),
        _make_od_config(),
    )

    weights = list(adapter.weights_iterator())

    assert ("img_in.qweight_type", 1) in weights
    assert ("img_in.qweight", 2) in weights
    assert ("time_text_embed.timestep_embedder.linear_1.qweight_type", 3) in weights
    assert ("time_text_embed.timestep_embedder.linear_1.qweight", 4) in weights


def test_qwen_adapter_keeps_modulation_quantized_weights(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.model_loader.gguf_adapters.qwen_image as qwen_image_module

    monkeypatch.setattr(
        qwen_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("transformer_blocks.0.img_mod.1.qweight_type", 1),
                ("transformer_blocks.0.img_mod.1.qweight", 2),
                ("transformer_blocks.0.txt_mod.1.qweight_type", 3),
                ("transformer_blocks.0.txt_mod.1.qweight", 4),
            ]
        ),
    )

    adapter = QwenImageGGUFAdapter(
        "dummy.gguf",
        _FakeTransformer(),
        _make_source(),
        _make_od_config(),
    )

    weights = list(adapter.weights_iterator())

    assert ("transformer_blocks.0.img_mod.1.qweight_type", 1) in weights
    assert ("transformer_blocks.0.img_mod.1.qweight", 2) in weights
    assert ("transformer_blocks.0.txt_mod.1.qweight_type", 3) in weights
    assert ("transformer_blocks.0.txt_mod.1.qweight", 4) in weights
