# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AR stage must register attention layers for its own language model only.

The Qwen-VL parent ``__init__`` builds a stock language model before MammothModa2
replaces it. Its attention layers stay in ``static_forward_context`` unless they are
removed, and the KV cache is then sized and allocated for both models. The parent
``__init__`` and the replacement are stubbed here so only the registry is exercised.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import StageMissingLayer

import vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 as mm2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_LAYERS = 3
VISION_LAYER = "ar.visual.blocks.0.attn"
AR_CLASSES = [
    (mm2.MammothModa2ARForConditionalGeneration, mm2.Qwen2_5_VLForConditionalGeneration),
    (mm2.MammothModa2Qwen3ARForConditionalGeneration, mm2.Qwen3VLForConditionalGeneration),
]
AR_IDS = ["preview_qwen2_5_vl", "dev_qwen3_vl"]


class _FakeAttention(nn.Module):
    def __init__(self, registry: dict, name: str):
        super().__init__()
        registry[name] = self


class _FakeLM(nn.Module):
    def __init__(self, registry: dict, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList(
            _FakeAttention(registry, f"{prefix}.layers.{i}.self_attn.attn") for i in range(NUM_LAYERS)
        )
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: None


class _FakeCompiledLM(_FakeLM):
    cleanups = 0

    def cleanup(self):
        type(self).cleanups += 1


def _build(
    monkeypatch,
    ar_class,
    parent,
    *,
    encoder_only=False,
    parent_lm=_FakeLM,
    parent_registers=True,
    pp_rank=(1, True),
    stray_parent_layer=False,
):
    registry: dict = {}
    built = {}

    def parent_init(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.visual = _FakeAttention(registry, VISION_LAYER)
        stock = nn.Module()
        stock.model = parent_lm(registry if parent_registers else {}, f"{prefix}.language_model.model")
        built["parent_lm"] = weakref.ref(stock)
        if stray_parent_layer:
            # A parent-prefixed layer that is not part of the module being dropped.
            registry[f"{prefix}.language_model.model.layers.9.self_attn.attn"] = nn.Module()
        self.language_model = StageMissingLayer("language_model", stock) if encoder_only else stock

    # The model module only imports get_current_vllm_config with this fix; keep the
    # reverted source testable so the registry assertion is what fails there.
    monkeypatch.setattr(
        mm2,
        "get_current_vllm_config",
        lambda: SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=registry)),
        raising=False,
    )
    monkeypatch.setattr(parent, "__init__", parent_init)
    monkeypatch.setattr(mm2, "init_vllm_registered_model", lambda *, prefix, **kwargs: _FakeLM(registry, prefix))
    monkeypatch.setattr(mm2, "MammothModa2Qwen3ForCausalLM", lambda *, vllm_config, prefix: _FakeLM(registry, prefix))
    monkeypatch.setattr(mm2, "TorchCompileWithNoGuardsWrapper", _FakeCompiledLM, raising=False)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    world_size, is_first_rank = pp_rank
    monkeypatch.setattr(
        mm2, "get_pp_group", lambda: SimpleNamespace(world_size=world_size, is_first_rank=is_first_rank)
    )

    hf_config = SimpleNamespace(llm_config=SimpleNamespace(text_config=SimpleNamespace()))
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, architectures=["MammothModa2Model"])
    )
    vllm_config.with_hf_config = lambda hf, architectures=None: vllm_config
    model = ar_class(vllm_config=vllm_config, prefix="ar")
    return model, registry, built


@pytest.mark.parametrize("encoder_only", [False, True], ids=["default", "mm_encoder_only"])
@pytest.mark.parametrize(("ar_class", "parent"), AR_CLASSES, ids=AR_IDS)
def test_ar_stage_registers_only_the_replacement_language_model(monkeypatch, ar_class, parent, encoder_only):
    model, registry, built = _build(monkeypatch, ar_class, parent, encoder_only=encoder_only)

    assert set(registry) == {f"ar.language_model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)} | {VISION_LAYER}
    gc.collect()
    assert built["parent_lm"]() is None


@pytest.mark.parametrize(("ar_class", "parent"), AR_CLASSES, ids=AR_IDS)
def test_compiled_parent_language_model_releases_its_hook(monkeypatch, ar_class, parent):
    _FakeCompiledLM.cleanups = 0
    _build(monkeypatch, ar_class, parent, parent_lm=_FakeCompiledLM)
    assert _FakeCompiledLM.cleanups == 1


@pytest.mark.parametrize(("ar_class", "parent"), AR_CLASSES, ids=AR_IDS)
def test_unexpected_parent_registry_fails_loudly(monkeypatch, ar_class, parent):
    with pytest.raises(RuntimeError, match="Unexpected attention registry"):
        _build(monkeypatch, ar_class, parent, parent_registers=False)


@pytest.mark.parametrize(("ar_class", "parent"), AR_CLASSES, ids=AR_IDS)
def test_later_pipeline_rank_without_parent_layers_starts(monkeypatch, ar_class, parent):
    _, registry, _ = _build(monkeypatch, ar_class, parent, parent_registers=False, pp_rank=(2, False))
    assert set(registry) == {f"ar.language_model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)} | {VISION_LAYER}


@pytest.mark.parametrize(("ar_class", "parent"), AR_CLASSES, ids=AR_IDS)
def test_first_pipeline_rank_without_parent_layers_fails(monkeypatch, ar_class, parent):
    with pytest.raises(RuntimeError, match="Unexpected attention registry"):
        _build(monkeypatch, ar_class, parent, parent_registers=False, pp_rank=(2, True))


@pytest.mark.parametrize(("ar_class", "parent"), AR_CLASSES, ids=AR_IDS)
def test_later_pipeline_rank_still_rejects_a_leftover_parent_layer(monkeypatch, ar_class, parent):
    with pytest.raises(RuntimeError, match="Unexpected attention registry"):
        _build(monkeypatch, ar_class, parent, parent_registers=False, pp_rank=(2, False), stray_parent_layer=True)
