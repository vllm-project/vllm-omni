# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.aura_omni.qwen3_vl import (
    AuraQwen3VLForConditionalGeneration,
    AuraQwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm_omni.model_executor.models.registry import OmniModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCtx:
    def __init__(self, config):
        self.config = config
        self.processor_type = None

    def get_hf_config(self, typ=None):
        if typ is not None:
            raise TypeError("wrong config class")
        return self.config

    def get_hf_processor(self, typ=None, **kwargs):
        self.processor_type = typ
        return SimpleNamespace(kwargs=kwargs)


def test_aura_qwen3_vl_processing_accepts_remote_qwen3_vl_config():
    info = object.__new__(AuraQwen3VLProcessingInfo)
    info.ctx = _FakeCtx(
        SimpleNamespace(
            model_type="qwen3_vl",
            vision_config=SimpleNamespace(spatial_merge_size=2),
            text_config=SimpleNamespace(hidden_size=4096),
        )
    )

    assert info.get_hf_config().model_type == "qwen3_vl"


def test_aura_qwen3_vl_processing_rejects_unrelated_remote_config():
    info = object.__new__(AuraQwen3VLProcessingInfo)
    info.ctx = _FakeCtx(SimpleNamespace(model_type="not_qwen3_vl"))

    with pytest.raises(TypeError, match="wrong config class"):
        info.get_hf_config()


def test_aura_qwen3_vl_model_arch_registered():
    model_cls = OmniModelRegistry._try_load_model_cls("AuraQwen3VLForConditionalGeneration")

    assert model_cls is AuraQwen3VLForConditionalGeneration


def test_aura_qwen3_vl_processing_forces_upstream_processor_class():
    info = object.__new__(AuraQwen3VLProcessingInfo)
    ctx = _FakeCtx(SimpleNamespace(model_type="qwen3_vl"))
    info.ctx = ctx

    processor = info.get_hf_processor(extra="value")

    assert ctx.processor_type is Qwen3VLProcessor
    assert processor.kwargs["use_fast"] is True
    assert processor.kwargs["extra"] == "value"
