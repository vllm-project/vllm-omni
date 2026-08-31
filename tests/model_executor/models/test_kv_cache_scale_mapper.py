# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for vLLM's KV-cache scale mapper load path."""

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STALE_API_FILES = [
    "vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py",
    "vllm_omni/model_executor/models/mammoth_moda2/mammoth_moda2.py",
    "vllm_omni/model_executor/models/mimo_audio/mimo_audio_llm.py",
    "vllm_omni/model_executor/models/qwen2_5_omni/qwen2_old.py",
]

_SOURCE_SCALE_NAME = "layers.0.self_attn.k_proj.output_scale"
_MAPPED_SCALE_NAME = "layers.0.self_attn.attn.k_scale"


class _RecordingCacheScaleMapper:
    def __init__(self) -> None:
        self.called = False

    def apply(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        self.called = True
        for name, weight in weights:
            yield name.replace(".k_proj.output_scale", ".attn.k_scale"), weight


class _QuantConfigWithoutOldCacheScale:
    def __init__(self) -> None:
        self.mapper = _RecordingCacheScaleMapper()

    def get_cache_scale_mapper(self) -> _RecordingCacheScaleMapper:
        return self.mapper


class _FakeModel:
    def __init__(self, quant_config: _QuantConfigWithoutOldCacheScale) -> None:
        self.quant_config = quant_config
        self.param = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def named_parameters(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.nn.Parameter]]:
        del remove_duplicate
        return [(_MAPPED_SCALE_NAME, self.param)]


class _FakeHunyuanModel(_FakeModel):
    def __init__(self, quant_config: _QuantConfigWithoutOldCacheScale) -> None:
        super().__init__(quant_config)
        self.config = SimpleNamespace(
            tie_word_embeddings=False,
            num_attention_heads=1,
            num_key_value_heads=1,
            use_cla=False,
        )

    def get_expert_mapping(self) -> tuple[list[object], dict[str, object]]:
        return [], {}

    def _split_qkv_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight


@pytest.mark.parametrize("rel_path", _STALE_API_FILES)
def test_model_loaders_do_not_call_removed_get_cache_scale(rel_path: str) -> None:
    """Affected loaders must not call the removed vLLM get_cache_scale API."""
    source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert ".get_cache_scale(" not in source


@pytest.mark.parametrize(
    ("module_name", "class_name", "fake_model_cls"),
    [
        (
            "vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3",
            "HunyuanModel",
            _FakeHunyuanModel,
        ),
        (
            "vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old",
            "Qwen2Model",
            _FakeModel,
        ),
    ],
)
def test_auto_weights_loader_delegated_loaders_accept_mapped_cache_scale(
    module_name: str,
    class_name: str,
    fake_model_cls: type[_FakeModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegated loaders must not expect the removed get_cache_scale method."""
    module = __import__(module_name, fromlist=[class_name])
    monkeypatch.setattr(module, "is_pp_missing_parameter", lambda *_: False)

    fake_model = fake_model_cls(_QuantConfigWithoutOldCacheScale())
    load_weights = getattr(module, class_name).load_weights

    loaded = load_weights(fake_model, [(_MAPPED_SCALE_NAME, torch.ones(()))])

    assert loaded == {_MAPPED_SCALE_NAME}
    torch.testing.assert_close(fake_model.param.data, torch.ones(()))


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        (
            "vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2",
            "MammothModa2Qwen2ForCausalLM",
        ),
        (
            "vllm_omni.model_executor.models.mimo_audio.mimo_audio_llm",
            "MiMoAudioLLMForConditionalGeneration",
        ),
    ],
)
def test_direct_custom_loaders_apply_cache_scale_mapper(
    module_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct custom loaders must apply the new mapper before fall-through load."""
    module = __import__(module_name, fromlist=[class_name])
    monkeypatch.setattr(module, "is_pp_missing_parameter", lambda *_: False)

    quant_config = _QuantConfigWithoutOldCacheScale()
    fake_model = _FakeModel(quant_config)
    load_weights = getattr(module, class_name).load_weights

    loaded = load_weights(fake_model, [(_SOURCE_SCALE_NAME, torch.ones(()))])

    assert quant_config.mapper.called
    assert loaded == {_MAPPED_SCALE_NAME}
    torch.testing.assert_close(fake_model.param.data, torch.ones(()))
