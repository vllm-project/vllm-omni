# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the NemotronVoiceChat thinker text-config adapter.

The thinker backbone config is resolved at runtime from another repo
(``nvidia/NVIDIA-Nemotron-Nano-9B-v2`` or ``NEMOTRON_VOICECHAT_LLM_PATH``).
vLLM only inspects the object returned by ``get_text_config()`` and requires
``num_attention_heads`` on it, so the adapter must normalize whatever it gets.
"""

import pytest
from transformers import PretrainedConfig

from vllm_omni.model_executor.models.nemotron_voicechat.configuration_nemotron_voicechat import (
    NemotronVoiceChatConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_nemo_dict() -> dict:
    return {
        "data": {"frame_length": 0.08, "source_sample_rate": 16000, "target_sample_rate": 22050},
        "model": {
            "inference_speaker_name": "Aria",
            "stt": {
                "model": {
                    "perception": {"encoder": {}, "preprocessor": {}, "modality_adapter": {}},
                    "pretrained_llm": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                }
            },
            "speech_generation": {
                "model": {
                    "tts_config": {"backbone_config": {"hidden_size": 1152}},
                    "codec_config": {"num_quantizers": 31, "codebook_size": 1024},
                }
            },
        },
    }


def _config_with_backbone(monkeypatch, backbone: PretrainedConfig) -> NemotronVoiceChatConfig:
    """Build the config with a stub backbone resolver (no hub access)."""
    from vllm_omni.model_executor.models.nemotron_voicechat import configuration_nemotron_voicechat as mod

    monkeypatch.setattr(
        mod.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: backbone),
    )
    return NemotronVoiceChatConfig(**_minimal_nemo_dict())


def test_plain_backbone_is_returned_unchanged(monkeypatch) -> None:
    backbone = PretrainedConfig(num_attention_heads=40, hidden_size=4480)
    cfg = _config_with_backbone(monkeypatch, backbone)
    assert cfg.get_text_config() is backbone


def test_composite_backbone_is_unwrapped(monkeypatch) -> None:
    # Shape seen when the runtime resolver lands on a VL-style config: the head
    # counts live one level down, which vLLM's single-level check would miss.
    inner = PretrainedConfig(num_attention_heads=40, hidden_size=4480)
    composite = PretrainedConfig()
    composite.text_config = inner

    cfg = _config_with_backbone(monkeypatch, composite)
    resolved = cfg.get_text_config()
    assert resolved is inner
    assert resolved.num_attention_heads == 40


def test_unusable_backbone_raises_actionable_error(monkeypatch) -> None:
    cfg = _config_with_backbone(monkeypatch, PretrainedConfig())
    with pytest.raises(ValueError, match="num_attention_heads"):
        cfg.get_text_config()
    with pytest.raises(ValueError, match="NVIDIA-Nemotron-Nano-9B-v2"):
        cfg.get_text_config()


def test_skeleton_config_fails_loudly() -> None:
    """Bare construction (no NeMo section) must not silently hand out a stub."""
    cfg = NemotronVoiceChatConfig()
    with pytest.raises(ValueError, match="num_attention_heads"):
        cfg.get_text_config()


def test_double_nested_backbone_is_unwrapped(monkeypatch) -> None:
    inner = PretrainedConfig(num_attention_heads=40, hidden_size=4480)
    middle = PretrainedConfig()
    middle.text_config = inner
    outer = PretrainedConfig()
    outer.text_config = middle

    cfg = _config_with_backbone(monkeypatch, outer)
    assert cfg.get_text_config() is inner


def test_placeholder_heads_does_not_stop_unwrapping(monkeypatch) -> None:
    # An outer composite may carry a placeholder `num_attention_heads = None`
    # while the real head counts live on the nested `text_config`.
    inner = PretrainedConfig(num_attention_heads=40, hidden_size=4480)
    outer = PretrainedConfig(num_attention_heads=None)
    outer.text_config = inner

    cfg = _config_with_backbone(monkeypatch, outer)
    assert cfg.get_text_config() is inner
