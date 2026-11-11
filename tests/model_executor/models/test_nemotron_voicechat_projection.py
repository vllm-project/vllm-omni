# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Checkpoint projection loading must avoid CPU QR and fail closed."""

from dataclasses import dataclass, field

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored import duplex_ear_tts, ear_tts_model
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_talker import (
    NemotronVoiceChatTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _ProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = {"use_audio_prompt_frozen_projection": True}
        self.register_buffer("audio_prompt_projection_W", torch.full((2, 2), float("nan")))
        self.other = nn.Parameter(torch.zeros(2))


@dataclass
class _TalkerConfig:
    tts_cfg: dict[str, object] = field(
        default_factory=lambda: {
            "tts_config": {},
            "inference_guidance_enabled": False,
        }
    )
    tts_data: dict[str, object] = field(default_factory=dict)
    target_sample_rate: int = 22050
    frame_length: float = 0.08


@dataclass
class _DeviceConfig:
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


@dataclass
class _VllmConfig:
    device_config: _DeviceConfig = field(default_factory=_DeviceConfig)


def _projection_model(monkeypatch: pytest.MonkeyPatch) -> ear_tts_model.RVQEARTTSModel:
    monkeypatch.setattr(ear_tts_model.AutoConfig, "for_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ear_tts_model.AutoModel, "from_config", lambda _config: _Backbone())
    monkeypatch.setattr(ear_tts_model, "find_and_delete_module", lambda *_args: None)
    monkeypatch.setattr(ear_tts_model, "MoGHead", lambda **_kwargs: nn.Identity())
    config = DictConfig(
        {
            "backbone_type": "test",
            "backbone_config": {},
            "random_target_masking": False,
            "latent_size": 2,
            "context_hidden_size": None,
            "cas_config": None,
            "use_gated_fusion_for_text_audio": False,
            "use_audio_prompt_frozen_projection": True,
            "disable_eos_prediction": True,
            "mog_head_config": {},
        }
    )
    with torch.device("cpu"):
        return ear_tts_model.RVQEARTTSModel(config)


def _duplex_with_projection() -> duplex_ear_tts.DuplexEARTTS:
    duplex = duplex_ear_tts.DuplexEARTTS.__new__(duplex_ear_tts.DuplexEARTTS)
    nn.Module.__init__(duplex)
    duplex.tts_model = _ProjectionModel()
    duplex.audio_codec = None
    duplex.audio_prompt_latents = nn.ParameterDict()
    return duplex


def _talker_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[NemotronVoiceChatTalkerForConditionalGeneration, duplex_ear_tts.DuplexEARTTS]:
    duplex = _duplex_with_projection()

    def build_duplex(_cfg: dict[str, object]) -> duplex_ear_tts.DuplexEARTTS:
        return duplex

    monkeypatch.setattr(duplex_ear_tts, "DuplexEARTTS", build_duplex)
    talker = NemotronVoiceChatTalkerForConditionalGeneration.__new__(NemotronVoiceChatTalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.config = _TalkerConfig()
    talker.vllm_config = _VllmConfig()
    talker._dtype = torch.float32
    talker._speaker_name = "Aria"
    talker._use_native_backbone = False
    talker._native_guidance = False
    talker._use_cuda_graphs = False
    talker._step_graph = None
    talker._mog_graph = None
    return talker, duplex


def test_projection_constructor_skips_cpu_qr(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_qr(*_args, **_kwargs):
        raise AssertionError("CPU QR must not run for checkpoint-owned projection")

    monkeypatch.setattr(torch.linalg, "qr", reject_qr)
    model = _projection_model(monkeypatch)

    assert "audio_prompt_projection_W" in dict(model.named_buffers())
    assert model.audio_prompt_projection_W.dtype == torch.float32
    assert torch.isnan(model.audio_prompt_projection_W).all()


def test_partial_load_restores_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    talker, duplex = _talker_loading(monkeypatch)
    projection = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    talker.load_weights(
        [
            ("tts_model.tts_model.audio_prompt_projection_W", projection),
            # Force DuplexEARTTS.load_state_dict through its partial-init retry.
            ("tts_model.tts_model.other", torch.ones(3)),
        ]
    )

    torch.testing.assert_close(duplex.tts_model.audio_prompt_projection_W, projection, atol=0, rtol=0)


def test_partial_load_rejects_missing_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    talker, _ = _talker_loading(monkeypatch)

    with pytest.raises(ValueError, match="audio_prompt_projection_W.*refusing partial initialization"):
        talker.load_weights(
            [
                # The retry fills missing keys from the model's own state dict,
                # so this mismatch previously hid the absent projection.
                ("tts_model.tts_model.other", torch.ones(3)),
            ]
        )
