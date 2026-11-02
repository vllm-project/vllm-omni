# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Checkpoint projection loading must not require CPU LAPACK initialization."""

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored import ear_tts_model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embedding


def _model(monkeypatch, **kwargs):
    monkeypatch.setattr(ear_tts_model.AutoConfig, "for_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(ear_tts_model.AutoModel, "from_config", lambda config: _Backbone())
    monkeypatch.setattr(ear_tts_model, "find_and_delete_module", lambda *args: None)
    monkeypatch.setattr(ear_tts_model, "MoGHead", lambda **kwargs: nn.Identity())
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
        return ear_tts_model.RVQEARTTSModel(config, **kwargs)


def test_checkpoint_projection_restores_buffer_without_cpu_qr(monkeypatch):
    def no_lapack(*args, **kwargs):
        raise RuntimeError("CPU QR requires LAPACK")

    monkeypatch.setattr(torch.linalg, "qr", no_lapack)
    model = _model(monkeypatch, initialize_audio_prompt_projection=False)
    assert model.audio_prompt_projection_W.shape == (4, 4)
    assert model.audio_prompt_projection_W.dtype == torch.float32
    assert "audio_prompt_projection_W" in dict(model.named_buffers())
    assert "audio_prompt_projection_W" not in dict(model.named_parameters())

    # Inference loads through DuplexEARTTS, whose parent-module traversal
    # reports missing child buffers. RVQEARTTSModel.load_state_dict itself
    # intentionally retries partial training initialization after an error.
    container = nn.Module()
    container.add_module("tts_model", model)
    checkpoint = container.state_dict()
    checkpoint.pop("tts_model.audio_prompt_projection_W")
    missing, unexpected = container.load_state_dict(checkpoint, strict=False)
    assert missing == ["tts_model.audio_prompt_projection_W"]
    assert unexpected == []
    with pytest.raises(RuntimeError, match="audio_prompt_projection_W"):
        container.load_state_dict(checkpoint)

    projection = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    checkpoint["tts_model.audio_prompt_projection_W"] = projection
    container.load_state_dict(checkpoint)
    torch.testing.assert_close(model.audio_prompt_projection_W, projection, atol=0, rtol=0)


def test_default_constructor_keeps_random_projection_initialization(monkeypatch):
    calls = []

    def qr(matrix):
        calls.append(matrix.shape)
        return torch.eye(4), torch.eye(4)

    monkeypatch.setattr(torch.linalg, "qr", qr)
    model = _model(monkeypatch)
    assert calls == [torch.Size([4, 4]), torch.Size([4, 4])]
    diagonal = model.audio_prompt_projection_W.diag()
    assert torch.all((diagonal >= 0.4) & (diagonal <= 2.5))
    torch.testing.assert_close(model.audio_prompt_projection_W, torch.diag(diagonal), atol=0, rtol=0)
