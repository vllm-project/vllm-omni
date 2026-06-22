# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for StepAudioEditX Code2Wav wrapper."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.model_executor.models.step_audio_editx.step_audio_decoder import (
    CosyVoice,
    StepAudioCode2wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))


class _FakeCore:
    def __init__(self) -> None:
        self.flow = _FakeFlow()
        self.feature_extract_calls: list[tuple[torch.Tensor, int]] = []
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.forward_chunk_calls: list[tuple[torch.Tensor, torch.Tensor, str | None, bool]] = []

    def _feature_extract(self, input_wav: torch.Tensor, sample_rate: int):
        self.feature_extract_calls.append((input_wav.detach().clone(), sample_rate))
        return (
            torch.ones((1, 4, 80), dtype=torch.float32),
            torch.ones((1, 192), dtype=torch.float32) * 2,
        )

    def forward(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_embedding: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls.append((token.clone(), prompt_token.clone(), speech_feat.clone(), speech_embedding.clone()))
        return torch.arange(6, dtype=torch.float32).reshape(1, 6)

    def forward_chunk(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_embedding: torch.Tensor,
        session_id: str | None,
        last_chunk: bool,
    ) -> torch.Tensor | None:
        self.forward_chunk_calls.append((token.clone(), prompt_token.clone(), session_id, last_chunk))
        if last_chunk:
            return torch.arange(4, dtype=torch.float32).reshape(1, 4)
        return None


def _make_model(*, async_chunk: bool = False) -> StepAudioCode2wav:
    model = StepAudioCode2wav.__new__(StepAudioCode2wav)
    torch.nn.Module.__init__(model)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk))
    model.core = _FakeCore()
    model.prompt_feature_cache = {}
    return model


def test_reshape_pads_to_mixed_codec_groups() -> None:
    out = CosyVoice._reshape([1, 2, 1030])

    expected = torch.tensor(
        [
            [1, 1031],
            [2, 1025],
            [1024, 1025],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(out, expected)


def test_extract_prompt_token_accepts_tensor_and_strips_batch_dim() -> None:
    ref = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = StepAudioCode2wav._extract_prompt_token([{"codes": {"ref": ref}}], {})

    assert out is not None
    assert out.tolist() == [1, 2, 3]


def test_extract_runtime_inputs_rejects_batched_reference_audio() -> None:
    with patch(
        "vllm_omni.model_executor.models.step_audio_editx.step_audio_decoder.StepAudioTokenizer._load_audio",
        return_value=[(torch.zeros(1, 8), 16000), (torch.zeros(1, 8), 16000)],
    ):
        with pytest.raises(ValueError, match="expects one reference audio"):
            StepAudioCode2wav._extract_runtime_inputs([{"latent": ["a", "b"]}], {})


def test_extract_runtime_inputs_accepts_ref_audio_before_latent() -> None:
    ref_audio = torch.ones((1, 8), dtype=torch.float32)

    with patch(
        "vllm_omni.model_executor.models.step_audio_editx.step_audio_decoder.StepAudioTokenizer._load_audio",
        return_value=(ref_audio, 16000),
    ) as load_audio:
        out_audio, out_sr = StepAudioCode2wav._extract_runtime_inputs(
            [{"ref_audio": "ref.wav", "latent": "latent.wav"}],
            {},
        )

    load_audio.assert_called_once_with("ref.wav", None)
    torch.testing.assert_close(out_audio, ref_audio)
    assert out_sr == 16000


def test_sync_forward_extracts_features_and_decodes_audio() -> None:
    model = _make_model(async_chunk=False)
    ref_audio = torch.ones((1, 8), dtype=torch.float32)

    with patch.object(model, "preprocess_wav", return_value=(ref_audio, 16000)) as preprocess:
        out = model.forward(
            input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            sample_rate=16000,
            runtime_additional_information=[
                {
                    "latent": ref_audio,
                    "codes": {"ref": torch.tensor([9, 8, 7], dtype=torch.long)},
                }
            ],
        )

    preprocess.assert_called_once()
    assert out.multimodal_outputs["audio"].shape == (1, 6)
    token, prompt_token, speech_feat, speech_embedding = model.core.forward_calls[-1]
    assert token.tolist() == [1, 2, 3, 4, 5]
    assert prompt_token.tolist() == [9, 8, 7]
    assert speech_feat.dtype == torch.float32
    assert speech_embedding.dtype == torch.float32


def test_async_forward_caches_conditioning_until_last_chunk() -> None:
    model = _make_model(async_chunk=True)
    ref_audio = torch.ones((1, 8), dtype=torch.float32)

    with patch.object(model, "preprocess_wav", return_value=(ref_audio, 16000)) as preprocess:
        first = model.forward(
            input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            sample_rate=16000,
            runtime_additional_information=[
                {
                    "latent": ref_audio,
                    "codes": {"ref": torch.tensor([9, 8, 7], dtype=torch.long)},
                    "meta": {"req_id": "rid", "stream_finished": False},
                }
            ],
        )
        second = model.forward(
            input_ids=torch.tensor([6, 7, 8, 9, 10], dtype=torch.long),
            sample_rate=16000,
            runtime_additional_information=[
                {
                    "meta": {"req_id": "rid", "stream_finished": True},
                }
            ],
        )

    preprocess.assert_called_once()
    assert first.multimodal_outputs == {}
    assert second.multimodal_outputs["audio"].shape == (1, 4)
    assert len(model.core.feature_extract_calls) == 1
    assert [call[2:] for call in model.core.forward_chunk_calls] == [
        ("rid", False),
        ("rid", True),
    ]
    assert "rid" not in model.prompt_feature_cache


def test_forward_requires_input_ids() -> None:
    model = _make_model()

    with pytest.raises(ValueError, match="requires input_ids"):
        model.forward(input_ids=None)
