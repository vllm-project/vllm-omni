# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Omni-Diffusion S2I request and media helpers."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import vllm_omni.diffusion.models.omni_diffusion.pipeline_s2i as s2i_module
from vllm_omni.diffusion.models.omni_diffusion.pipeline_s2i import (
    _decode_audio_source,
    _get_audio_source,
    _get_prompt_text,
    _image_tensor_to_pil,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(
    prompt: str | dict = "generate an image",
    **extra_args: object,
) -> DiffusionRequestBatch:
    return DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(
                    seed=1,
                    extra_args=dict(extra_args),
                ),
                request_id="test-s2i",
            )
        ]
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("draw a cat", "draw a cat"),
        ({"prompt": "draw a dog"}, "draw a dog"),
        ({"prompt": None}, ""),
        ({}, ""),
    ],
)
def test_get_prompt_text(prompt: str | dict, expected: str) -> None:
    assert _get_prompt_text(prompt) == expected


@pytest.mark.parametrize("prompt", [123, {"prompt": 123}])
def test_get_prompt_text_rejects_non_string_values(prompt: object) -> None:
    with pytest.raises(TypeError, match="prompt"):
        _get_prompt_text(prompt)  # type: ignore[arg-type]


def test_get_audio_source_prefers_extra_args() -> None:
    req = _request(
        {"prompt": "draw", "multi_modal_data": {"audio": "ignored.wav"}},
        audio_path="selected.wav",
    )
    assert _get_audio_source(req) == "selected.wav"


def test_get_audio_source_accepts_audio_url() -> None:
    assert _get_audio_source(_request(audio_url="file:///audio.wav")) == "file:///audio.wav"


def test_get_audio_source_reads_multimodal_prompt() -> None:
    req = _request({"prompt": "draw", "multi_modal_data": {"audio": "input.wav"}})
    assert _get_audio_source(req) == "input.wav"


def test_get_audio_source_rejects_invalid_or_missing_audio() -> None:
    with pytest.raises(TypeError, match="string"):
        _get_audio_source(_request(audio_path=123))
    with pytest.raises(ValueError, match="requires an audio input"):
        _get_audio_source(_request())


def test_decode_audio_source_uses_media_connector_for_data_uri(monkeypatch) -> None:
    expected = np.zeros(16, dtype=np.float32)

    class FakeMediaConnector:
        def fetch_audio(self, source: str) -> tuple[np.ndarray, int]:
            assert source == "data:audio/wav;base64,AAAA"
            return expected, 16000

    monkeypatch.setattr(s2i_module, "MediaConnector", FakeMediaConnector)

    audio, sample_rate = _decode_audio_source("data:audio/wav;base64,AAAA")

    torch.testing.assert_close(audio, torch.from_numpy(expected))
    assert audio.is_contiguous()
    assert sample_rate == 16000


@pytest.mark.parametrize("as_file_url", [False, True])
def test_decode_audio_source_loads_local_file(
    monkeypatch,
    tmp_path: Path,
    as_file_url: bool,
) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.touch()
    expected = np.zeros((2, 16), dtype=np.float32)

    def fake_load_audio(path: str, *, sr: int | None, mono: bool):
        assert path == str(audio_path)
        assert sr is None
        assert mono is False
        return expected, 44100

    monkeypatch.setattr(s2i_module, "load_audio", fake_load_audio)
    source = audio_path.as_uri() if as_file_url else str(audio_path)

    audio, sample_rate = _decode_audio_source(source)

    torch.testing.assert_close(audio, torch.from_numpy(expected))
    assert sample_rate == 44100


def test_decode_audio_source_rejects_http_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        _decode_audio_source("https://example.com/input.wav")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _decode_audio_source(str(tmp_path / "missing.wav"))


def test_image_tensor_to_pil_converts_normalized_batch() -> None:
    tensor = torch.zeros((1, 3, 2, 3), dtype=torch.float32)
    tensor[:, 0] = 1.0

    image = _image_tensor_to_pil(tensor)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    "shape",
    [(3, 4, 4), (2, 3, 4, 4), (1, 1, 4, 4)],
)
def test_image_tensor_to_pil_rejects_invalid_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        _image_tensor_to_pil(torch.zeros(shape))
