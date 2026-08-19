# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image

from examples.offline_inference.x_to_video_audio import x_to_video_audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_builds_canonical_image_audio_to_video_prompt() -> None:
    image = Image.new("RGB", (16, 8), "red")
    audio = (np.arange(8, dtype=np.float32), 24000)

    prompt = x_to_video_audio.build_x_to_video_audio_prompt(
        "A person speaks to the camera.",
        {"image": [image], "audio": [audio]},
    )

    assert prompt == {
        "prompt": "A person speaks to the camera.",
        "modalities": ["video"],
        "multi_modal_data": {"image": [image], "audio": [audio]},
    }


def test_canonical_prompt_omits_empty_media() -> None:
    assert x_to_video_audio.build_x_to_video_audio_prompt("A landscape.") == {
        "prompt": "A landscape.",
        "modalities": ["video"],
    }


def test_official_prompt_cleanup_stays_in_prompt_file_handling() -> None:
    prompt = (
        "[SPEAKER_TIMESTAMPS_START]metadata[SPEAKER_TIMESTAMPS_END]\n\n"
        "A person waves.\n[AUDIO_DESCRIPTION_START]noise[AUDIO_DESCRIPTION_END]"
    )

    assert x_to_video_audio._clean_official_prompt(prompt) == "A person waves."


def test_media_loader_preserves_complete_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    Image.new("RGBA", (4, 3), "blue").save(image_path)
    complete_waveform = np.arange(80000, dtype=np.float32)
    requested_rates: list[int | None] = []

    def fake_load_audio(path: str, *, sr: int | None) -> tuple[np.ndarray, int]:
        assert path == "reference.wav"
        requested_rates.append(sr)
        return complete_waveform, 16000

    monkeypatch.setattr(x_to_video_audio, "load_audio", fake_load_audio)

    images, audios = x_to_video_audio.load_image_and_audio(
        [str(image_path)],
        ["reference.wav"],
        audio_sample_rate=16000,
    )

    assert images[0].mode == "RGB"
    assert requested_rates == [16000]
    assert audios[0][0] is complete_waveform
    assert audios[0][0].shape == (80000,)
    assert audios[0][1] == 16000


@pytest.mark.parametrize(
    "video",
    [
        np.zeros((3, 2, 4, 4), dtype=np.float32),
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
    ],
    ids=["channel_first_float", "frame_first_uint8"],
)
def test_output_contract_keeps_model_video_layout_opaque(video: np.ndarray) -> None:
    audio = np.zeros(16, dtype=np.float32)
    result = SimpleNamespace(
        images=[video],
        multimodal_output={"audio": audio, "fps": 25, "audio_sample_rate": 44100},
    )

    output = x_to_video_audio.extract_x_to_video_audio_output([result])

    assert output.video is video
    assert output.audio is audio
    assert output.fps == 25.0
    assert output.audio_sample_rate == 44100


def test_output_contract_accepts_video_from_multimodal_output() -> None:
    video = object()
    result = SimpleNamespace(images=[], multimodal_output={"video": video, "fps": 12})

    output = x_to_video_audio.extract_x_to_video_audio_output(result)

    assert output == x_to_video_audio.XToVideoAudioOutput(
        video=video,
        audio=None,
        fps=12.0,
        audio_sample_rate=None,
    )


def test_output_contract_requires_model_neutral_metadata() -> None:
    result = SimpleNamespace(images=[object()], multimodal_output={"audio": np.zeros(4)})

    with pytest.raises(RuntimeError, match="must declare a valid 'fps'"):
        x_to_video_audio.extract_x_to_video_audio_output(result)

    with pytest.raises(RuntimeError, match="must declare a valid 'audio_sample_rate'"):
        x_to_video_audio.extract_x_to_video_audio_output(result, fps=24)


def test_output_encoder_normalizes_declared_range_and_channel_first_layout() -> None:
    video = np.zeros((3, 2, 4, 6), dtype=np.float32)
    output = x_to_video_audio.XToVideoAudioOutput(
        video=video,
        audio=None,
        fps=8,
        audio_sample_rate=None,
        output_tensor_range="negative_one_to_one",
    )

    payload = x_to_video_audio.encode_x_to_video_audio_output(output)

    with av.open(io.BytesIO(payload)) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(decoded) == 2
    assert decoded[0].shape == (4, 6, 3)
    np.testing.assert_allclose(decoded, 128, atol=2)
