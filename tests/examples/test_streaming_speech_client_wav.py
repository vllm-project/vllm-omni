# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import struct
import wave
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]

_CLIENT_PATH = (
    Path(__file__).resolve().parents[2] / "examples/online_serving/text_to_speech/qwen3_tts/streaming_speech_client.py"
)


def _load_client():
    spec = importlib.util.spec_from_file_location("streaming_speech_client", _CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _streaming_wav_header(sample_rate: int = 24000, channels: int = 1) -> bytes:
    bits_per_sample = 16
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,
    )


def test_save_streaming_wav_finalizes_placeholder_sizes(tmp_path: Path):
    client = _load_client()
    pcm = b"\x01\x00" * 480

    filename = client.save_audio_file(
        str(tmp_path), "utterance_000_sentence_000", "wav", [_streaming_wav_header(), pcm]
    )
    saved = Path(filename).read_bytes()

    assert filename.endswith(".wav")
    assert saved[:4] == b"RIFF"
    assert struct.unpack_from("<I", saved, 4)[0] == len(saved) - 8
    assert struct.unpack_from("<I", saved, 40)[0] == len(pcm)
    with wave.open(filename, "rb") as wav_file:
        assert wav_file.getnframes() == 480
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.readframes(480) == pcm
