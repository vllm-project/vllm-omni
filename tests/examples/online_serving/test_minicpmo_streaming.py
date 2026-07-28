import base64
import io
import runpy
import wave
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_append_wav_chunk_extracts_pcm_frames() -> None:
    example = Path(__file__).parents[3] / "examples" / "online_serving" / "minicpmo" / "streaming_chat_completion.py"
    append_wav_chunk = runpy.run_path(example)["append_wav_chunk"]
    encoded_buffer = io.BytesIO()
    with wave.open(encoded_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x02\x03\x04")

    pcm_parts: list[bytes] = []
    audio_format = append_wav_chunk(
        base64.b64encode(encoded_buffer.getvalue()).decode(),
        pcm_parts,
    )

    assert audio_format == (1, 2, 24000)
    assert pcm_parts == [b"\x01\x02\x03\x04"]
