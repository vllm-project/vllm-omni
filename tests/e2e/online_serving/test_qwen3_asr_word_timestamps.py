# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-audio coverage for Qwen3-ASR plus the injected pooling aligner."""

import pytest
import requests
from vllm.assets.audio import AudioAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

_SERVER_PARAMS = OmniServerParams(
    model="Qwen/Qwen3-ASR-0.6B",
    stage_config_path=get_deploy_config_path("qwen3_asr.yaml"),
    server_args=[
        "--forced-aligner",
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "--enforce-eager",
        "--max-num-seqs",
        "2",
        "--kv-cache-memory-bytes",
        str(512 * 1024**2),
    ],
)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.parametrize("omni_server", [_SERVER_PARAMS], indirect=True, ids=["asr-aligner"]),
]


@pytest.fixture(scope="module")
def speech_audio():
    asset = AudioAsset("mary_had_lamb")
    audio, sample_rate = asset.audio_and_sample_rate
    return asset.get_local_path().read_bytes(), len(audio) / sample_rate


def _transcribe(server, audio, response_format, **fields):
    return requests.post(
        f"http://{server.host}:{server.port}/v1/audio/transcriptions",
        files={"file": ("speech.ogg", audio, "audio/ogg")},
        data={
            "model": server.model,
            "language": "en",
            "response_format": response_format,
            **fields,
        },
        timeout=120,
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_qwen3_asr_word_timestamps(omni_server, speech_audio):
    audio, duration = speech_audio
    response = _transcribe(omni_server, audio, "verbose_json", **{"timestamp_granularities[]": "word"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert "mary" in result["text"].lower(), result
    assert "lamb" in result["text"].lower(), result
    assert "<asr_text>" not in result["text"]
    assert result["duration"] == pytest.approx(duration, abs=0.05)
    assert len(result["words"]) >= 5, result
    previous_end = 0.0
    for word in result["words"]:
        assert word["word"].strip()
        assert previous_end <= word["start"] <= word["end"] <= duration + 0.05
        previous_end = word["end"]
    # A nonempty list of zero-length times does not establish that the audio
    # reached the aligner; require alignment across the spoken recording.
    assert previous_end > duration / 2, result


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_qwen3_asr_plain_transcription_with_aligner_enabled(omni_server, speech_audio):
    response = _transcribe(omni_server, speech_audio[0], "json")
    assert response.status_code == 200, response.text
    result = response.json()
    assert "mary" in result["text"].lower(), result
    assert "<asr_text>" not in result["text"]
    assert not result.get("words")
