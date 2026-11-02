# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU oracles prevent acceptance tests from passing empty/truncated audio."""

import base64
import json

import numpy as np
import pytest

from tests.e2e.zonos2.audio_checks import check_audio, check_lifecycle, data_uri, decode_sse, decode_wav, pcm16_wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "samples,rate,frames",
    [
        (np.zeros(0), 44100, None),
        (np.zeros(5120), 44100, None),
        (np.ones(5120), 24000, None),
        (np.full(5120, np.nan), 44100, None),
        (np.full(5120, np.inf), 44100, None),
        (np.ones(5120), 44100, 11),
    ],
)
def test_audio_oracle_rejects_invalid_outputs(samples, rate, frames):
    with pytest.raises(AssertionError):
        check_audio(samples, rate, frames=frames)


def test_local_reference_data_uri_and_wav_roundtrip():
    samples = np.sin(np.arange(10240) * 0.03).astype(np.float32) * 0.1
    wav = pcm16_wav(samples)
    assert base64.b64decode(data_uri(wav).split(",", 1)[1]) == wav
    decoded, sr = decode_wav(wav)
    np.testing.assert_allclose(decoded, samples, atol=6e-5, rtol=0)
    assert check_audio(decoded, sr, frames=20)["samples"] == 10240


def event(value):
    return "data: " + json.dumps(value) + "\n\n"


def test_sse_split_delta_and_terminal_oracle():
    body = event({"type": "speech.audio.delta", "audio": base64.b64encode(b"ab").decode()})
    body += event({"type": "speech.audio.delta", "audio": base64.b64encode(b"cd").decode()})
    body += event({"type": "speech.audio.done"})
    assert decode_sse(body.encode()) == (b"abcd", 2)
    with pytest.raises(AssertionError, match="terminal"):
        decode_sse(body.replace(event({"type": "speech.audio.done"}), "").encode())


@pytest.mark.parametrize("missing", ["decode", "talker_cleanup", "dac_cleanup"])
def test_lifecycle_oracle_requires_exact_samples_and_both_cleanup_callbacks(missing):
    rows: list[dict] = [
        {"kind": "finish", "request": "r-123", "eos_frame": 10, "frames": 20, "countdown": 0, "reached_cap": False},
        {
            "kind": "decode",
            "request": "r-123",
            "samples": 5120,
            "finite": True,
            "dtype": "torch.float32",
            "codec_loaded": True,
        },
        {"kind": "talker_cleanup", "ids": ["r-123"]},
        {"kind": "dac_cleanup", "ids": ["r-123"]},
    ]
    assert check_lifecycle(rows, "r")["decoded_samples"] == 5120
    with pytest.raises(AssertionError):
        check_lifecycle([row for row in rows if row["kind"] != missing], "r")
