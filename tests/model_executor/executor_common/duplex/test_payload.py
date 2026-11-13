# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64

import numpy as np
import pytest

from vllm_omni.model_executor.common.duplex.payload import (
    decode_pcm_f32le_payload,
    payload_audio,
    payload_sample_count,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: int, *, sample_rate_hz: int = 24000, fmt: str = "pcm_f32le", key: str = "audio") -> dict:
    audio = base64.b64encode(np.zeros(samples, dtype="<f4").tobytes()).decode("ascii")
    return {"type": "audio", "format": fmt, "sample_rate_hz": sample_rate_hz, key: audio}


def test_decode_returns_the_pcm_of_a_matching_payload() -> None:
    raw = decode_pcm_f32le_payload(_payload(1920), sample_rate_hz=24000, exact_samples=1920)

    assert len(raw) == 1920 * 4


def test_decode_accepts_the_older_data_key() -> None:
    assert len(decode_pcm_f32le_payload(_payload(8, key="data"), sample_rate_hz=24000)) == 32
    assert payload_audio({"data": "x"}) == "x"
    assert payload_audio("not a mapping") is None


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("frame", "must be a mapping"),
        (_payload(1920, fmt="pcm16"), "format must be pcm_f32le"),
        (_payload(1920, sample_rate_hz=16000), "sample_rate_hz must be 24000"),
        (_payload(1919), "exactly 1920 samples"),
        ({"format": "pcm_f32le", "sample_rate_hz": 24000, "audio": 12}, "must be base64"),
    ],
)
def test_decode_rejects_payloads_that_do_not_match_the_model_unit(payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match) as excinfo:
        decode_pcm_f32le_payload(payload, sample_rate_hz=24000, exact_samples=1920, model="TestModel")

    assert str(excinfo.value).startswith("TestModel ")


def test_decode_without_exact_samples_accepts_any_whole_number_of_samples() -> None:
    assert len(decode_pcm_f32le_payload(_payload(7), sample_rate_hz=24000)) == 28


def test_payload_sample_count_is_lenient() -> None:
    assert payload_sample_count(_payload(16000, sample_rate_hz=16000)) == 16000
    assert payload_sample_count(_payload(16000, fmt="pcm16")) is None
    assert payload_sample_count({"format": "pcm_f32le", "audio": "not*base64"}) is None
    assert payload_sample_count(None) is None
