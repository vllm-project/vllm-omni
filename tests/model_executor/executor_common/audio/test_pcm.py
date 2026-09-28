# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64

import numpy as np
import pytest

from vllm_omni.model_executor.common.audio.pcm import (
    PCM_F32LE_BYTES_PER_SAMPLE,
    decode_pcm_f32le_base64,
    pcm_f32le_sample_count,
    pcm_f32le_samples,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _b64(samples: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(samples, dtype="<f4").tobytes()).decode("ascii")


def test_decode_returns_the_raw_bytes_of_whole_finite_samples() -> None:
    samples = np.arange(4, dtype=np.float32)

    raw = decode_pcm_f32le_base64(_b64(samples))

    assert raw == samples.tobytes()
    assert pcm_f32le_sample_count(raw) == 4
    assert PCM_F32LE_BYTES_PER_SAMPLE == 4


def test_decode_accepts_an_empty_chunk() -> None:
    assert decode_pcm_f32le_base64("") == b""
    assert pcm_f32le_sample_count(b"") == 0


@pytest.mark.parametrize(
    ("encoded", "match"),
    [
        (None, "must be base64 pcm_f32le"),
        (b"bytes", "must be base64 pcm_f32le"),
        ("not*base64", "not valid base64"),
        (base64.b64encode(b"\x00\x00\x00").decode("ascii"), "divisible by four"),
        (_b64(np.array([0.0, np.nan], dtype=np.float32)), "must be finite"),
        (_b64(np.array([np.inf], dtype=np.float32)), "must be finite"),
    ],
)
def test_decode_rejects_malformed_audio_naming_the_model(encoded: object, match: str) -> None:
    with pytest.raises(ValueError, match=match) as excinfo:
        decode_pcm_f32le_base64(encoded, model="TestModel")

    assert str(excinfo.value).startswith("TestModel ")


def test_samples_are_a_writable_contiguous_float32_copy() -> None:
    raw = np.arange(3, dtype="<f4").tobytes()

    samples = pcm_f32le_samples(raw)

    assert samples.dtype == np.float32
    assert samples.flags.writeable and samples.flags.c_contiguous
    assert samples.tolist() == [0.0, 1.0, 2.0]
