# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.openai.audio import convert_input_audio_with_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pcm16_input_conversion_is_pure_and_reports_target_rate():
    pcm16 = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    encoded = base64.b64encode(pcm16).decode("ascii")

    converted, fmt, sample_rate = convert_input_audio_with_rate(
        encoded,
        "pcm16",
        sample_rate_hz=16_000,
    )

    assert encoded == base64.b64encode(pcm16).decode("ascii")
    assert fmt == "pcm_f32le"
    assert sample_rate == 16_000
    samples = np.frombuffer(base64.b64decode(converted), dtype="<f4")
    assert np.allclose(samples, [-1.0, 0.0, 32767 / 32768])


@pytest.mark.parametrize("sample_rate_hz", [8_000, 192_000])
def test_pcm16_input_conversion_accepts_supported_sample_rate_boundaries(sample_rate_hz: int):
    encoded = base64.b64encode(np.zeros(8, dtype="<i2").tobytes()).decode("ascii")

    _, fmt, converted_rate = convert_input_audio_with_rate(
        encoded,
        "pcm16",
        sample_rate_hz=sample_rate_hz,
    )

    assert fmt == "pcm_f32le"
    assert converted_rate == 16_000


@pytest.mark.parametrize("sample_rate_hz", [7_999, 192_001, 10**1000])
def test_pcm16_input_conversion_rejects_sample_rates_outside_supported_range(sample_rate_hz: int):
    encoded = base64.b64encode(np.zeros(8, dtype="<i2").tobytes()).decode("ascii")

    with pytest.raises(ValueError, match="sample_rate_hz"):
        convert_input_audio_with_rate(
            encoded,
            "pcm16",
            sample_rate_hz=sample_rate_hz,
        )
