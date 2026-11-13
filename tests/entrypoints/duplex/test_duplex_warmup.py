# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The startup warmup sends the model's own silence unit, not a fixed 16 kHz one."""

from __future__ import annotations

import base64

import pytest

from vllm_omni.entrypoints.duplex.warmup import warmup_silence_unit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Plugin:
    silence_continuation_samples = 1920

    def silence_unit_payload(self) -> dict[str, object]:
        return {
            "type": "audio",
            "audio": base64.b64encode(bytes(1920 * 4)).decode("ascii"),
            "format": "pcm_f32le",
            "sample_rate_hz": 24000,
        }


class _LegacyPlugin:
    silence_continuation_samples = 8000


def test_the_plugin_unit_is_used_verbatim() -> None:
    unit = warmup_silence_unit(_Plugin())

    assert unit["format"] == "pcm_f32le"
    assert unit["sample_rate_hz"] == 24000
    assert len(base64.b64decode(str(unit["audio"]))) == 1920 * 4


def test_a_plugin_without_the_hook_falls_back_to_16k_zeros_of_its_sample_count() -> None:
    unit = warmup_silence_unit(_LegacyPlugin())

    assert unit["sample_rate_hz"] == 16000
    assert unit["format"] == "pcm_f32le"
    assert len(base64.b64decode(str(unit["audio"]))) == 8000 * 4


def test_no_plugin_means_the_minicpm_sized_default() -> None:
    unit = warmup_silence_unit(None)

    assert unit["sample_rate_hz"] == 16000
    assert len(base64.b64decode(str(unit["audio"]))) == 16000 * 4
