# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 emits the rate the AudioVAE actually decodes at."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import _resolve_output_sample_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_output_rate_follows_vae_not_top_level_config() -> None:
    # The waveform is decoded by the AudioVAE at its own out_sample_rate
    # (it is even a decode conditioning input). The top-level config's
    # sample_rate describes the reference/input side and can disagree, so
    # the resolver must not consult it.
    native_tts = SimpleNamespace(audio_vae=SimpleNamespace(out_sample_rate=44100))

    assert _resolve_output_sample_rate(native_tts) == 44100
