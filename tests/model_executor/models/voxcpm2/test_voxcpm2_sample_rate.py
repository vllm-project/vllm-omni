# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 emits the rate the AudioVAE actually decodes at."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm2.model_state import VoxCPM2AudioOutput
from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import _resolve_output_sample_rate
from vllm_omni.sample_rate import SAMPLE_RATE_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_output_rate_follows_vae_not_top_level_config() -> None:
    # The waveform is decoded by the AudioVAE at its own out_sample_rate
    # (it is even a decode conditioning input). The top-level config's
    # sample_rate describes the reference/input side and can disagree, so
    # the resolver must not consult it.
    native_tts = SimpleNamespace(audio_vae=SimpleNamespace(out_sample_rate=44100))

    assert _resolve_output_sample_rate(native_tts) == 44100


def test_packed_rows_carry_the_resolved_rate() -> None:
    sr = torch.tensor(16000, dtype=torch.int32)
    output = VoxCPM2AudioOutput.packed(
        wav=torch.zeros(4),
        lengths=torch.tensor([2, 2]),
        sample_rate=sr,
        batch_rows=(0, 1),
        num_reqs=2,
    )

    rows = output.get_output()

    assert len(rows) == 2
    for row in rows:
        assert row is not None
        assert row[SAMPLE_RATE_KEY] is sr
        assert int(row[SAMPLE_RATE_KEY]) == 16000
