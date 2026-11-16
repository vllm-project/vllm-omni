# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI duplex audio must carry the model's advertised sample rate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import JoyAIDataPlane
from vllm_omni.sample_rate import MissingSampleRateError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_audio_without_a_rate_fails_loudly() -> None:
    plane = JoyAIDataPlane(lambda *args, **kwargs: "encoded")
    plane.begin_request("req")
    output = SimpleNamespace(
        request_id="req",
        stage_id=plane.audio_stage_id,
        finished=True,
        outputs=[],
        multimodal_output={"audio": np.zeros(8, dtype=np.float32)},
    )

    with pytest.raises(MissingSampleRateError, match="must emit 'sr'"):
        list(plane.project({"data_plane_outputs": [output]}, context=DuplexDataPlaneContext(modalities=("audio",))))


def test_advertised_rate_reaches_the_projection() -> None:
    plane = JoyAIDataPlane(lambda *args, **kwargs: "encoded")
    plane.begin_request("req")
    output = SimpleNamespace(
        request_id="req",
        stage_id=plane.audio_stage_id,
        finished=True,
        outputs=[],
        multimodal_output={"audio": np.zeros(24000, dtype=np.float32), "sr": 16000},
    )

    [event] = list(
        plane.project({"data_plane_outputs": [output]}, context=DuplexDataPlaneContext(modalities=("audio",)))
    )

    assert event["sample_rate_hz"] == 16000
    assert event["audio_duration_ms"] == 1500
