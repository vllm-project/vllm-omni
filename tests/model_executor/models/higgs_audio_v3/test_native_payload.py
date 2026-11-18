# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
    talker2code2wav,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
)
from vllm_omni.outputs.mm_outputs import MultimodalPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("rows", [0, 4, 8, 9, 30])
@pytest.mark.parametrize("offset", [0, -16, 1016])
def test_native_full_payload_preserves_valid_dedelayed_prefix(rows, offset, wrapped):
    audio = torch.arange(rows * 8).reshape(rows, 8) + offset
    payload = {"codes": {"audio": audio}}
    if wrapped:
        payload = MultimodalPayload(metadata=payload)
    out = SimpleNamespace(finished=True, outputs=[SimpleNamespace(multimodal_output=payload)])
    legacy = talker2code2wav([out])[0]["prompt_token_ids"]
    native = talker2code2wav_full_payload(None, {"codes.audio": audio}, None)
    assert native["codes"]["audio"].tolist() == legacy
    # Book q reads delayed row q + frame. Keep every complete valid frame
    # before the first special/pad; zero-filling invalid codes or blindly
    # trimming the last frame would synthesize audio or discard real audio.
    valid_frames = []
    for frame in range(max(rows - 7, 0)):
        values = [(q + frame) * 8 + q + offset for q in range(8)]
        if any(value < 0 or value >= 1024 for value in values):
            break
        valid_frames.append(values)
    expected = [values[q] for q in range(8) for values in valid_frames]
    assert legacy == expected
    assert all(0 <= value < 1024 for value in legacy)
    assert native["meta"]["finished"]
    assert talker2code2wav_token_only([out])[0]["prompt_token_ids"] == legacy


def test_native_control_slot_without_audio_payload():
    out = SimpleNamespace(finished=True, outputs=[SimpleNamespace(multimodal_output=None)])
    assert talker2code2wav_token_only([out])[0]["prompt_token_ids"] == [0]
