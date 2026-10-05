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
def test_native_full_payload_preserves_legacy_dedelay_and_tail(rows, offset, wrapped):
    audio = torch.arange(rows * 8).reshape(rows, 8) + offset
    payload = {"codes": {"audio": audio}}
    if wrapped:
        payload = MultimodalPayload(metadata=payload)
    out = SimpleNamespace(finished=True, outputs=[SimpleNamespace(multimodal_output=payload)])
    legacy = talker2code2wav([out])[0]["prompt_token_ids"]
    native = talker2code2wav_full_payload(None, {"codes.audio": audio}, None)
    assert native["codes"]["audio"].tolist() == legacy
    # Independent expected layout: book q reads delayed rows q + frame.
    frames = max(rows - 7, 0) if rows >= 8 else 0
    kept = frames - 1 if frames >= 2 else frames
    expected = []
    for q in range(8):
        for frame in range(kept):
            value = (q + frame) * 8 + q + offset
            expected.append(value if 0 <= value < 1024 else 0)
    assert legacy == expected
    assert native["meta"]["finished"]
    assert talker2code2wav_token_only([out])[0]["prompt_token_ids"] == legacy


def test_native_control_slot_without_audio_payload():
    out = SimpleNamespace(finished=True, outputs=[SimpleNamespace(multimodal_output=None)])
    assert talker2code2wav_token_only([out])[0]["prompt_token_ids"] == [0]
