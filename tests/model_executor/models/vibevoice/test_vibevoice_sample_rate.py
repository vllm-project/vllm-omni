# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice stateful inference emits the canonical sample-rate payload shape."""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.models.vibevoice.stateful as stateful_module
from vllm_omni.model_executor.models.vibevoice.stateful import (
    VibeVoiceStatefulInference,
    _Row,
)
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050

AUDIO_BOS = 1
AUDIO_TOKEN = 2
TEXT_TOKEN = 3


def _bare_inference() -> VibeVoiceStatefulInference:
    inference = object.__new__(VibeVoiceStatefulInference)
    inference._rows = [
        _Row(request_id="r1", sample_valid=True, input_embedding=torch.zeros(1, 1)),
        _Row(request_id="r2", sample_valid=True, input_embedding=torch.zeros(1, 1)),
    ]
    inference._sampled = [[AUDIO_TOKEN], [TEXT_TOKEN]]
    inference._hidden = torch.zeros(2, 1)
    inference._audio_bos_token_id = AUDIO_BOS
    inference._audio_token_id = AUDIO_TOKEN
    inference._negative_kv = SimpleNamespace(reset_segment=lambda request_id: None)
    return inference


def test_finish_step_payload_rate_follows_module_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No config field exposes the acoustic tokenizer's rate, so the module
    # constant is the single source; pin it to an unusual value and the
    # emission must follow.
    monkeypatch.setattr(stateful_module, "SAMPLE_RATE", UNUSUAL_RATE)
    monkeypatch.setattr(
        VibeVoiceStatefulInference,
        "_run_audio_tokens",
        lambda self, request_ids, negative_inputs, positive: [torch.zeros(3)],
    )
    inference = _bare_inference()

    payload = inference.finish_step(["r1", "r2"])

    # Only the audio-token row produces a chunk.
    assert len(payload["model_outputs"]) == 1
    assert payload["model_outputs"][0].shape == (3,)
    srs = payload[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(payload) == UNUSUAL_RATE
