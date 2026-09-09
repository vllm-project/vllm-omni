# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.indextts2 import (
    STOP_MEL_TOKEN,
    talker2s2mel_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_no_latent_full_payload_uses_sampled_output_ids_and_strips_stop():
    request = SimpleNamespace(
        request_id="req-1",
        output_token_ids=[17, 23, STOP_MEL_TOKEN],
        sampling_params=SimpleNamespace(seed=42),
    )
    payload = {
        "meta": {
            "use_gpt_latent": False,
            "S_ref": torch.ones(1, 4),
            "ref_mel": torch.ones(1, 80, 3),
            "style": torch.ones(1, 192),
        }
    }

    result = talker2s2mel_full_payload(None, payload, request)

    assert result is not None
    assert result["mel_codes"].tolist() == [[17, 23]]
    assert result["code_lens"].tolist() == [2]
    assert result["use_gpt_latent"] is False
    assert result["seed"] == 42


def test_no_latent_full_payload_rejects_legacy_payload_codes():
    request = SimpleNamespace(
        request_id="legacy",
        output_token_ids=[],
        sampling_params=SimpleNamespace(seed=None),
    )
    payload = {
        "codes": {"mel": torch.tensor([31, 37, STOP_MEL_TOKEN])},
        "meta": {"use_gpt_latent": False},
    }

    with pytest.raises(ValueError, match="no completed output_token_ids"):
        talker2s2mel_full_payload(None, payload, request)
