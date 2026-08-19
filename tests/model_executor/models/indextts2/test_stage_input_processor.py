# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.model_executor.stage_input_processors.indextts2 import (
    STOP_MEL_TOKEN,
    talker2s2mel_full_payload,
)
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
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


def test_no_latent_full_payload_keeps_first_tick_conditioning_after_later_ticks():
    request = SimpleNamespace(
        request_id="req-multi-token",
        output_token_ids=[17, 23, STOP_MEL_TOKEN],
        sampling_params=SimpleNamespace(seed=None),
    )
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner._pending_full_payload_send = {}
    runner._custom_process_func = talker2s2mel_full_payload
    first_tick = {
        "meta": {
            "use_gpt_latent": False,
            "S_ref": torch.ones(1, 4),
            "ref_mel": torch.ones(1, 80, 3),
            "style": torch.ones(1, 192),
            "duration_factor": 0.5,
        }
    }
    later_tick = {"meta": {"use_gpt_latent": False}}

    runner.accumulate_full_payload_output(
        request.request_id,
        flatten_payload(first_tick),
        request,
    )
    runner.accumulate_full_payload_output(
        request.request_id,
        flatten_payload(later_tick),
        request,
    )
    accumulated, accumulated_request = runner._materialize_full_payload_entry(
        runner._pending_full_payload_send[request.request_id]
    )
    result = talker2s2mel_full_payload(
        None,
        accumulated,
        accumulated_request,
    )

    assert result is not None
    assert result["mel_codes"].tolist() == [[17, 23]]
    assert result["duration_factor"] == 0.5
    torch.testing.assert_close(result["S_ref"], first_tick["meta"]["S_ref"])
    torch.testing.assert_close(result["ref_mel"], first_tick["meta"]["ref_mel"])
    torch.testing.assert_close(result["style"], first_tick["meta"]["style"])


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
