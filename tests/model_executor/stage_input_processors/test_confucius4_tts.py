# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    MAX_GENERATED_SEMANTIC_TOKENS,
)
from vllm_omni.model_executor.stage_input_processors import confucius4_tts as processor_module
from vllm_omni.model_executor.stage_input_processors.confucius4_tts import (
    talker2decoder_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _RequestStub:
    request_id: str
    sampling_params: SamplingParams | None
    output_token_ids: list[int] | None = None


def _meta() -> dict[str, torch.Tensor]:
    return {
        "text_token_ids": torch.tensor([5, 6, 7]),
        "condition_vector": torch.ones(1, 4, 8),
        "ref_mel": torch.ones(1, 3, 80),
        "style": torch.ones(1, 192),
    }


def _payload() -> dict[str, dict[str, torch.Tensor]]:
    return {"meta": _meta()}


def test_full_payload_strips_eos_and_propagates_seed() -> None:
    request = _RequestStub(
        request_id="request",
        sampling_params=SamplingParams(seed=7),
        output_token_ids=[10, 11, 8193],
    )

    result = talker2decoder_full_payload(None, _payload(), request)

    assert result["semantic_codes"].tolist() == [10, 11]
    assert result["text_token_ids"].tolist() == [5, 6, 7]
    assert result["condition_vector"].shape == (1, 4, 8)
    assert result["seed"] == 7
    assert "latent" not in result
    assert all(value.device.type == "cpu" for key, value in result.items() if key != "seed")


def test_full_payload_matches_upstream_cleanup_of_sampled_ids() -> None:
    # Upstream drops BOS wherever it was sampled, cuts at EOS and clamps ids.
    request = _RequestStub(
        request_id="request",
        sampling_params=None,
        output_token_ids=[10, 8192, 9000, 11, 8193, 12],
    )

    result = talker2decoder_full_payload(None, _payload(), request)

    assert result["semantic_codes"].tolist() == [10, 8191, 11]


def test_full_payload_keeps_all_ids_when_request_stopped_on_length() -> None:
    request = _RequestStub("request", None, output_token_ids=[3, 4, 5])
    result = talker2decoder_full_payload(None, _payload(), request)
    assert result["semantic_codes"].tolist() == [3, 4, 5]


@pytest.mark.parametrize(
    ("output_token_ids", "warns"),
    [
        # Whole public budget used without EOS: the text did not fit.
        ([3] * MAX_GENERATED_SEMANTIC_TOKENS, True),
        # Stopped on EOS exactly at the budget: the utterance is complete.
        ([3] * (MAX_GENERATED_SEMANTIC_TOKENS - 1) + [8193], False),
        # A smaller caller-chosen max_new_tokens is an intentional cut.
        ([3, 4, 5], False),
    ],
)
def test_full_payload_warns_only_when_the_semantic_budget_truncates(
    monkeypatch: pytest.MonkeyPatch,
    output_token_ids: list[int],
    warns: bool,
) -> None:
    warning = Mock()
    monkeypatch.setattr(processor_module.logger, "warning", warning)
    request = _RequestStub("request", None, output_token_ids=output_token_ids)

    talker2decoder_full_payload(None, _payload(), request)

    assert warning.called is warns
    if warns:
        assert "truncated" in warning.call_args.args[0]


def test_full_payload_accepts_flat_dotted_keys() -> None:
    request = _RequestStub("request", None, output_token_ids=[3])
    flat = {f"meta.{key}": value for key, value in _meta().items()}
    assert talker2decoder_full_payload(None, flat, request)["semantic_codes"].tolist() == [3]


@pytest.mark.parametrize("token_ids", [None, [], [8193], [8192, 8193]])
def test_full_payload_rejects_requests_without_semantic_codes(token_ids) -> None:
    request = _RequestStub("request", None, output_token_ids=token_ids)
    with pytest.raises(ValueError, match="output_token_ids|no semantic codes"):
        talker2decoder_full_payload(None, _payload(), request)


@pytest.mark.parametrize("missing", ["text_token_ids", "condition_vector", "ref_mel", "style"])
def test_full_payload_requires_every_conditioning_tensor(missing: str) -> None:
    payload = _payload()
    del payload["meta"][missing]
    request = _RequestStub("request", None, output_token_ids=[3, 4])
    with pytest.raises(ValueError, match=f"requires non-empty {missing}"):
        talker2decoder_full_payload(None, payload, request)


def test_connector_accumulation_materializes_decoder_payload() -> None:
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )

    transport = _OmniConnectorPayloadTransportMixin()
    transport._pending_full_payload_send = {}
    transport._full_payload_replace_keys_cached = frozenset()
    request = _RequestStub(
        request_id="request",
        sampling_params=SamplingParams(seed=9),
        output_token_ids=[10, 11, 8193],
    )
    meta = _meta()

    # The prefill step carries the conditioning tensors; decode steps only the counter.
    chunks = [
        {"meta.semantic_code_count": 0, **{f"meta.{key}": value for key, value in meta.items()}},
        {"meta.semantic_code_count": 1},
        {"meta.semantic_code_count": 2},
    ]
    for chunk in chunks:
        transport.accumulate_full_payload_output("request", chunk, request)

    full, materialized_request = transport._materialize_full_payload_entry(
        transport._pending_full_payload_send["request"]
    )
    result = talker2decoder_full_payload(transport, full, materialized_request)

    assert result["semantic_codes"].tolist() == [10, 11]
    for key, value in meta.items():
        assert torch.equal(result[key], value)
    assert result["seed"] == 9


def test_request_finished_on_prefill_still_materializes_payload() -> None:
    # max_tokens=1: the only model step is the prefill, which samples one code.
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )

    transport = _OmniConnectorPayloadTransportMixin()
    transport._pending_full_payload_send = {}
    transport._full_payload_replace_keys_cached = frozenset()
    request = _RequestStub("request", None, output_token_ids=[10])
    meta = _meta()
    transport.accumulate_full_payload_output(
        "request",
        {"meta.semantic_code_count": 0, **{f"meta.{key}": value for key, value in meta.items()}},
        request,
    )

    full, materialized_request = transport._materialize_full_payload_entry(
        transport._pending_full_payload_send["request"]
    )
    result = talker2decoder_full_payload(transport, full, materialized_request)

    assert result["semantic_codes"].tolist() == [10]
    for key, value in meta.items():
        assert torch.equal(result[key], value)
