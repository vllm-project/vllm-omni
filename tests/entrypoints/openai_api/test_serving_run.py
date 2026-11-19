# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import msgspec
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.config.endpoint_policy import shutdown_unsupported_routes
from vllm_omni.engine.messages import NextStageInputMessage, OutputMessage
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.protocol.run import RunRequest
from vllm_omni.entrypoints.openai.serving_run import (
    ServingRun,
    decode_output,
    decode_stage_input,
    encode_payload,
    run_endpoint_restrictions,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROMPT = {"prompt": "hi"}


def _next_stage_input(requests: list[EngineCoreRequest] | None = None) -> NextStageInputMessage:
    return NextStageInputMessage(
        request_id="req-run",
        source_stage_id=0,
        receiver_stage_id=1,
        requests=requests or [],
        submit_kwargs=None,
        stage_output=None,
        prompt=_PROMPT,
        sampling_params_list=[SamplingParams(), SamplingParams()],
    )


def _engine_client(response: NextStageInputMessage | OutputMessage | None = None) -> SimpleNamespace:
    """A two-stage LLM engine client whose run calls return ``response`` (by default, a stage 1 input)."""
    response = response or _next_stage_input()
    return SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="llm")] * 2,
        run_entry_stage=AsyncMock(return_value=response),
        run_downstream_stage=AsyncMock(return_value=response),
    )


def _run_client(engine_client: SimpleNamespace | None = None) -> TestClient:
    """A client for an app with the API router, serving /v1/run with ``engine_client``."""
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.run_serving = ServingRun(engine_client or _engine_client())
    return TestClient(app)


def test_stage_input_round_trips_llm_receiver():
    """Ensure an LLM receiver's requests, including omni-only fields, survive encoding."""
    request = build_engine_core_request_from_tokens("req-run", {"prompt_token_ids": [7]}, SamplingParams(max_tokens=9))
    request.additional_information = serialize_additional_information({"codes": torch.arange(2)})
    stage_input = _next_stage_input([request])
    assert decode_stage_input(encode_payload(stage_input), ("llm", "llm")) == stage_input


def test_stage_input_round_trips_sampling_params():
    """Ensure a stage's sampling params (including complex types) survive encoding."""
    stage_input = msgspec.structs.replace(
        _next_stage_input(),
        sampling_params_list=[
            SamplingParams(structured_outputs=StructuredOutputsParams(json={"type": "object"})),
            OmniDiffusionSamplingParams(
                lora_request=LoRARequest(lora_name="adapter", lora_int_id=1, lora_path="/adapter"),
            ),
        ],
    )
    decoded = decode_stage_input(encode_payload(stage_input), ("llm", "diffusion"))
    assert decoded.sampling_params_list == stage_input.sampling_params_list


def test_non_final_call_returns_only_next_stage_id_and_stage_input():
    """Ensure a call that yields returns only the next stage's id and its encoded input, with no output key."""
    next_stage_input = _next_stage_input()
    body = _run_client(_engine_client(next_stage_input)).post("/v1/run", json={"stage_input": _PROMPT}).json()
    assert set(body) == {"stage_id", "stage_input"}
    assert body["stage_id"] == next_stage_input.receiver_stage_id
    assert decode_stage_input(body["stage_input"], ("llm", "llm")) == next_stage_input


@pytest.mark.asyncio
async def test_final_call_returns_encoded_output():
    """Ensure the final stage's call returns its raw output, encoded with the audio on its completion output."""
    audio = torch.arange(4.0)
    completion = CompletionOutput(index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None)
    completion.multimodal_output = {"audio": audio}
    final_output = RequestOutput(
        request_id="run-final",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    serving = ServingRun(
        _engine_client(OutputMessage(request_id="run-final", stage_id=1, engine_outputs=final_output, finished=True))
    )
    request = RunRequest(stage_id=1, stage_input=encode_payload(_next_stage_input()))
    response = await serving.run(request, request_id="run-final")
    output = decode_output(response.output)
    assert type(output) is RequestOutput
    assert torch.equal(output.outputs[0].multimodal_output["audio"], audio)


@pytest.mark.asyncio
async def test_entry_stage_call_rejects_encoded_stage_input():
    """Ensure stage 0 rejects an encoded stage input instead of running it as a text prompt."""
    engine_client = _engine_client()
    request = RunRequest(stage_id=0, stage_input=encode_payload(_next_stage_input()))
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_entry_stage.assert_not_awaited()


@pytest.mark.asyncio
async def test_downstream_stage_call_rejects_stage_input_built_for_another_stage():
    """Ensure a call is rejected when its stage_input was built for a different stage than its stage_id."""
    engine_client = _engine_client()
    stage_input = _next_stage_input()
    request = RunRequest(stage_id=stage_input.receiver_stage_id + 1, stage_input=encode_payload(stage_input))
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_downstream_stage.assert_not_awaited()


@pytest.mark.asyncio
async def test_downstream_stage_call_rejects_sampling_params():
    """Ensure a downstream call can't set sampling params, since its stage_input already carries them."""
    engine_client = _engine_client()
    request = RunRequest(stage_id=1, stage_input=encode_payload(_next_stage_input()), sampling_params=[{}])
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_downstream_stage.assert_not_awaited()


def test_run_route_returns_400_for_malformed_stage_input():
    """Ensure a stage_input that isn't a valid payload is a client error, not a server error."""
    response = _run_client().post("/v1/run", json={"stage_id": 1, "stage_input": "not a payload"})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "stage_ids",
    [{"receiver_stage_id": 9999}, {"receiver_stage_id": -1}, {"receiver_stage_id": "1"}, {"source_stage_id": 1}],
)
def test_run_route_returns_400_for_stage_ids_outside_pipeline(stage_ids):
    """Ensure a stage_input whose source stage doesn't precede a receiver in the pipeline is a client error."""
    stage_input = msgspec.structs.replace(_next_stage_input(), **stage_ids)
    body = {"stage_id": stage_input.receiver_stage_id, "stage_input": encode_payload(stage_input)}
    response = _run_client().post("/v1/run", json=body)
    assert response.status_code == 400


def test_run_route_keeps_client_error_status():
    """Ensure a client error raised during a run call keeps its status code."""
    status_code = 422
    engine_client = _engine_client()
    engine_client.run_entry_stage.side_effect = OmniClientError("bad prompt", status_code=status_code)
    response = _run_client(engine_client).post("/v1/run", json={"stage_input": _PROMPT})
    assert response.status_code == status_code


@pytest.mark.parametrize("value", ["", "0"])
def test_run_route_rejects_calls_unless_enabled_by_env(monkeypatch: pytest.MonkeyPatch, value: str):
    """Ensure /v1/run returns 400 naming VLLM_OMNI_ENABLE_RUN_ENDPOINT unless that variable turns it on."""
    monkeypatch.setenv("VLLM_OMNI_ENABLE_RUN_ENDPOINT", value)
    client = _run_client()
    shutdown_unsupported_routes(client.app, run_endpoint_restrictions())

    response = client.post("/v1/run", json={"stage_input": _PROMPT})

    assert response.status_code == 400
    assert "VLLM_OMNI_ENABLE_RUN_ENDPOINT" in response.text


def test_run_route_serves_calls_when_enabled_by_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure /v1/run serves calls when VLLM_OMNI_ENABLE_RUN_ENDPOINT=1."""
    monkeypatch.setenv("VLLM_OMNI_ENABLE_RUN_ENDPOINT", "1")
    client = _run_client()
    shutdown_unsupported_routes(client.app, run_endpoint_restrictions())

    response = client.post("/v1/run", json={"stage_input": _PROMPT})

    assert response.status_code == 200
