# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit decisions validate input, bypass disabled calls and own cancellation."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import AsyncMock

import anyio
import pytest
import torch
from pydantic import ValidationError
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.drvexo_decision import GeneralRequest, general_decision, input_row

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Tokenizer:
    encode: Callable


@dataclass
class _ServingChat:
    renderer: object
    engine_client: object


@dataclass
class _Request:
    is_disconnected: AsyncMock
    app: object


@dataclass
class _Engine:
    generate: Callable
    abort: AsyncMock
    default_sampling_params_list: list


@dataclass
class _Renderer:
    get_tokenizer: Callable


@dataclass
class _Application:
    state: object


@dataclass
class _DecisionOutput:
    error: object
    multimodal_output: dict


@dataclass
class _ApplicationState:
    openai_serving_chat: object
    drvexo_decision_limiter: object


def body(**kwargs):
    values = dict(
        task="choice",
        instruction="Match the state.",
        state={"color": "blue"},
        candidates=[{"id": "a", "text": "blue"}, {"id": "b", "text": "red"}],
    )
    return GeneralRequest(**(values | kwargs))


@pytest.mark.asyncio
async def test_disabled_does_not_access_backbone():
    result = await general_decision(body(general_decision_head_enabled=False), object())
    assert result == {"enabled": False, "probabilities": None, "executable": False}


def test_duplicate_ids_and_nonboolean_flag_rejected():
    with pytest.raises(ValidationError):
        body(general_decision_head_enabled="true")
    with pytest.raises(ValueError, match="unique"):
        input_row(body(candidates=[{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]))


def request(engine):
    tokenizer = _Tokenizer(encode=lambda text, **kw: list(text.encode()))
    chat = _ServingChat(renderer=_Renderer(get_tokenizer=lambda: tokenizer), engine_client=engine)
    return _Request(
        is_disconnected=AsyncMock(return_value=False),
        app=_Application(
            state=_ApplicationState(openai_serving_chat=chat, drvexo_decision_limiter=anyio.CapacityLimiter(1))
        ),
    )


@pytest.mark.asyncio
async def test_scores_are_request_local_and_non_executable():
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        yield _DecisionOutput(
            error=None,
            multimodal_output={
                "drvexo_general_scores": [torch.tensor([float(len(calls))])],
                "drvexo_general_status": [torch.tensor([1])],
            },
        )

    engine = _Engine(generate=generate, abort=AsyncMock(), default_sampling_params_list=[SamplingParams(max_tokens=10)])
    result = await general_decision(body(), request(engine))
    assert result["selected_id"] == "b"
    assert result["scores"] == {"a": 1.0, "b": 2.0}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["executable"] is False
    assert calls[0]["request_id"] != calls[1]["request_id"]
    assert calls[0]["sampling_params_list"][0].max_tokens == 1
    engine.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_aborts_only_its_owned_candidate():
    entered, closed = asyncio.Event(), asyncio.Event()
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        entered.set()
        try:
            await asyncio.Event().wait()
            yield None
        finally:
            closed.set()

    engine = _Engine(generate=generate, abort=AsyncMock(), default_sampling_params_list=[SamplingParams(max_tokens=10)])
    task = asyncio.create_task(general_decision(body(), request(engine)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    engine.abort.assert_awaited_once_with(calls[0]["request_id"])


@pytest.mark.asyncio
async def test_busy_decision_rejected_without_engine_work():
    from fastapi import HTTPException

    generate = AsyncMock()
    engine = _Engine(generate=generate, abort=AsyncMock(), default_sampling_params_list=[])
    raw = request(engine)
    limiter = raw.app.state.drvexo_decision_limiter
    limiter.acquire_on_behalf_of_nowait("other-request")
    with pytest.raises(HTTPException) as error:
        await general_decision(body(), raw)
    assert error.value.status_code == 429
    generate.assert_not_called()
    limiter.release_on_behalf_of("other-request")


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_candidate", [1, 2])
async def test_scoring_timeout_aborts_owned_candidate_and_releases_admission(mocker, blocked_candidate):
    from fastapi import HTTPException

    from vllm_omni.entrypoints.openai import drvexo_decision as module

    closed, calls = [], []

    async def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) < blocked_candidate:
            yield _DecisionOutput(
                error=None,
                multimodal_output={
                    "drvexo_general_scores": [torch.tensor([1.0])],
                    "drvexo_general_status": [torch.tensor([1])],
                },
            )
            return
        try:
            await anyio.sleep_forever()
            yield None
        finally:
            closed.append(True)

    engine = _Engine(generate=generate, abort=AsyncMock(), default_sampling_params_list=[SamplingParams(max_tokens=1)])
    raw = request(engine)
    mocker.patch.object(module, "DECISION_TIMEOUT_SECONDS", 0.02)
    with pytest.raises(HTTPException) as error:
        await general_decision(body(), raw)
    assert error.value.status_code == 504
    assert closed == [True]
    assert len(calls) == blocked_candidate
    engine.abort.assert_awaited_once_with(calls[-1]["request_id"])
    assert raw.app.state.drvexo_decision_limiter.borrowed_tokens == 0
