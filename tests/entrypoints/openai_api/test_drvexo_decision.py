# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit decisions validate input, bypass disabled calls and own cancellation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from pydantic import ValidationError
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.drvexo_decision import GeneralRequest, general_decision, input_row

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))
    chat = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer), engine_client=engine)
    return SimpleNamespace(
        is_disconnected=AsyncMock(return_value=False),
        app=SimpleNamespace(state=SimpleNamespace(openai_serving_chat=chat)),
    )


@pytest.mark.asyncio
async def test_scores_are_request_local_and_non_executable():
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(
            error=None,
            multimodal_output={
                "drvexo_general_scores": [torch.tensor([float(len(calls))])],
                "drvexo_general_status": [torch.tensor([1])],
            },
        )

    engine = SimpleNamespace(
        generate=generate, abort=AsyncMock(), default_sampling_params_list=[SamplingParams(max_tokens=10)]
    )
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

    engine = SimpleNamespace(
        generate=generate, abort=AsyncMock(), default_sampling_params_list=[SamplingParams(max_tokens=10)]
    )
    task = asyncio.create_task(general_decision(body(), request(engine)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    engine.abort.assert_awaited_once_with(calls[0]["request_id"])
