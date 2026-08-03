# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints.openai.api_server import _TextOnlyEngineClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.aborted = []
        self.vllm_config = "thinker-config"

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.calls.append((prompt, sampling_params, request_id, kwargs))
        yield "output"

    async def abort(self, request_id):
        self.aborted.append(request_id)


async def test_generate_pins_text_output_modalities():
    engine = FakeEngine()
    client = _TextOnlyEngineClient(engine)

    outputs = [out async for out in client.generate("prompt", None, "req-1")]

    assert outputs == ["output"]
    (_, _, _, kwargs) = engine.calls[0]
    assert kwargs["output_modalities"] == ["text"]


async def test_generate_preserves_caller_output_modalities():
    engine = FakeEngine()
    client = _TextOnlyEngineClient(engine)

    async for _ in client.generate("prompt", None, "req-1", output_modalities=["text", "audio"]):
        pass

    (_, _, _, kwargs) = engine.calls[0]
    assert kwargs["output_modalities"] == ["text", "audio"]


async def test_generate_forwards_other_kwargs():
    engine = FakeEngine()
    client = _TextOnlyEngineClient(engine)

    async for _ in client.generate("prompt", None, "req-1", priority=3, trace_headers={"a": "b"}):
        pass

    (_, _, _, kwargs) = engine.calls[0]
    assert kwargs["priority"] == 3
    assert kwargs["trace_headers"] == {"a": "b"}


async def test_non_generate_calls_delegate_to_wrapped_engine():
    engine = FakeEngine()
    client = _TextOnlyEngineClient(engine)

    await client.abort("req-2")

    assert engine.aborted == ["req-2"]
    assert client.vllm_config == "thinker-config"
