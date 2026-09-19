# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import threading
from io import BytesIO

import pytest
from PIL import Image
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def make_handler(mocker: MockerFixture):
    def build(images: list[Image.Image]):
        output = OmniRequestOutput.from_diffusion(
            request_id="image-result",
            images=images,
            stage_durations={"stage_0_gen_ms": 12.5},
            peak_memory_mb=123.0,
        )

        async def generate(**kwargs):
            yield output

        engine = mocker.Mock(spec=AsyncOmni)
        engine.stage_configs = []
        engine.default_sampling_params_list = []
        engine.generate = generate
        handler = OmniOpenAIServingChat.for_diffusion(engine, "test-image")
        handler._diffusion_extra_output_params = frozenset()
        request = ChatCompletionRequest(
            model="test-image", messages=[{"role": "user", "content": "a small image"}], stream=False
        )

        async def complete():
            return await handler._create_diffusion_chat_completion(request, {}, {})

        return complete, output

    return build


@pytest.mark.parametrize("nested", [False, True])
def test_image_bytes_order_and_metrics(make_handler, nested):
    images = [Image.new("RGB", (3, 2), "red"), Image.new("RGBA", (2, 3), (1, 2, 3, 4))]
    expected = []
    for img in images:
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            expected.append(buffer.getvalue())
    complete, output = make_handler(images)
    if nested:
        # The handler supports layered lists as well as the declared flat output.
        setattr(output, "images", [[images[0]], images[1]])
    response = asyncio.run(complete())
    assert not isinstance(response, ErrorResponse)
    content = response.choices[0].message.content
    assert isinstance(content, list) and len(content) == len(images)
    for item, png in zip(content, expected):
        assert item["type"] == "image_url"
        prefix, encoded = item["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded) == png
        assert item["stage_durations"] == output.stage_durations
        assert item["peak_memory_mb"] == output.peak_memory_mb


def test_image_encode_error_returns_existing_error(make_handler, monkeypatch):
    img = Image.new("RGB", (2, 2))

    def fail(*args, **kwargs):
        raise OSError("PNG encoder failed")

    monkeypatch.setattr(img, "save", fail)
    complete, _ = make_handler([img])
    response = asyncio.run(complete())
    assert isinstance(response, ErrorResponse)
    assert response.error.code == 500
    assert "PNG encoder failed" in response.error.message


def test_encoding_leaves_event_loop_responsive(make_handler, monkeypatch):
    img = Image.new("RGB", (2, 2))
    save = img.save
    release = threading.Event()

    async def exercise():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def blocked_save(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(3), "event loop could not release the PNG encoder"
            return save(*args, **kwargs)

        monkeypatch.setattr(img, "save", blocked_save)
        complete, _ = make_handler([img])
        task = asyncio.create_task(complete())
        try:
            await asyncio.wait_for(started.wait(), 5)
            # This coroutine must run while native encoding is still in progress.
            assert not task.done()
        finally:
            release.set()
        response = await asyncio.wait_for(task, 5)
        assert not isinstance(response, ErrorResponse)

    asyncio.run(exercise())
