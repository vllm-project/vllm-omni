# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-only contracts for the async native image request helper."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from tests.helpers.client import send_async_image_diffusion_request
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


@pytest.fixture
def sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(width=32, height=24, num_outputs_per_prompt=2)


@pytest.fixture
def image_output() -> OmniRequestOutput:
    return OmniRequestOutput.from_diffusion(
        request_id="image-request",
        images=[Image.new("RGB", (32, 24)), Image.new("RGB", (32, 24))],
    )


def _engine_with_outputs(
    mocker: MockerFixture,
    outputs: Sequence[OmniRequestOutput],
    *,
    error: RuntimeError | None = None,
) -> AsyncOmni:
    async def generate() -> AsyncIterator[OmniRequestOutput]:
        for output in outputs:
            yield output
        if error is not None:
            raise error

    engine = mocker.Mock(spec=["generate"])
    engine.generate.return_value = generate()
    return cast("AsyncOmni", engine)


async def test_async_image_request_consumes_stream_and_returns_final_output(
    mocker: MockerFixture,
    sampling_params: OmniDiffusionSamplingParams,
    image_output: OmniRequestOutput,
) -> None:
    consumed = []
    intermediate = OmniRequestOutput(request_id="image-request", finished=False)

    async def generate() -> AsyncIterator[OmniRequestOutput]:
        yield intermediate
        yield image_output
        consumed.append(True)

    engine = mocker.Mock(spec=["generate"])
    engine.generate.return_value = generate()
    result = await send_async_image_diffusion_request(
        engine,
        prompt="A red cube",
        sampling_params=sampling_params,
        request_id="image-request",
    )

    engine.generate.assert_called_once_with(
        prompt={"prompt": "A red cube"},
        request_id="image-request",
        sampling_params_list=[sampling_params],
        output_modalities=["image"],
    )
    assert result is image_output
    assert consumed == [True]


async def test_async_image_request_rejects_empty_stream(
    mocker: MockerFixture, sampling_params: OmniDiffusionSamplingParams
) -> None:
    with pytest.raises(AssertionError, match="returned no output"):
        await send_async_image_diffusion_request(
            _engine_with_outputs(mocker, []),
            prompt="A red cube",
            sampling_params=sampling_params,
            request_id="image-request",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finished", False, "did not finish"),
        ("request_id", "other-request", "Expected request_id="),
        ("final_output_type", "audio", "Expected image output"),
        ("error", "worker failed", "Image generation failed: worker failed"),
    ],
)
async def test_async_image_request_rejects_invalid_final_output(
    mocker: MockerFixture,
    sampling_params: OmniDiffusionSamplingParams,
    image_output: OmniRequestOutput,
    field: str,
    value: object,
    message: str,
) -> None:
    setattr(image_output, field, value)
    with pytest.raises(AssertionError, match=message):
        await send_async_image_diffusion_request(
            _engine_with_outputs(mocker, [image_output]),
            prompt="A red cube",
            sampling_params=sampling_params,
            request_id="image-request",
        )


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        ([], "No images in response"),
        ([(32, 24)], "Expected 2 images, got 1"),
        ([(32, 24), (31, 24)], "Expected width=32, got 31"),
        ([(32, 24), (32, 23)], "Expected height=24, got 23"),
    ],
)
async def test_async_image_request_checks_every_image(
    mocker: MockerFixture,
    sampling_params: OmniDiffusionSamplingParams,
    image_output: OmniRequestOutput,
    sizes: list[tuple[int, int]],
    message: str,
) -> None:
    image_output.images = [Image.new("RGB", size) for size in sizes]
    with pytest.raises(AssertionError, match=message):
        await send_async_image_diffusion_request(
            _engine_with_outputs(mocker, [image_output]),
            prompt="A red cube",
            sampling_params=sampling_params,
            request_id="image-request",
        )


async def test_async_image_request_propagates_stream_failure_after_output(
    mocker: MockerFixture,
    sampling_params: OmniDiffusionSamplingParams,
    image_output: OmniRequestOutput,
) -> None:
    error = RuntimeError("generation stream failed")
    with pytest.raises(RuntimeError, match="generation stream failed") as exc_info:
        await send_async_image_diffusion_request(
            _engine_with_outputs(mocker, [image_output], error=error),
            prompt="A red cube",
            sampling_params=sampling_params,
            request_id="image-request",
        )
    assert exc_info.value is error
