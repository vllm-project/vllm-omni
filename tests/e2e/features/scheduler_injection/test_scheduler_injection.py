# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for diffusion scheduler injection (``OmniDiffusionConfig.scheduler``).

Validates that setting ``scheduler`` to a dotted scheduler class path injects
that scheduler (here, the verl-omni SDE sampler) into the stock pipeline —
no ``custom_pipeline_args`` needed — and that omitting it keeps the
pipeline's default scheduler (the default construction path must stay
bit-identical).
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack

import numpy as np
import pytest

from tests.e2e.features.helpers.custom_scheduler import MARKER_ENV_VAR
from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

# verl-omni-style SDE sampler (log-probs in step). Injection must use this
# class on the stock Qwen-Image pipeline — no custom_pipeline_args.
INJECTED_SCHEDULER = "tests.e2e.features.helpers.custom_pipeline.FlowMatchSDEDiscreteSchedulerForTest"

# Same tiny random model as the custom_pipeline e2e tests.
MODEL = "tiny-random/Qwen-Image"

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _sampling_params(*, seed: int = 42) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=0.0,
        height=256,
        width=256,
        seed=seed,
    )


async def _generate_once(
    engine: AsyncOmni,
    *,
    request_id: str,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniRequestOutput:
    last_output = None
    async for output in engine.generate(
        prompt="a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline",
        request_id=request_id,
        sampling_params_list=[sampling_params],
        output_modalities=["image"],
    ):
        last_output = output

    assert last_output is not None
    assert isinstance(last_output, OmniRequestOutput)
    return last_output


def _assert_valid_image_output(output: OmniRequestOutput) -> None:
    assert output.final_output_type == "image"
    assert output.images, "Expected at least one generated image"

    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    assert arr.ndim == 3 and arr.shape[2] == 3, f"Expected HWC RGB image, got shape={arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert 0.0 <= float(arr[0, 0, 0]) <= 1.0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_scheduler_injection_uses_injected_scheduler(tmp_path, monkeypatch):
    """``diffusion_scheduler=<dotted path>`` must construct and step the injected SDE class."""
    marker = tmp_path / "scheduler_events.txt"
    monkeypatch.setenv(MARKER_ENV_VAR, str(marker))

    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            diffusion_scheduler=INJECTED_SCHEDULER,
            enforce_eager=True,
            max_num_seqs=1,
        )
        after.callback(engine.shutdown)

        output = await _generate_once(
            engine,
            request_id=f"test_injected_{uuid.uuid4().hex[:8]}",
            sampling_params=_sampling_params(seed=42),
        )

        _assert_valid_image_output(output)

    assert marker.exists(), "injected scheduler never wrote its marker file"
    events = marker.read_text(encoding="utf-8").splitlines()
    assert "constructed" in events, f"injected scheduler was not constructed: {events}"
    assert "stepped" in events, f"injected scheduler never stepped: {events}"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_default_scheduler_path_without_injection(tmp_path, monkeypatch):
    """Control run: without ``scheduler``, the pipeline default is used and the
    injected test class is never touched."""
    marker = tmp_path / "scheduler_events.txt"
    monkeypatch.setenv(MARKER_ENV_VAR, str(marker))

    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            enforce_eager=True,
            max_num_seqs=1,
        )
        after.callback(engine.shutdown)

        output = await _generate_once(
            engine,
            request_id=f"test_default_{uuid.uuid4().hex[:8]}",
            sampling_params=_sampling_params(seed=42),
        )

        _assert_valid_image_output(output)

    assert not marker.exists(), "default path must not construct/step the injected scheduler"
