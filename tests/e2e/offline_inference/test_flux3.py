# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end offline-serving smoke tests for Flux3 image generation modes.

These drive the *full* serving stack — registry lookup, real checkpoint loading
(real transformer weights), the omni engine + runner, ``forward()`` and VAE decode —
via ``OmniRunnerHandler.send_diffusion_request`` (which asserts a well-formed
response internally). They are value-agnostic: the text encoder currently returns
random conditioning, so we assert only that each mode runs end-to-end and returns a
response.

Mode/conditioning wiring mirrors ``Flux3Pipeline.forward``: I2I is selected by
the presence of ``multi_modal_data["image"]``.

Set ``FLUX3_MODEL_PATH`` to a flux3 checkpoint directory; the tests
skip when it is unset.
"""

from __future__ import annotations

import os

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


# TODO: remove once vLLM telemetry is globally disabled in CI.
def _assert_tracking_disabled() -> None:
    try:
        from vllm.usage.usage_lib import is_usage_stats_enabled
    except Exception:
        return
    if is_usage_stats_enabled():
        raise SystemExit(
            "vLLM usage-stats telemetry is ENABLED — aborting. "
            "Set VLLM_NO_USAGE_STATS=1 (or DO_NOT_TRACK=1) before running."
        )


_assert_tracking_disabled()

MODEL = os.getenv("FLUX3_MODEL_PATH")
PROMPT = "a scenic mountain landscape at golden hour"

pytestmark = [
    pytest.mark.skipif(not MODEL, reason="Set FLUX3_MODEL_PATH to a flux3 checkpoint directory."),
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize(
        "omni_runner",
        [(MODEL, None, {})],
        indirect=True,
    ),
]

# The cheapest image test additionally carries core_model so test-ready.yml (L2) picks it up.
_core_model = pytest.mark.core_model


def _sampling(height: int, width: int, **extra_args) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=2,
        seed=42,
        extra_args=dict(extra_args),
    )


def _cond_image():
    return ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")


# ── image-output modes ──────────────────────────────────────────────────────
@_core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux3_t2i(omni_runner_handler: OmniRunnerHandler):
    omni_runner_handler.send_diffusion_request(
        {"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling(512, 512)}
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux3_i2i(omni_runner_handler: OmniRunnerHandler):
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "restyle this image as an oil painting",
            "multi_modal_data": {"image": _cond_image()},
            "sampling_params": _sampling(512, 512),
        }
    )
