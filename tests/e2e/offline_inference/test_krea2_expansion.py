# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L4 functionality tests for the Krea 2 text-to-image diffusion pipeline (``Krea2Pipeline``).

Runs against the public few-step distilled checkpoint ``krea/Krea-2-Turbo`` by default (override with the
``KREA2_MODEL`` environment variable, e.g. ``krea/Krea-2-Raw`` for the Raw checkpoint or a local diffusers
directory). They cover a basic functional smoke plus the layerwise-CPU-offload path that the pipeline declares
via ``SupportsComponentDiscovery`` / ``_layerwise_offload_blocks_attrs``.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
PROMPT = "a fox in the snow, photorealistic"

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.full_model,
]


def _sampling() -> OmniDiffusionSamplingParams:
    # Small resolution + few steps to keep the L4 case light. guidance_scale resolution is checkpoint-aware inside
    # the pipeline (distilled -> no-CFG, Raw -> CFG), so this stays agnostic to which checkpoint KREA2_MODEL points at.
    return OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=0.0,
        seed=42,
    )


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_krea2_text_to_image_001(omni_runner_handler: OmniRunnerHandler) -> None:
    """Basic functional smoke: a single prompt produces a decoded image."""
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize(
    "omni_runner",
    [(MODEL, None, {"enable_layerwise_offload": True})],
    indirect=True,
)
def test_krea2_layerwise_offload(omni_runner_handler: OmniRunnerHandler) -> None:
    """Exercise layerwise CPU offload on the DiT (SupportsComponentDiscovery + _layerwise_offload_blocks_attrs)."""
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})
