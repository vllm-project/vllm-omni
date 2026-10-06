# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline smoke for Kandinsky 6 TI2VA.

Short request (256x256, 5 frames, 2 steps) so it finishes inside the default
online poll window. Pro geometry (480x864, 125 frames, 50 steps) stays the
serving default, not this smoke.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient, get_model_prefix
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

_HUB_MODEL = "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
_LOCAL_BUNDLE = Path(__file__).resolve().parents[4] / "kandinsky6_bundle"


def _resolve_model() -> str:
    env = os.environ.get("KANDINSKY6_MODEL")
    if env:
        return env
    if _LOCAL_BUNDLE.is_dir():
        return str(_LOCAL_BUNDLE)
    return _HUB_MODEL


def _checkpoint_on_disk(model: str) -> bool:
    if Path(model).is_dir():
        return True
    prefix = get_model_prefix()
    return bool(prefix) and Path(prefix + model).is_dir()


MODEL = _resolve_model()
PROMPT = "A golden retriever runs along a sunny beach, waves crashing."

pytestmark = [
    pytest.mark.skipif(
        not _checkpoint_on_disk(MODEL),
        reason=(
            "Kandinsky 6 Diffusers weights are gated. Set KANDINSKY6_MODEL or "
            "MODEL_PREFIX to a local copy of "
            "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers."
        ),
    ),
]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize(
    "omni_runner",
    [(MODEL, None, {"enable_cpu_offload": True, "enforce_eager": True})],
    indirect=True,
)
def test_text_to_video_and_audio_001(offline_client: OfflineOmniClient):
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_frames=5,
        fps=24,
        num_inference_steps=2,
        guidance_scale=5.0,
        seed=42,
        extra_args={"sample_audio": True},
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "sampling_params": sampling,
    }
    offline_client.send_diffusion_request(request_config)
