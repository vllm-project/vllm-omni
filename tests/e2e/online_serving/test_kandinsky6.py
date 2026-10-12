# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Online smoke for Kandinsky 6 TI2VA via ``/v1/videos``.

Short geometry so the job finishes inside the helper's default 300 s poll.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, get_model_prefix

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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
            "Kandinsky 6 Diffusers weights are not on disk. Download "
            "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers or set "
            "KANDINSKY6_MODEL / MODEL_PREFIX."
        ),
    ),
]

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--enable-cpu-offload", "--enforce-eager"],
                init_timeout=3600,
                stage_init_timeout=3600,
            ),
            id="cpu_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_video_and_audio_001(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Short TI2VA job completes and returns an MP4."""
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "height": 256,
            "width": 256,
            "num_frames": 5,
            "fps": 24,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "seed": 42,
        },
    }
    online_client.send_video_diffusion_request(request_config)
