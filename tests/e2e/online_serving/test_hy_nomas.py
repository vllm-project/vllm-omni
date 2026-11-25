# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os

import pytest

from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, get_model_prefix
from tests.model_tests.diffusion.utils import resolve_tiny_model_path

pytestmark = pytest.mark.diffusion

MODEL = os.environ.get("HY_NOMAS_MODEL", "tencent/Hy-Nomas-Diffusers")


@pytest.mark.local_model
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--deploy-config", "vllm_omni/deploy/hy_nomas.yaml"]),
            id="single_device",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=1),
        ),
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--enforce-eager", "--ulysses-degree", "2"]),
            id="sp2",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=2),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt,extra,num_frames",
    [
        pytest.param("A forest beside a river.", {}, 9, id="text"),
        pytest.param(
            "A forest beside a river. ||| The camera moves along the river.",
            {
                "segment_frame_boundaries": [0, 4, 9],
                "camera_trajectory": "pan_left",
                "segment_has_camera": [True, True],
            },
            9,
            id="camera_segments",
        ),
        pytest.param("A beach at sunrise.", {"duration": 1}, None, id="duration_without_audio"),
    ],
)
def test_video_job(
    omni_server: OmniServer, online_client: OnlineOmniClient, prompt: str, extra: dict, num_frames: int | None
):
    responses = online_client.send_video_diffusion_request(
        {
            "model": omni_server.model,
            "form_data": {
                "prompt": prompt,
                "size": "256x256",
                "num_frames": num_frames,
                "generate_sound": False,
                "fps": 24,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "seed": 42,
                "extra_params": json.dumps(extra),
            },
        }
    )
    assert responses[0].videos
    assert_video_valid(
        responses[0].videos[0], num_frames=25 if num_frames is None else num_frames, width=256, height=256, fps=24
    )


@pytest.mark.parametrize(
    ("sp_size", "num_frames"),
    [
        pytest.param(
            sp_size,
            (10 + (sp_size - 1) * 5 - 1) * 4 + 1,
            id=f"window-sp{sp_size}",
            marks=[
                *hardware_marks(res={"cuda": ["H200", "B200"] if sp_size == 2 else "H200"}, num_cards=sp_size),
                pytest.mark.full_model if sp_size == 2 else pytest.mark.local_model,
            ],
        )
        for sp_size in (2, 4, 8)
    ],
)
def test_window_parallel_video_job(sp_size: int, num_frames: int, tmp_path, run_level: str):
    model_name = get_model_prefix() + MODEL
    model_path = resolve_tiny_model_path(model_name) if run_level == "core_model" else model_name
    args = [
        "--enforce-eager",
        "--ulysses-degree",
        str(sp_size),
        "--stage-overrides",
        json.dumps({"0": {"devices": ",".join(str(rank) for rank in range(sp_size))}}),
        "--custom-pipeline-args",
        '{"decoder_parallel_mode":"window"}',
        "--revision",
        "a1ae04d11ae4ea176d1f76de634c115944f9882d",
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "2100",
    ]
    if model_path != model_name:
        args.extend(["--served-model-name", model_name])
    env = {"VLLM_OMNI_SERVER_STORAGE__PATH": str(tmp_path)}
    with OmniServer(model_path, args, env_dict=env, startup_timeout=2100) as server:
        client = OnlineOmniClient(host=server.host, port=server.port, run_level=run_level)
        client.send_video_diffusion_request(
            {
                "model": model_name,
                "form_data": {
                    "prompt": "A forest beside a river.",
                    "width": 256,
                    "height": 256,
                    "num_frames": num_frames,
                    "generate_sound": False,
                    "fps": 24,
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "seed": 42,
                },
            }
        )
