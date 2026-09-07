# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end streaming serving smoke test for LingBot-World v2.

One ``WS /v1/realtime/video`` session must drive the whole rollout and return
one video chunk per AR block. This is the entry point the recipe documents;
``tests/e2e/offline_inference/test_lingbot_world_v2_stepwise.py`` covers the
same step-execution contract at the latent level, below the transport.
"""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get(
    "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH",
    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
)
# The pipeline resizes the first frame to the requested geometry, so any real
# photograph conditions the rollout; the shared vLLM asset keeps the test
# runnable with no environment setup and adds no binary to this repository.
_IMAGE_ASSET = "2560px-Gfp-wisconsin-madison-the-nature-boardwalk"
_IMAGE_PATH_OVERRIDE = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH")

# The AR cache geometry is fixed at load time, so a request must ask for
# exactly the resolution the deploy config declares.
_HEIGHT = 480
_WIDTH = 832
_FRAMES_PER_BLOCK = 3
_TEMPORAL_COMPRESSION = 4
# One three-frame action list per generated chunk.
_CAMERA_ACTION_SCRIPT = [
    [["w"], ["w"], ["w"]],
    [["a"], [], []],
    [[], [], []],
]
_NUM_CHUNKS = len(_CAMERA_ACTION_SCRIPT)
_NUM_FRAMES = (_NUM_CHUNKS * _FRAMES_PER_BLOCK - 1) * _TEMPORAL_COMPRESSION + 1
_PROMPT = "The camera moves slowly forward through the scene."

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
]


def _first_frame() -> Path:
    if _IMAGE_PATH_OVERRIDE:
        return Path(_IMAGE_PATH_OVERRIDE).expanduser().resolve()
    return Path(ImageAsset(_IMAGE_ASSET).get_path("jpg"))


lingbot_world_v2_stepwise_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("lingbot_world_v2_stepwise.yaml"),
        ),
        id="lingbot_world_v2_stepwise",
    )
]


def _session_start(model: str, image: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(image.name)[0] or "image/png"
    encoded = base64.b64encode(image.read_bytes()).decode()
    return {
        "type": "session.start",
        "model": model,
        "prompt": _PROMPT,
        # LingBot is image-conditioned: without a first frame the request is
        # rejected during pre-processing.
        "image_reference": {"image_url": f"data:{media_type};base64,{encoded}"},
        "width": _WIDTH,
        "height": _HEIGHT,
        "num_frames": _NUM_FRAMES,
        "fps": 16,
        "seed": 42,
        "extra_params": {"flow_shift": 5.0, "camera_action_script": _CAMERA_ACTION_SCRIPT},
    }


async def _stream_session(url: str, model: str, image: Path) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    import websockets

    events: list[dict[str, Any]] = []
    media = bytearray()
    done: dict[str, Any] = {}
    async with websockets.connect(url, max_size=None, ping_interval=None) as websocket:
        await websocket.send(json.dumps(_session_start(model, image)))
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=900)
            if isinstance(message, bytes):
                media.extend(message)
                continue
            event = json.loads(message)
            events.append(event)
            if event["type"] == "error":
                pytest.fail(f"server returned an error event: {event}")
            if event["type"] == "session.done":
                done = event
                break
    return events, bytes(media), done


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", lingbot_world_v2_stepwise_server_params, indirect=True)
def test_lingbot_world_v2_stepwise_streams_one_chunk_per_block(omni_server) -> None:
    """Serve the real checkpoint and stream N chunks from a single WS session."""

    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/video"
    events, media, done = asyncio.run(_stream_session(url, omni_server.model, _first_frame()))

    started = [event for event in events if event["type"] == "video.start"]
    assert len(started) == 1, "one session.start must open exactly one rollout"

    # A trailer chunk may follow the media chunks; only media chunks carry a
    # generation index, and there is exactly one per AR block.
    media_chunks = [event for event in events if event["type"] == "video.chunk_metadata" and event["kind"] == "media"]
    assert [event["generation_chunk_index"] for event in media_chunks] == list(range(_NUM_CHUNKS))
    for event in media_chunks:
        assert event["request_id"] == started[0]["request_id"]
        assert event["num_frames"] > 0
        assert event["byte_length"] > 0

    assert len(media) == sum(event["byte_length"] for event in events if event["type"] == "video.chunk_metadata")
    assert done["stopped"] is False
    assert done["chunks"] >= _NUM_CHUNKS
