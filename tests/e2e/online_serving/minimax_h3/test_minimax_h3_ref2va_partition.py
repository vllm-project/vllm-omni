# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Light single-partition Ref2VA boot for the Merge DP2 job.

The combined DP2+DLO server loads Ref2VA as ``transformers_ref``. This
process boots ``--task-type ref2va`` so a live GPU server still takes the
per-partition download, checks ``expected_partition == "ref2va"``, and
denoises on ``self.transformer``.

``--enforce-eager`` skips inductor compile. The wave is two concurrent
512x288 one-step clips: one step is a supported full-range schedule, 4s is
the duration floor, and DP AllGather still needs a homogeneous pair. The
full 1344x768 four-step FL2VA and Ref2VA waves stay on the combined server.
This file is its own pytest process so a DLO teardown cannot strand Turbo
or FastH3.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

from ._common import DLO_SERVER_ARGS, FPS, MODEL, assert_h3_video, post_sync, run_dlo_wave

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]
TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100", "npu": "A3"}, num_cards=2)
_PARTITION_WIDTH = 512
_PARTITION_HEIGHT = 288
# One denoise evaluation. ``minimax_h3_time_shift_sigmas`` still emits both
# endpoints, so the step walks the full [1, 0] range.
_PARTITION_STEPS = 1
_REFERENCE = base64.b64decode(generate_synthetic_image(_PARTITION_WIDTH, _PARTITION_HEIGHT, seed=43)["base64"])


def _partition_params() -> OmniServerParams:
    """DP2+DLO Ref2VA partition. Eager skips inductor compile in this process."""
    return OmniServerParams(
        model=MODEL,
        server_args=[*DLO_SERVER_ARGS, "--task-type", "ref2va", "--enforce-eager"],
        stage_init_timeout=1800,
        init_timeout=1800,
    )


def _run_small_ref2va(client: OpenAIClientHandler, seed: int) -> bytes:
    """One image-only Ref2VA request at the small eager shape."""
    form = {
        "model": MODEL,
        "prompt": "A cinematic live-action scene with synchronized environmental sound.",
        "width": str(_PARTITION_WIDTH),
        "height": str(_PARTITION_HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(_PARTITION_STEPS),
        "flow_shift": "12",
        "seed": str(seed),
        "extra_params": json.dumps(
            {
                "task": "ref2va",
                "duration": 4.0,
                "aspect_ratio": "16:9",
                "audio_flow_shift": 3.0,
            },
            separators=(",", ":"),
        ),
    }
    return post_sync(
        client,
        form,
        files=[("input_references", ("reference.jpg", io.BytesIO(_REFERENCE), "image/jpeg"))],
    )


@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(_partition_params(), id="minimax_h3_dlo_dp2_ref2va_partition", marks=TWO_CARD_MARKS)],
    indirect=True,
)
def test_minimax_h3_ref2va_partition(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Run one eager DP wave on the Ref2VA partition's own transformer."""
    for video in run_dlo_wave(openai_client, _run_small_ref2va):
        assert_h3_video(video, width=_PARTITION_WIDTH, height=_PARTITION_HEIGHT)
