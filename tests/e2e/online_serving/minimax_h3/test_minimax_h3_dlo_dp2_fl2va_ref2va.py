# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 DP2+DLO waves, split across Merge and CUDA Nightly.

``--task-type combined`` loads both DiTs. Each full-wave request still
selects its task through ``extra_params.task`` (``run_fl2va`` /
``run_ref2va``). Those two tests share one ``OmniServerParams``. pytest
9.1.1 reuses the module-scoped ``omni_server`` fixture while that value
stays equal, and finalizes it when a different value appears. A DLO
teardown can leave a stale ``resource_tracker`` ``/psm_*`` entry.

The Ref2VA partition case boots ``--task-type ref2va`` so a live CUDA GPU
still takes the per-partition download, checks ``expected_partition ==
"ref2va"``, and denoises on ``self.transformer`` (not ``transformers_ref``).
It is marked ``full_model`` only. The full waves are marked
``advanced_model`` only. Merge ``-m advanced_model`` and the CUDA Nightly
Function Test ``-m full_model`` therefore never collect them in one
process. Running this file with no level marker does.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

from ._common import (
    DLO_SERVER_ARGS,
    FPS,
    MODEL,
    assert_h3_video,
    dlo_params,
    post_sync,
    run_dlo_wave,
    run_fl2va,
    run_ref2va,
)

# Level marks are per test. A module-level advanced_model mark would put the
# partition boot back on the Merge command that already runs this file.
pytestmark = pytest.mark.diffusion
TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100", "npu": "A3"}, num_cards=2)
# H100 and B200 share the item so the X2V Function Test expression
# ``H100 and B200 and cards_2`` collects it. The server stays DP2+DLO.
PARTITION_MARKS = hardware_marks(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=2)
# One object so the two full-wave tests cannot drift into two boots.
_COMBINED_SERVER = pytest.param(
    dlo_params("combined"),
    id="minimax_h3_dlo_dp2_combined",
    marks=TWO_CARD_MARKS,
)
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


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", [_COMBINED_SERVER], indirect=True)
def test_minimax_h3_dlo_dp2_fl2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate a complete concurrent DLO/DP2 wave with two FL2VA jobs."""
    for video in run_dlo_wave(openai_client, run_fl2va):
        assert_h3_video(video, width=1344, height=768)


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", [_COMBINED_SERVER], indirect=True)
def test_minimax_h3_dlo_dp2_ref2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate a complete concurrent DLO/DP2 wave with two image-only Ref2VA jobs."""
    for video in run_dlo_wave(openai_client, run_ref2va):
        assert_h3_video(video, width=1344, height=768)


@pytest.mark.full_model
@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(_partition_params(), id="minimax_h3_dlo_dp2_ref2va_partition", marks=PARTITION_MARKS)],
    indirect=True,
)
def test_minimax_h3_ref2va_partition(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Run one eager DP wave on the Ref2VA partition's own transformer."""
    for video in run_dlo_wave(openai_client, _run_small_ref2va):
        assert_h3_video(video, width=_PARTITION_WIDTH, height=_PARTITION_HEIGHT)
