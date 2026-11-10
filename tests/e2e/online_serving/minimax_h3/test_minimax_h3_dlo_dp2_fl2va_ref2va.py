# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 FL2VA and Ref2VA on one DP2 + DLO combined server.

``--task-type combined`` loads both DiTs. Each request still selects its task
through ``extra_params.task`` (``run_fl2va`` / ``run_ref2va``). The two waves
stay sequential and homogeneous: DLO AllGather requires every rank in a DP
wave to execute the same DiT path.

Both tests pass one ``OmniServerParams`` value. pytest 9.1.1 (the repo pin)
reuses the module-scoped ``omni_server`` fixture while that value stays equal,
and finalizes it when a different value appears. This module must not add
another server param. DLO teardown can leave a stale ``resource_tracker``
``/psm_*`` entry in the parent and strand the next collective. Turbo LoRA
(TP2) and FastH3 LoRA (HSDP/USP) stay in their own pytest processes for that
reason and because those layouts are not this DP2 server. The Ref2VA
partition's own transformer is covered by
``test_minimax_h3_ref2va_partition.py``, also its own process.
"""

from __future__ import annotations

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

from ._common import assert_h3_video, dlo_params, run_dlo_wave, run_fl2va, run_ref2va

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]
TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100", "npu": "A3"}, num_cards=2)
# One object so the two tests cannot drift into two boots.
_COMBINED_SERVER = pytest.param(
    dlo_params("combined"),
    id="minimax_h3_dlo_dp2_combined",
    marks=TWO_CARD_MARKS,
)


@pytest.mark.parametrize("omni_server", [_COMBINED_SERVER], indirect=True)
def test_minimax_h3_dlo_dp2_fl2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate a complete concurrent DLO/DP2 wave with two FL2VA jobs."""
    for video in run_dlo_wave(openai_client, run_fl2va):
        assert_h3_video(video, width=1344, height=768)


@pytest.mark.parametrize("omni_server", [_COMBINED_SERVER], indirect=True)
def test_minimax_h3_dlo_dp2_ref2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate a complete concurrent DLO/DP2 wave with two image-only Ref2VA jobs."""
    for video in run_dlo_wave(openai_client, run_ref2va):
        assert_h3_video(video, width=1344, height=768)
