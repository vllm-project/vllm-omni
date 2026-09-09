# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 FL2VA DLO + DP2 L3 online-serving case."""

from __future__ import annotations

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

from ._common import assert_h3_video, dlo_params, run_dlo_wave, run_fl2va

pytestmark = [pytest.mark.core_model, pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.slow]
H100_TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(dlo_params("fl2va"), id="minimax_h3_dlo_dp2_fl2va", marks=H100_TWO_CARD_MARKS)],
    indirect=True,
)
def test_minimax_h3_dlo_dp2_fl2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate a complete concurrent DLO/DP2 wave with two FL2VA jobs."""
    for video in run_dlo_wave(openai_client, run_fl2va):
        assert_h3_video(video, width=1344, height=768)
