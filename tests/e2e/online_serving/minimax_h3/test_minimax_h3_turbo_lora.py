# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 Turbo LoRA + DLO L3 online-serving case."""

from __future__ import annotations

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

from ._common import (
    TURBO_LORA,
    TURBO_SUPPORTED,
    assert_h3_video,
    fl2va_files,
    post_sync,
    turbo_form,
    turbo_params,
)

pytestmark = [pytest.mark.core_model, pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.slow]
H100_TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


@pytest.mark.skipif(
    not TURBO_SUPPORTED,
    reason="requires the MiniMax-H3 Turbo loader from PR #6550",
)
@pytest.mark.skipif(
    TURBO_LORA is None,
    reason="set VLLM_TEST_MINIMAX_H3_TURBO_LORA or populate the local HF cache",
)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            turbo_params(TURBO_LORA),
            id="minimax_h3_dlo_turbo_lora_tp2",
            marks=H100_TWO_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_minimax_h3_dlo_turbo_lora_fl2va(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Validate the official four-evaluation Turbo LoRA + DLO FL2VA request."""
    video = post_sync(
        openai_client,
        turbo_form(seed=2201),
        files=fl2va_files(),
    )
    assert_h3_video(video, width=1344, height=768)
