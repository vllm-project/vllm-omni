# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request MixFusion opt-in plumbing (CPU).

The runner stamps these extras on newly admitted step requests, so the
opt-in precedence and threshold defaults are pinned here rather than in a
model pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.mixfusion.request import mixfusion_request_extras

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

CONFIG_THRESHOLDS = {"mixfusion_min_chunk_tokens": 64, "mixfusion_max_chunks": 16}
ENABLED = {"use_ragged_latents": True, **CONFIG_THRESHOLDS}


@pytest.mark.parametrize(
    ("extra_args", "server_default", "supported", "expected"),
    [
        # The pipeline capability gates everything.
        ({"enable_mixfusion": True}, True, False, {}),
        ({}, True, True, ENABLED),
        ({}, False, True, {}),
        # The request flag overrides the server default either way.
        ({"enable_mixfusion": True}, False, True, ENABLED),
        ({"enable_mixfusion": False}, True, True, {}),
        # Request thresholds override the config ones.
        (
            {"enable_mixfusion": True, "mixfusion_min_chunk_tokens": 8, "mixfusion_max_chunks": 4},
            False,
            True,
            {"use_ragged_latents": True, "mixfusion_min_chunk_tokens": 8, "mixfusion_max_chunks": 4},
        ),
    ],
)
def test_mixfusion_request_extras(extra_args, server_default, supported, expected):
    sampling = SimpleNamespace(extra_args=dict(extra_args))
    config = SimpleNamespace(enable_mixfusion=server_default, **CONFIG_THRESHOLDS)

    assert mixfusion_request_extras(sampling, config, supported=supported) == expected
