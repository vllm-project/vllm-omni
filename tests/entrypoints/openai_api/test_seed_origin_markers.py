# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The serving layer marks whether a seed came from the client or stage defaults.

Seeded-request isolation keys on a client-supplied seed only, so a deploy-level
``default_sampling_params.seed`` must not make every request look reproducible.
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.serving_speech import _mark_seed_origin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage_default_seed_is_marked_as_default() -> None:
    params = [SamplingParams(seed=42), SamplingParams(seed=None)]
    _mark_seed_origin(params, request_seed=None)

    assert params[0].extra_args == {"omni_default_seed": True}
    # A stage without a seed keeps its params untouched.
    assert params[1].extra_args is None


def test_client_seed_is_marked_as_user_seed() -> None:
    params = [SamplingParams(seed=42)]
    _mark_seed_origin(params, request_seed=7)

    assert params[0].extra_args == {"omni_user_seed": True}


def test_existing_extra_args_are_preserved() -> None:
    params = [SamplingParams(seed=42, extra_args={"tts_local_seed": 7})]
    _mark_seed_origin(params, request_seed=7)

    assert params[0].extra_args == {"tts_local_seed": 7, "omni_user_seed": True}
