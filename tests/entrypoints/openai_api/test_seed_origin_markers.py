# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The serving layer marks only client-supplied seeds for isolation.

Seeded-request isolation keys on the ``omni_user_seed`` marker only, so a
deploy-level ``default_sampling_params.seed`` must stay unmarked and keep normal
batching.
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.serving_speech import _mark_seed_origin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage_default_seed_stays_unmarked() -> None:
    params = [SamplingParams(seed=42)]
    _mark_seed_origin(params, request_seed=None)

    assert params[0].extra_args != {"omni_user_seed": True}
    assert (params[0].extra_args or {}).get("omni_user_seed") is None


def test_client_seed_marks_the_matching_stage_only() -> None:
    params = [SamplingParams(seed=7), SamplingParams(seed=42)]
    _mark_seed_origin(params, request_seed=7)

    assert params[0].extra_args == {"omni_user_seed": True}
    assert params[1].extra_args != {"omni_user_seed": True}


def test_shared_defaults_are_not_mutated(monkeypatch) -> None:
    shared = SamplingParams(seed=7)
    params = [shared]
    _mark_seed_origin(params, request_seed=7)

    # The marked copy replaced the shared object; the default itself is intact.
    assert params[0] is not shared
    assert shared.extra_args is None


def test_existing_extra_args_are_preserved() -> None:
    params = [SamplingParams(seed=7, extra_args={"tts_local_seed": 7})]
    _mark_seed_origin(params, request_seed=7)

    assert params[0].extra_args == {"tts_local_seed": 7, "omni_user_seed": True}
