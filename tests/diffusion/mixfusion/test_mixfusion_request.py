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

from vllm_omni.diffusion.mixfusion.request import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MIN_CHUNK_TOKENS,
    mixfusion_request_extras,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _sampling(**extra_args):
    return SimpleNamespace(extra_args=dict(extra_args))


def _config(**overrides):
    values = {
        "enable_mixfusion": False,
        "mixfusion_min_chunk_tokens": 64,
        "mixfusion_max_chunks": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_unsupported_pipeline_gets_no_extras():
    assert (
        mixfusion_request_extras(_sampling(enable_mixfusion=True), _config(enable_mixfusion=True), supported=False)
        == {}
    )


def test_server_default_enables_mixfusion_with_config_thresholds():
    extras = mixfusion_request_extras(_sampling(), _config(enable_mixfusion=True), supported=True)

    assert extras == {
        "use_ragged_latents": True,
        "mixfusion_min_chunk_tokens": 64,
        "mixfusion_max_chunks": 16,
    }


def test_server_default_falls_back_to_shared_defaults():
    extras = mixfusion_request_extras(_sampling(), SimpleNamespace(enable_mixfusion=True), supported=True)

    assert extras["mixfusion_min_chunk_tokens"] == DEFAULT_MIN_CHUNK_TOKENS
    assert extras["mixfusion_max_chunks"] == DEFAULT_MAX_CHUNKS


def test_request_flag_overrides_server_default_off():
    extras = mixfusion_request_extras(_sampling(enable_mixfusion=True), _config(), supported=True)

    assert extras["use_ragged_latents"] is True


def test_request_flag_overrides_server_default_on():
    assert (
        mixfusion_request_extras(_sampling(enable_mixfusion=False), _config(enable_mixfusion=True), supported=True)
        == {}
    )


def test_request_thresholds_override_config():
    extras = mixfusion_request_extras(
        _sampling(enable_mixfusion=True, mixfusion_min_chunk_tokens=8, mixfusion_max_chunks=4),
        _config(enable_mixfusion=True),
        supported=True,
    )

    assert extras["mixfusion_min_chunk_tokens"] == 8
    assert extras["mixfusion_max_chunks"] == 4


def test_missing_sampling_extras_is_treated_as_empty():
    assert mixfusion_request_extras(SimpleNamespace(), _config(), supported=True) == {}
