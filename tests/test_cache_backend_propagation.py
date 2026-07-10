# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for diffusion cache backend propagation to deploy-YAML stages.

Guards the bug where the top-level ``--cache-backend`` (and its ``cache_config``)
is silently dropped under a deploy YAML: per-stage engine args default
``cache_backend`` to "none", and the top-level value was never injected into the
diffusion stage, so the diffusion cache (TeaCache / cache-dit) never activated.

The tests drive the real ``AsyncOmniEngine._resolve_stage_configs`` flow (only the
YAML/model stage loading is stubbed out) rather than calling the injection helper
directly, so they stay valid if the internal wiring is refactored.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.engine.async_omni_engine"


def _stage(stage_type="diffusion", cache_backend=None):
    ea = OmegaConf.create({})
    if cache_backend is not None:
        ea.cache_backend = cache_backend
    return SimpleNamespace(stage_type=stage_type, engine_args=ea)


def _resolve(stages, **cache_kwargs):
    """Run _resolve_stage_configs end-to-end with stage loading stubbed.

    ``cache_kwargs`` carries the top-level CLI knobs (cache_backend/cache_config)
    exactly as they arrive under ``--stage-configs-path``.
    """
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    kwargs = {"stage_configs_path": "deploy.yaml", **cache_kwargs}
    with patch(f"{_MODULE}.load_and_resolve_stage_configs", return_value=("deploy.yaml", stages)):
        engine._resolve_stage_configs("test-model", kwargs)
    return stages


def test_cache_backend_and_config_injected():
    """Top-level cache_backend/cache_config reach a diffusion stage that has none.

    ``cache_config`` arrives from the CLI as a raw JSON string (``--cache-config``
    is ``type=str``); it must be parsed into a dict before reaching the stage.
    """
    stage = _stage()
    _resolve([stage], cache_backend="tea_cache", cache_config='{"rel_l1_thresh": 0.2}')
    assert stage.engine_args.cache_backend == "tea_cache"
    assert dict(stage.engine_args.cache_config) == {"rel_l1_thresh": 0.2}


def test_none_or_missing_is_noop():
    """cache_backend="none" or absent must not set anything (no silent enable)."""
    stage = _stage()
    _resolve([stage], cache_backend="none")
    assert getattr(stage.engine_args, "cache_backend", None) in (None, "none")

    stage2 = _stage()
    _resolve([stage2])
    assert getattr(stage2.engine_args, "cache_backend", None) in (None, "none")


def test_explicit_stage_value_wins():
    """An explicit per-stage cache_backend is not overwritten by the top-level one."""
    stage = _stage(cache_backend="cache_dit")
    _resolve([stage], cache_backend="tea_cache")
    assert stage.engine_args.cache_backend == "cache_dit"


def test_config_not_copied_to_different_backend():
    """A top-level cache_config must not be attached to a stage that explicitly
    selected a different backend (the config belongs to the top-level backend)."""
    stage = _stage(cache_backend="cache_dit")
    _resolve([stage], cache_backend="tea_cache", cache_config='{"rel_l1_thresh": 0.2}')
    assert stage.engine_args.cache_backend == "cache_dit"
    assert getattr(stage.engine_args, "cache_config", None) in (None, {})


def test_config_filled_for_matching_stage_backend():
    """When the stage already selected the same backend but has no config, the
    top-level cache_config is filled in."""
    stage = _stage(cache_backend="tea_cache")
    _resolve([stage], cache_backend="tea_cache", cache_config='{"rel_l1_thresh": 0.3}')
    assert stage.engine_args.cache_backend == "tea_cache"
    assert dict(stage.engine_args.cache_config) == {"rel_l1_thresh": 0.3}


def test_non_diffusion_stage_untouched():
    """Non-diffusion stages must never receive the diffusion cache backend."""
    stage = _stage(stage_type="thinker")
    _resolve([stage], cache_backend="tea_cache", cache_config='{"rel_l1_thresh": 0.2}')
    assert getattr(stage.engine_args, "cache_backend", None) is None
