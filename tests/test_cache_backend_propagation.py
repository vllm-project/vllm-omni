# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for diffusion cache backend propagation to deploy-YAML stages.

Guards the bug where the top-level ``--cache-backend`` (and its ``cache_config``)
is silently dropped under a deploy YAML: per-stage engine args default
``cache_backend`` to "none", and the top-level value was never injected into the
diffusion stage, so the diffusion cache (TeaCache / cache-dit) never activated.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _diffusion_stage(cache_backend=None):
    ea = OmegaConf.create({})
    if cache_backend is not None:
        ea.cache_backend = cache_backend
    return SimpleNamespace(stage_type="diffusion", engine_args=ea)


def test_cache_backend_and_config_injected():
    """Top-level cache_backend/cache_config reach a diffusion stage that has none.

    ``cache_config`` arrives from the CLI as a raw JSON string (``--cache-config``
    is ``type=str``); it must be parsed into a dict before reaching the stage.
    """
    cfg = _diffusion_stage()
    AsyncOmniEngine._inject_stage_cache_backend(
        cfg, {"cache_backend": "tea_cache", "cache_config": '{"rel_l1_thresh": 0.2}'}
    )
    assert cfg.engine_args.cache_backend == "tea_cache"
    assert dict(cfg.engine_args.cache_config) == {"rel_l1_thresh": 0.2}


def test_none_or_missing_is_noop():
    """cache_backend="none" or absent must not set anything (no silent enable)."""
    cfg = _diffusion_stage()
    AsyncOmniEngine._inject_stage_cache_backend(cfg, {"cache_backend": "none"})
    assert getattr(cfg.engine_args, "cache_backend", None) in (None, "none")
    AsyncOmniEngine._inject_stage_cache_backend(cfg, {})
    assert getattr(cfg.engine_args, "cache_backend", None) in (None, "none")


def test_explicit_stage_value_wins():
    """An explicit per-stage cache_backend is not overwritten by the top-level one."""
    cfg = _diffusion_stage(cache_backend="cache_dit")
    AsyncOmniEngine._inject_stage_cache_backend(cfg, {"cache_backend": "tea_cache"})
    assert cfg.engine_args.cache_backend == "cache_dit"


def test_config_not_copied_to_different_backend():
    """A top-level cache_config must not be attached to a stage that explicitly
    selected a different backend (the config belongs to the top-level backend)."""
    cfg = _diffusion_stage(cache_backend="cache_dit")
    AsyncOmniEngine._inject_stage_cache_backend(
        cfg, {"cache_backend": "tea_cache", "cache_config": '{"rel_l1_thresh": 0.2}'}
    )
    assert cfg.engine_args.cache_backend == "cache_dit"
    assert getattr(cfg.engine_args, "cache_config", None) in (None, {})


def test_config_filled_for_matching_stage_backend():
    """When the stage already selected the same backend but no config, the
    top-level cache_config is filled in."""
    cfg = _diffusion_stage(cache_backend="tea_cache")
    AsyncOmniEngine._inject_stage_cache_backend(
        cfg, {"cache_backend": "tea_cache", "cache_config": '{"rel_l1_thresh": 0.3}'}
    )
    assert cfg.engine_args.cache_backend == "tea_cache"
    assert dict(cfg.engine_args.cache_config) == {"rel_l1_thresh": 0.3}
