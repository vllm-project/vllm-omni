# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for StepCacheBackend and StepCacheState."""

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.cache.stepcache import (
    STEP_CACHE_CONFIG_ATTR,
    STEP_CACHE_STATE_ATTR,
    StepCacheBackend,
    StepCacheConfig,
    StepCacheState,
    is_stepcache_active,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_prev_predictions(sim: float, dim: int = 8) -> list[tuple[torch.Tensor]]:
    v_prev = torch.randn(1, dim)
    if sim >= 0.99:
        v_last = v_prev.clone()
    else:
        v_last = torch.randn(1, dim)
    return [(v_prev,), (v_last,)]


class TestStepCacheConfig:
    def test_from_diffusion_cache_config_respects_step_cache_dit_enabled(self):
        enabled = StepCacheConfig.from_diffusion_cache_config(DiffusionCacheConfig(step_cache_dit_enabled=True))
        disabled = StepCacheConfig.from_diffusion_cache_config(DiffusionCacheConfig(step_cache_dit_enabled=False))
        assert enabled.enabled is True
        assert disabled.enabled is False


class TestStepCacheState:
    def test_disabled_always_runs(self):
        state = StepCacheState(StepCacheConfig(enabled=False))
        assert state.should_run_step(_make_prev_predictions(0.99)) is True

    def test_insufficient_history_always_runs(self):
        state = StepCacheState(StepCacheConfig(min_history_steps=2))
        assert state.should_run_step([(torch.randn(1, 4),)]) is True

    def test_high_similarity_skips_with_countdown(self):
        state = StepCacheState(
            StepCacheConfig(
                sim_thresholds=(0.95,),
                skip_countdowns=(2,),
            )
        )
        preds = _make_prev_predictions(0.99)
        assert state.should_run_step(preds) is False
        assert state.skip_countdown == 2
        assert state.should_run_step(preds) is False
        assert state.should_run_step(preds) is True

    def test_low_similarity_runs(self):
        state = StepCacheState(StepCacheConfig())
        preds = _make_prev_predictions(0.5)
        assert state.should_run_step(preds) is True

    def test_trim_history(self):
        state = StepCacheState(StepCacheConfig(max_history=2))
        history: list[tuple[torch.Tensor]] = [(torch.randn(1, 2),) for _ in range(4)]
        state.trim_history(history)
        assert len(history) == 2


class TestStepCacheBackend:
    def test_enable_dreamzero_pipeline(self):
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "DreamZeroPipeline"

        backend = StepCacheBackend(DiffusionCacheConfig())
        backend.enable(mock_pipeline)

        assert backend.enabled is True
        cfg = getattr(mock_pipeline, STEP_CACHE_CONFIG_ATTR)
        assert isinstance(cfg, StepCacheConfig)
        assert is_stepcache_active(mock_pipeline)
        state = getattr(mock_pipeline, STEP_CACHE_STATE_ATTR)
        assert isinstance(state, StepCacheState)

    def test_refresh_resets_state(self):
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "DreamZeroPipeline"
        backend = StepCacheBackend(DiffusionCacheConfig())
        backend.enable(mock_pipeline)
        state = getattr(mock_pipeline, STEP_CACHE_STATE_ATTR)
        state.skip_countdown = 3
        backend.refresh(mock_pipeline, num_inference_steps=16, verbose=False)
        assert state.skip_countdown == 0

    def test_enable_unsupported_pipeline_raises(self):
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "QwenImagePipeline"
        backend = StepCacheBackend(DiffusionCacheConfig())

        with pytest.raises(ValueError, match="step_cache backend does not support"):
            backend.enable(mock_pipeline)

    def test_selector_returns_backend(self):
        backend = get_cache_backend("step_cache", {"velocity_sim_thresholds": [0.9]})
        assert isinstance(backend, StepCacheBackend)
        assert backend.config.velocity_sim_thresholds == [0.9]

    def test_legacy_backend_aliases(self):
        assert isinstance(get_cache_backend("stepcache", {}), StepCacheBackend)
        assert isinstance(get_cache_backend("step_cache_dit", {}), StepCacheBackend)
