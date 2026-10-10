# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _policy(cache_backend: str = "none", cache_config: DiffusionCacheConfig | None = None):
    from vllm_omni.diffusion.models.pan2.quality_policy import PAN2QualityPolicy

    od_config = OmniDiffusionConfig(model=None, cache_backend=cache_backend, cache_config=cache_config or {})
    return PAN2QualityPolicy(od_config)


@pytest.mark.parametrize("cache_backend", ["none", "cache_dit"])
def test_high_quality_selects_pan2_profile(cache_backend):
    from vllm_omni.diffusion.models.pan2.quality_policy import PAN2_HIGH_CACHE_KEY, pan2_high_quality_cache_config

    spec = _policy(cache_backend, DiffusionCacheConfig()).resolve(quality="high", num_inference_steps=50)

    assert spec.installation_key == PAN2_HIGH_CACHE_KEY
    assert spec.cache_config == pan2_high_quality_cache_config()
    assert spec.num_inference_steps == 50


@pytest.mark.parametrize("cache_backend", ["none", "cache_dit"])
def test_lossless_runs_without_cache(cache_backend):
    assert _policy(cache_backend, DiffusionCacheConfig()).resolve(quality="lossless", num_inference_steps=50) is None


def test_omitted_quality_follows_the_server_cache_backend():
    from vllm_omni.diffusion.models.pan2.quality_policy import PAN2_GENERIC_CACHE_KEY

    server_config = DiffusionCacheConfig(residual_diff_threshold=0.12)
    spec = _policy("cache_dit", server_config).resolve(quality=None, num_inference_steps=30)
    assert spec.installation_key == PAN2_GENERIC_CACHE_KEY
    assert spec.cache_config == server_config
    assert spec.num_inference_steps == 30

    assert _policy("none").resolve(quality=None, num_inference_steps=30) is None
