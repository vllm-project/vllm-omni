# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _quality_od_config(*, cache_backend: str = "cache_dit", **overrides):
    values = {
        "cache_backend": cache_backend,
        "cache_config": DiffusionCacheConfig(),
        "diffusion_attention_config": SimpleNamespace(default=None, per_role={}),
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
        "enforce_eager": True,
        "num_gpus": 4,
        "quantization_config": None,
        "parallel_config": SimpleNamespace(
            allgather_degree=1,
            cfg_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            ring_degree=1,
            sequence_parallel_size=4,
            tensor_parallel_size=1,
            ulysses_degree=4,
            use_hsdp=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolve_quality(policy, **overrides):
    values = {
        "quality": "high",
        "num_inference_steps": 50,
    }
    values.update(overrides)
    return policy.resolve(**values)


def test_high_quality_policy_emits_cache_profile_without_deployment_gates():
    from vllm_omni.diffusion.models.minimax_h3.quality_policy import (
        MiniMaxH3QualityPolicy,
    )

    policy = MiniMaxH3QualityPolicy(
        _quality_od_config(
            enable_layerwise_offload=True,
            enforce_eager=False,
            num_gpus=1,
        )
    )

    plan = _resolve_quality(policy, num_inference_steps=17)
    spec = plan.cache_dit

    assert plan.level == "high"
    assert spec is not None
    assert spec.installation_key == "minimax_h3.high"
    assert spec.num_inference_steps == 17
    assert spec.cache_config.Fn_compute_blocks == 1
    assert spec.cache_config.Bn_compute_blocks == 0
    assert spec.cache_config.max_warmup_steps == 4
    assert spec.cache_config.residual_diff_threshold == pytest.approx(0.04)
    assert spec.cache_config.max_continuous_cached_steps == 1
    assert spec.cache_config.enable_taylorseer is False
    assert spec.cache_config.scm_steps_mask_policy is None


def test_high_quality_requires_cache_dit_startup_capability():
    from vllm_omni.diffusion.models.minimax_h3.quality_policy import (
        MiniMaxH3QualityPolicy,
    )

    policy = MiniMaxH3QualityPolicy(
        _quality_od_config(cache_backend="none"),
    )

    with pytest.raises(ValueError, match="requires the server to start"):
        _resolve_quality(policy)


@pytest.mark.parametrize(
    ("cache_backend", "quality", "expected_key"),
    [
        ("cache_dit", None, "minimax_h3.generic"),
        ("cache_dit", "lossless", None),
        ("cache_dit", "high", "minimax_h3.high"),
        ("none", None, None),
        ("none", "lossless", None),
    ],
)
def test_model_policy_owns_request_cache_target(cache_backend, quality, expected_key):
    from vllm_omni.diffusion.models.minimax_h3.quality_policy import (
        MiniMaxH3QualityPolicy,
    )

    config = _quality_od_config(cache_backend=cache_backend)
    policy = MiniMaxH3QualityPolicy(config)

    plan = _resolve_quality(
        policy,
        quality=quality,
    )

    assert plan.level == quality
    if expected_key is None:
        assert plan.cache_dit is None
    else:
        assert plan.cache_dit is not None
        assert plan.cache_dit.installation_key == expected_key
        if quality is None:
            assert plan.cache_dit.cache_config is config.cache_config


def test_h3_adopts_runner_installed_cache_dit_backend():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    backend = object()
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._cache_dit_runtime = SimpleNamespace(
        adopt=lambda adopted, *, installation_key: setattr(
            pipeline,
            "adopted",
            (adopted, installation_key),
        )
    )

    pipeline.adopt_cache_dit_backend(backend)

    assert pipeline.adopted == (backend, "minimax_h3.generic")
