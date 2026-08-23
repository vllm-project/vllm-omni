# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.config.omni_config import _DiffusionConfigProjection
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _fixed_master_port(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def test_dense_legacy_is_default() -> None:
    config = OmniDiffusionConfig.from_kwargs()

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.DENSE_LEGACY


def test_paged_worker_local_is_rejected_until_implemented() -> None:
    with pytest.raises(ValueError, match="reserved but not implemented"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="paged_worker_local")


def test_unknown_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Diffusion KV diffusion_kv_mode"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="unknown")


def test_non_mapping_omni_kv_config_is_rejected() -> None:
    with pytest.raises(TypeError, match="omni_kv_config must be a mapping"):
        OmniDiffusionConfig.from_kwargs(omni_kv_config="paged_scheduler")


def test_paged_scheduler_rejects_dense_legacy_kv_receive() -> None:
    with pytest.raises(ValueError, match="does not support imported AR KV"):
        OmniDiffusionConfig.from_kwargs(
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_rows_per_request=1,
            max_num_batched_tokens=1,
            omni_kv_config={"need_recv_cache": True},
        )


def test_paged_scheduler_does_not_depend_on_model_registry() -> None:
    config = OmniDiffusionConfig.from_kwargs(
        model_class_name="FutureDiffusionModel",
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        max_num_batched_tokens=1,
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_paged_scheduler_mode_is_platform_agnostic(config_cls) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def test_paged_scheduler_requires_a_worker_row_limit() -> None:
    with pytest.raises(ValueError, match="requires diffusion_kv_max_rows_per_request"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="paged_scheduler")
    with pytest.raises(ValueError, match="requires diffusion_kv_max_rows_per_request"):
        _DiffusionConfigProjection.from_kwargs(diffusion_kv_mode="paged_scheduler")


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_worker_row_limit_must_be_a_positive_integer(invalid_limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_max_rows_per_request=invalid_limit)
    with pytest.raises((TypeError, ValueError)):
        _DiffusionConfigProjection.from_kwargs(diffusion_kv_max_rows_per_request=invalid_limit)
