# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
@pytest.mark.parametrize("mode", [None, "dense_legacy", DiffusionKVCacheMode.DENSE_LEGACY])
def test_prefix_caching_rejects_non_paged_mode(config_cls, mode) -> None:
    kwargs = {} if mode is None else {"diffusion_kv_mode": mode}
    with pytest.raises(
        ValueError,
        match="set diffusion_kv_mode='paged_scheduler' or disable enable_prefix_caching",
    ):
        config_cls.from_kwargs(enable_prefix_caching=True, **kwargs)


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_prefix_caching_accepts_paged_scheduler(config_cls) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=True,
    )

    assert config.enable_prefix_caching is True
    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


@pytest.mark.parametrize("build_config", [OmniDiffusionConfig, OmniDiffusionConfig.from_kwargs])
def test_prefix_caching_rejects_sleep_mode(build_config) -> None:
    with pytest.raises(ValueError, match="disable enable_prefix_caching or enable_sleep_mode"):
        build_config(
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_rows_per_request=2,
            enable_prefix_caching=True,
            enable_sleep_mode=True,
        )


@pytest.mark.parametrize("mode", ["dense_legacy", "paged_scheduler"])
def test_sleep_mode_accepts_disabled_prefix_caching(mode) -> None:
    config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode=mode,
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=False,
        enable_sleep_mode=True,
    )

    assert config.enable_sleep_mode is True
    assert config.enable_prefix_caching is False


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_native_kv_transfer_requires_prefix_caching_disabled(enable_prefix_caching) -> None:
    kwargs = dict(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=enable_prefix_caching,
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_consumer",
            "engine_id": "test-diffusion",
        },
    )
    if enable_prefix_caching:
        with pytest.raises(ValueError, match="disable enable_prefix_caching for AR KV import"):
            OmniDiffusionConfig.from_kwargs(**kwargs)
    else:
        config = OmniDiffusionConfig.from_kwargs(**kwargs)
        assert config.kv_transfer_config.kv_role == "kv_consumer"


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
@pytest.mark.parametrize("mode", ["dense_legacy", "paged_scheduler"])
def test_disabled_prefix_caching_accepts_both_modes(config_cls, mode) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_mode=mode,
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=False,
    )

    assert config.enable_prefix_caching is False
    assert config.diffusion_kv_mode.value == mode


def test_paged_worker_local_is_rejected_until_implemented() -> None:
    with pytest.raises(ValueError, match="reserved but not implemented"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="paged_worker_local")


def test_unknown_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Diffusion KV diffusion_kv_mode"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="unknown")


def test_non_mapping_omni_kv_config_is_rejected() -> None:
    with pytest.raises(TypeError, match="omni_kv_config must be a mapping"):
        OmniDiffusionConfig.from_kwargs(omni_kv_config="paged_scheduler")


def test_paged_scheduler_rejects_dense_connector_kv_receive() -> None:
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
