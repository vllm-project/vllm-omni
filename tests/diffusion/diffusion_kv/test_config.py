# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

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
            omni_kv_config={"need_recv_cache": True},
        )


def test_paged_scheduler_does_not_depend_on_model_registry() -> None:
    config = OmniDiffusionConfig.from_kwargs(
        model_class_name="FutureDiffusionModel",
        diffusion_kv_mode="paged_scheduler",
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER
