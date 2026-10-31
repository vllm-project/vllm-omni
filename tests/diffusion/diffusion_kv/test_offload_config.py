# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Config validation for diffusion KV page offload (RFC #9192 phase 1)."""

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_PAGED_KWARGS = {
    "model": "x",
    "diffusion_kv_mode": "paged_scheduler",
    "diffusion_kv_max_rows_per_request": 2,
    "enable_prefix_caching": True,
}


def test_page_offload_enabled_with_paged_prefix_caching(monkeypatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)
    config = OmniDiffusionConfig(**_PAGED_KWARGS, enable_diffusion_kv_page_offload=True)

    assert config.enable_diffusion_kv_page_offload is True
    assert config.diffusion_kv_offload_max_bytes is None


def test_page_offload_rejects_dense_legacy_mode(monkeypatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)

    with pytest.raises(ValueError, match="enable_diffusion_kv_page_offload=True requires diffusion_kv_mode"):
        OmniDiffusionConfig(
            model="x",
            diffusion_kv_max_rows_per_request=2,
            enable_prefix_caching=True,
            enable_diffusion_kv_page_offload=True,
        )


def test_page_offload_rejects_prefix_caching_off(monkeypatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)

    with pytest.raises(ValueError, match="enable_diffusion_kv_page_offload=True requires enable_prefix_caching"):
        OmniDiffusionConfig(
            model="x",
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_rows_per_request=2,
            enable_diffusion_kv_page_offload=True,
        )


def test_page_offload_rejects_sleep_mode(monkeypatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)

    with pytest.raises(ValueError, match="Diffusion KV page offload cannot be combined with sleep mode"):
        OmniDiffusionConfig(**_PAGED_KWARGS, enable_diffusion_kv_page_offload=True, enable_sleep_mode=True)


def test_page_offload_rejects_a_non_boolean_flag(monkeypatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)

    with pytest.raises(TypeError, match="enable_diffusion_kv_page_offload must be a bool"):
        OmniDiffusionConfig(**_PAGED_KWARGS, enable_diffusion_kv_page_offload="true")  # type: ignore[arg-type]


def test_page_offload_rejects_an_invalid_byte_budget(monkeypatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)

    with pytest.raises(ValueError, match="diffusion_kv_offload_max_bytes must be a positive integer"):
        OmniDiffusionConfig(**_PAGED_KWARGS, enable_diffusion_kv_page_offload=True, diffusion_kv_offload_max_bytes=0)
    with pytest.raises(ValueError, match="diffusion_kv_offload_max_bytes must be a positive integer"):
        OmniDiffusionConfig(
            **_PAGED_KWARGS,
            enable_diffusion_kv_page_offload=True,
            diffusion_kv_offload_max_bytes="1024",  # type: ignore[arg-type]
        )
