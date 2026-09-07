# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest import mock

import pytest
import torch
from vllm.config import KVTransferConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    parse_kv_transfer_config,
    shutdown_kv_connector,
)
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

KV_TRANSFER_CONFIG = {
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "engine_id": "dit-engine-1",
    "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
}


@pytest.fixture(autouse=True)
def _fixed_master_port(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def test_config_roundtrip_to_kv_vllm_config() -> None:
    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        max_model_len=64,
        kv_transfer_config=dict(KV_TRANSFER_CONFIG),
    )

    assert isinstance(od_config.kv_transfer_config, KVTransferConfig)
    assert od_config.kv_transfer_config.engine_id == "dit-engine-1"
    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    assert vllm_config.kv_transfer_config is od_config.kv_transfer_config


def test_parse_requires_explicit_engine_id() -> None:
    payload = dict(KV_TRANSFER_CONFIG)
    payload.pop("engine_id")
    with pytest.raises(ValueError, match="non-empty engine_id"):
        parse_kv_transfer_config(payload)


def test_diffusion_projection_rejects_missing_engine_id_before_materialize() -> None:
    """Structured resolve must not let upstream auto-mint engine_id first."""
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    payload = dict(KV_TRANSFER_CONFIG)
    payload.pop("engine_id")
    with pytest.raises(ValueError, match="non-empty engine_id"):
        _DiffusionConfigProjection.from_kwargs(kv_transfer_config=payload)


def test_diffusion_projection_preserves_explicit_engine_id() -> None:
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    projection = _DiffusionConfigProjection.from_kwargs(kv_transfer_config=dict(KV_TRANSFER_CONFIG))
    assert isinstance(projection.kv_transfer_config, KVTransferConfig)
    assert projection.kv_transfer_config.engine_id == "dit-engine-1"

    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        kv_transfer_config=projection.kv_transfer_config,
    )
    assert od_config.kv_transfer_config is not None
    assert od_config.kv_transfer_config.engine_id == "dit-engine-1"


class _ConcreteScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


def test_scheduler_assembles_kv_stub_and_shuts_it_down() -> None:
    scheduler = _ConcreteScheduler()
    fake_connector = mock.Mock()
    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        max_model_len=64,
        kv_transfer_config=dict(KV_TRANSFER_CONFIG),
    )
    kv_vllm_config = mock.Mock()
    kv_vllm_config.model_config.max_model_len = 64
    kv_vllm_config.max_in_flight_tokens = 64

    with (
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.kv_connector.KVConnectorFactory.create_connector",
            return_value=fake_connector,
        ),
        mock.patch("vllm_omni.diffusion.sched.base_scheduler.DiffusionKVCacheManager"),
    ):
        scheduler.initialize(
            od_config,
            kv_cache_config=mock.sentinel.kv_cache_config,
            scheduler_block_size=16,
            hash_block_size=16,
            kv_vllm_config=kv_vllm_config,
        )

    assert scheduler.kv_connector is fake_connector
    with mock.patch("vllm_omni.diffusion.diffusion_kv.kv_connector.shutdown_kv_connector") as shutdown:
        scheduler.close()
    shutdown.assert_called_once_with(scheduler_connector=fake_connector)


def test_native_config_requires_paged_scheduler() -> None:
    with pytest.raises(ValueError, match="requires diffusion_kv_mode='paged_scheduler'"):
        OmniDiffusionConfig.from_kwargs(kv_transfer_config=dict(KV_TRANSFER_CONFIG))


def test_kv_and_legacy_transfer_configs_are_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one KV transfer path"):
        OmniDiffusionConfig.from_kwargs(
            kv_transfer_config=dict(KV_TRANSFER_CONFIG),
            omni_kv_config={"need_recv_cache": True},
        )


def test_shutdown_kv_connector_is_idempotent() -> None:
    scheduler_connector = mock.Mock()
    with mock.patch("vllm_omni.diffusion.diffusion_kv.kv_connector.ensure_kv_transfer_shutdown") as ensure_shutdown:
        shutdown_kv_connector(scheduler_connector=scheduler_connector)
        shutdown_kv_connector()

    assert ensure_shutdown.call_count == 2
    scheduler_connector.shutdown.assert_called_once()
