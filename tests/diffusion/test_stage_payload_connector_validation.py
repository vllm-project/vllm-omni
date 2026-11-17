# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject unsupported synchronous payload transports before worker startup."""

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNSUPPORTED_CONNECTORS = (
    "MooncakeStoreConnector",
    "MooncakeConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
)


def _config(connector, **kwargs):
    return OmniDiffusionConfig(
        stage_id=1,
        omni_kv_config={
            "connector_config": {"type": connector},
            "omni_from_stage": "0",
            "omni_to_stage": "1",
        },
        **kwargs,
    )


@pytest.mark.parametrize("connector", UNSUPPORTED_CONNECTORS)
def test_payload_receiver_rejects_unsupported_connector(connector):
    with pytest.raises(ValueError, match=rf"Stage 1: connector '{connector}'.*get_with_deadline"):
        _config(connector, stage_input_payload_keys=("prompt_embeds",))


@pytest.mark.parametrize("connector", ("SharedMemoryConnector", "NixlConnector"))
def test_supported_payload_receiver_does_not_initialize_connector(connector, monkeypatch, mocker):
    constructor = mocker.Mock(side_effect=AssertionError("configuration must not initialize a backend"))
    monkeypatch.setitem(OmniConnectorFactory._registry, connector, constructor)
    config = _config(connector, stage_input_payload_keys=["prompt_embeds"])
    assert config.stage_input_payload_keys == ("prompt_embeds",)
    constructor.assert_not_called()


@pytest.mark.parametrize("connector", UNSUPPORTED_CONNECTORS)
def test_other_transfer_paths_remain_allowed(connector):
    config = _config(connector, stage_output_payload_keys=("prompt_embeds",))
    assert config.stage_output_payload_keys == ("prompt_embeds",)


def test_inline_payload_without_connector_remains_allowed():
    config = OmniDiffusionConfig(stage_input_payload_keys=("prompt_embeds",))
    assert config.omni_kv_config == {}


@pytest.mark.parametrize("supports_deadline_receive", (False, True))
def test_custom_connector_capability_is_explicit(supports_deadline_receive, monkeypatch, mocker):
    monkeypatch.setattr(OmniConnectorFactory, "_registry", dict(OmniConnectorFactory._registry))
    monkeypatch.setattr(
        OmniConnectorFactory, "_deadline_aware_connectors", set(OmniConnectorFactory._deadline_aware_connectors)
    )
    constructor = mocker.Mock(side_effect=AssertionError("must remain lazy"))
    OmniConnectorFactory.register_connector(
        "CustomConnector", constructor, supports_deadline_receive=supports_deadline_receive
    )
    if supports_deadline_receive:
        _config("CustomConnector", stage_input_payload_keys=("prompt_embeds",))
    else:
        with pytest.raises(ValueError, match="CustomConnector.*get_with_deadline"):
            _config("CustomConnector", stage_input_payload_keys=("prompt_embeds",))
    constructor.assert_not_called()


def test_middle_stage_cannot_bypass_validation_by_sending_payloads():
    with pytest.raises(ValueError, match="Stage 1.*MooncakeTransferEngineConnector"):
        OmniDiffusionConfig(
            stage_id=1,
            stage_input_payload_keys=("prompt_embeds",),
            stage_output_payload_keys=("latents",),
            omni_kv_config={
                "connector_config": {"type": "MooncakeTransferEngineConnector", "role": "sender"},
                "omni_from_stage": "1",
                "omni_to_stage": "2",
            },
        )


def test_resolved_wan_stage_fails_before_device_discovery(monkeypatch):
    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.engine import stage_init_utils
    from vllm_omni.entrypoints.utils import inject_omni_kv_config
    from vllm_omni.model_executor.models.wan2_2.pipeline import WAN2_2_EG_PIPELINE

    stage = VllmOmniConfig.from_pipeline_config(WAN2_2_EG_PIPELINE).stage_by_id(1)
    inject_omni_kv_config(stage, {"type": "MooncakeTransferEngineConnector"}, "0", "1")
    metadata = stage_init_utils.extract_stage_metadata_from_omni_stage_config(stage)

    def unexpected_device_discovery():
        pytest.fail("unsupported transport must be rejected before device discovery")

    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", unexpected_device_discovery)
    with pytest.raises(ValueError, match="Stage 1.*MooncakeTransferEngineConnector.*get_with_deadline"):
        stage_init_utils.build_diffusion_config("unused", stage, metadata)


def test_shm_factory_uses_platform_hook(mocker):
    from vllm_omni.distributed.omni_connectors import factory
    from vllm_omni.platforms import current_omni_platform

    constructor = mocker.Mock(return_value=object())
    select = mocker.patch.object(current_omni_platform, "get_shared_memory_connector_cls", return_value=constructor)
    config = {"host": "127.0.0.1"}
    assert factory._create_shm_connector(config) is constructor.return_value
    select.assert_called_once_with()
    constructor.assert_called_once_with(config)


@pytest.mark.parametrize("connector_config", ({}, {"host": "127.0.0.1"}, {"type": None}, {"type": ""}, {"type": 42}))
def test_payload_connector_requires_explicit_type(connector_config):
    with pytest.raises(ValueError, match="Stage 1.*requires.*'type'"):
        OmniDiffusionConfig(
            stage_id=1,
            stage_input_payload_keys=("prompt_embeds",),
            omni_kv_config={"connector_config": connector_config},
        )


def test_unknown_payload_connector_is_not_reported_as_unsupported():
    with pytest.raises(ValueError, match="Unknown connector: UnknownPayloadConnector.*Available:.*NixlConnector"):
        _config("UnknownPayloadConnector", stage_input_payload_keys=("prompt_embeds",))


def test_custom_payload_error_explains_registration():
    with pytest.raises(ValueError, match="supports_deadline_receive=True"):
        _config("MooncakeConnector", stage_input_payload_keys=("prompt_embeds",))


@pytest.mark.parametrize(
    ("name", "module_name", "class_name"),
    (
        ("SharedMemoryConnector", "shm_connector", "SharedMemoryConnector"),
        ("NixlConnector", "nixl_connector", "NixlConnector"),
        ("MooncakeStoreConnector", "mooncake_store_connector", "MooncakeStoreConnector"),
        ("MooncakeConnector", "mooncake_store_connector", "MooncakeStoreConnector"),
        ("MooncakeTransferEngineConnector", "mooncake_transfer_engine_connector", "MooncakeTransferEngineConnector"),
        ("MoriTransferEngineConnector", "mori_transfer_engine_connector", "MoriTransferEngineConnector"),
        ("YuanrongConnector", "yuanrong_connector", "YuanrongConnector"),
        ("YuanrongTransferEngineConnector", "yuanrong_transfer_engine_connector", "YuanrongTransferEngineConnector"),
    ),
)
def test_builtin_deadline_capability_matches_implementation(name, module_name, class_name):
    import importlib

    from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase

    prefix = (
        "vllm_omni.platforms.npu.omni_connectors"
        if name == "YuanrongTransferEngineConnector"
        else "vllm_omni.distributed.omni_connectors.connectors"
    )
    try:
        module = importlib.import_module(f"{prefix}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"mooncake", "mori", "datasystem", "yr", "torch_npu", "nixl"}:
            pytest.skip(f"optional native dependency unavailable: {exc.name}")
        raise
    connector_cls = getattr(module, class_name)
    assert OmniConnectorFactory.supports_deadline_receive(name) == (
        connector_cls.get_with_deadline is not OmniConnectorBase.get_with_deadline
    )


def test_platform_shm_implements_deadline_receive():
    from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
    from vllm_omni.platforms import current_omni_platform

    connector_cls = current_omni_platform.get_shared_memory_connector_cls()
    assert connector_cls.get_with_deadline is not OmniConnectorBase.get_with_deadline
