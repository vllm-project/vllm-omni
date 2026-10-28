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


@pytest.mark.parametrize("use_fallback", (False, True))
@pytest.mark.parametrize(
    ("factory_function", "module_name", "class_name"),
    (
        ("_create_shm_connector", "shm_connector", "SharedMemoryConnector"),
        ("_create_mooncake_store_connector", "mooncake_store_connector", "MooncakeStoreConnector"),
        ("_create_yuanrong_connector", "yuanrong_connector", "YuanrongConnector"),
        (
            "_create_mooncake_transfer_engine_connector",
            "mooncake_transfer_engine_connector",
            "MooncakeTransferEngineConnector",
        ),
        ("_create_mori_transfer_engine_connector", "mori_transfer_engine_connector", "MoriTransferEngineConnector"),
        ("_create_nixl_connector", "nixl_connector", "NixlConnector"),
    ),
)
def test_factory_preserves_lazy_backend_import_paths(
    factory_function, module_name, class_name, use_fallback, monkeypatch, mocker
):
    import builtins
    import sys
    from types import ModuleType

    from vllm_omni.distributed.omni_connectors import factory

    config = {"host": "127.0.0.1"}
    constructor = mocker.Mock(return_value=object())
    backend = ModuleType(module_name)
    setattr(backend, class_name, constructor)
    package = ModuleType("connectors")
    setattr(package, module_name, backend)
    original_import = builtins.__import__
    attempts = []

    def import_backend(name, globals=None, locals=None, fromlist=(), level=0):
        if globals and globals.get("__name__") == factory.__name__ and module_name in (fromlist or ()):
            attempts.append((name, level))
            if level == 1 and use_fallback:
                raise ImportError("exercise the existing fallback")
            return package
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_backend)
    monkeypatch.setattr(sys, "path", list(sys.path))
    result = getattr(factory, factory_function)(config)
    assert result is constructor.return_value
    constructor.assert_called_once_with(config)
    assert attempts == ([("connectors", 1), ("omni_connectors.connectors", 0)] if use_fallback else [("connectors", 1)])


def test_factory_fallback_preserves_exported_types(monkeypatch):
    import builtins
    import importlib.util
    import sys
    from types import ModuleType

    from vllm_omni.distributed.omni_connectors import factory
    from vllm_omni.distributed.omni_connectors.connectors import base
    from vllm_omni.distributed.omni_connectors.utils import config

    module_name = f"{factory.__package__}._factory_import_probe"
    spec = importlib.util.spec_from_file_location(module_name, factory.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    packages = {"connectors": ModuleType("connectors"), "utils": ModuleType("utils")}
    setattr(packages["connectors"], "base", base)
    setattr(packages["utils"], "config", config)
    original_import = builtins.__import__
    fallbacks = []

    def import_types(name, globals=None, locals=None, fromlist=(), level=0):
        if globals and globals.get("__name__") == module_name:
            if level == 1 and (name, fromlist) in (("connectors", ("base",)), ("utils", ("config",))):
                raise ImportError("exercise the existing type-import fallback")
            if name in ("omni_connectors.connectors", "omni_connectors.utils"):
                fallbacks.append(name)
                return packages[name.rsplit(".", 1)[1]]
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_types)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(probe)
    assert probe.OmniConnectorBase is base.OmniConnectorBase
    assert probe.ConnectorSpec is config.ConnectorSpec
    assert fallbacks == ["omni_connectors.connectors", "omni_connectors.utils"]
