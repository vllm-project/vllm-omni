# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Issue #8816: preserve explicit-vs-serialized FT configuration semantics."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.config import FaultToleranceConfig

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.stage_utils import _to_dict

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
DEFAULT_TIMEOUT = FaultToleranceConfig().engine_recovery_timeout_sec


def _stage(engine_args=None):
    return SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=engine_args if engine_args is not None else {},
        runtime=SimpleNamespace(devices=None),
    )


def _ft_transport_args(args):
    """Exercise the FT transport without #8503's unrelated compilation round-trip."""
    serialized = _to_dict(args)
    keys = ("enable_fault_tolerance", "fault_tolerance_config", "_omni_ft_config_was_typed")
    return {key: serialized[key] for key in keys if key in serialized}


def _build_and_capture(engine_args):
    stage = _stage()
    fake_config = SimpleNamespace(quant_config=None)
    sentinel_executor = object()
    captured = {}

    def create_config(args, *unused_args, **unused_kwargs):
        captured["enabled"] = args.enable_fault_tolerance
        captured["ft_config"] = args.fault_tolerance_config
        return fake_config

    with (
        mock.patch.object(
            stage_init_utils.OmniEngineArgs,
            "create_engine_config",
            autospec=True,
            side_effect=create_config,
        ),
        mock.patch.object(
            stage_init_utils.Executor,
            "get_class",
            return_value=sentinel_executor,
        ),
        mock.patch.object(
            stage_init_utils.OmniINCConfig,
            "maybe_upgrade",
            return_value=None,
        ),
    ):
        config, executor = stage_init_utils.build_vllm_config(
            stage,
            model="dummy-model",
            engine_args_dict=engine_args,
        )

    assert config is fake_config
    assert executor is sentinel_executor
    return captured


@pytest.mark.parametrize(
    ("case", "expected_enabled", "expected_timeout"),
    [
        ("serialized_default", False, DEFAULT_TIMEOUT),
        ("serialized_disabled_custom_typed", False, 180),
        ("serialized_explicit_enable", True, 180),
        ("explicit_config_only", True, 180),
        ("explicit_config_only_default", True, DEFAULT_TIMEOUT),
        ("explicit_config_with_false", True, 180),
        ("explicit_config_default_with_false", True, DEFAULT_TIMEOUT),
        ("explicit_enable", True, 180),
        ("explicit_disable", False, DEFAULT_TIMEOUT),
        ("no_ft_fields", False, DEFAULT_TIMEOUT),
    ],
)
def test_stage_reconstruction_preserves_ft_intent(case, expected_enabled, expected_timeout):
    if case == "serialized_default":
        engine_args = _ft_transport_args(OmniEngineArgs(model="dummy-model"))
    elif case == "serialized_disabled_custom_typed":
        engine_args = _ft_transport_args(
            OmniEngineArgs(
                model="dummy-model",
                fault_tolerance_config=FaultToleranceConfig(engine_recovery_timeout_sec=180),
            )
        )
    elif case == "serialized_explicit_enable":
        engine_args = _ft_transport_args(
            OmniEngineArgs(
                model="dummy-model",
                enable_fault_tolerance=True,
                fault_tolerance_config={"engine_recovery_timeout_sec": 180},
            )
        )
    elif case == "explicit_config_only":
        engine_args = {"fault_tolerance_config": {"engine_recovery_timeout_sec": 180}}
    elif case == "explicit_config_only_default":
        engine_args = {"fault_tolerance_config": {"engine_recovery_timeout_sec": DEFAULT_TIMEOUT}}
    elif case == "explicit_config_with_false":
        engine_args = {
            "enable_fault_tolerance": False,
            "fault_tolerance_config": {"engine_recovery_timeout_sec": 180},
        }
    elif case == "explicit_config_default_with_false":
        engine_args = {
            "enable_fault_tolerance": False,
            "fault_tolerance_config": {"engine_recovery_timeout_sec": DEFAULT_TIMEOUT},
        }
    elif case == "explicit_enable":
        engine_args = {
            "enable_fault_tolerance": True,
            "fault_tolerance_config": {"engine_recovery_timeout_sec": 180},
        }
    elif case == "explicit_disable":
        engine_args = {"enable_fault_tolerance": False}
    else:
        engine_args = {}

    captured = _build_and_capture(engine_args)
    assert captured["enabled"] is expected_enabled
    assert isinstance(captured["ft_config"], FaultToleranceConfig)
    assert captured["ft_config"].engine_recovery_timeout_sec == expected_timeout


def test_to_dict_marks_only_typed_ft_config():
    untouched = OmniEngineArgs(model="dummy-model")
    serialized = _to_dict(untouched)
    assert serialized["_omni_ft_config_was_typed"] is True
    assert isinstance(serialized["fault_tolerance_config"], dict)
    assert untouched.enable_fault_tolerance is False
    assert isinstance(untouched.fault_tolerance_config, FaultToleranceConfig)

    # A user-provided mapping is an explicit config, even if it equals defaults.
    raw = {
        "enable_fault_tolerance": False,
        "fault_tolerance_config": asdict(FaultToleranceConfig()),
    }
    assert "_omni_ft_config_was_typed" not in _to_dict(raw)


def test_stage_builder_preserves_provenance():
    args = OmniEngineArgs(model="dummy-model")
    stage = _stage(engine_args=args)
    with mock.patch.object(
        stage_init_utils, "_finalize_engine_args_dict", side_effect=lambda engine_args, **kwargs: engine_args
    ):
        projected = stage_init_utils.build_legacy_engine_args_dict(stage, "dummy-model")

    assert projected["_omni_ft_config_was_typed"] is True
    assert projected["enable_fault_tolerance"] is False
    assert isinstance(projected["fault_tolerance_config"], dict)


def test_stage_builder_does_not_mark_explicit_raw_config():
    explicit = {
        "enable_fault_tolerance": False,
        "fault_tolerance_config": asdict(FaultToleranceConfig()),
    }
    stage = _stage(engine_args=explicit)
    with mock.patch.object(
        stage_init_utils, "_finalize_engine_args_dict", side_effect=lambda engine_args, **kwargs: engine_args
    ):
        projected = stage_init_utils.build_legacy_engine_args_dict(stage, "dummy-model")
    assert "_omni_ft_config_was_typed" not in projected
    assert projected["fault_tolerance_config"] == explicit["fault_tolerance_config"]
    assert explicit["enable_fault_tolerance"] is False


def test_build_vllm_config_does_not_mutate_incoming_mapping():
    raw = _ft_transport_args(OmniEngineArgs(model="dummy-model"))
    original = dict(raw)
    _build_and_capture(raw)
    assert raw == original


def test_provenance_survives_msgpack_transport():
    import msgspec

    payload = _ft_transport_args(OmniEngineArgs(model="dummy-model"))
    received = msgspec.msgpack.decode(msgspec.msgpack.encode(payload))
    assert received["_omni_ft_config_was_typed"] is True
    assert _build_and_capture(received)["enabled"] is False


def test_structured_stage_does_not_project_implicit_ft_defaults():
    from vllm_omni.config.omni_config import OmniStageParallelConfig

    config = OmniStageParallelConfig()
    projected = stage_init_utils._project_upstream_config_fields(
        config, stage_init_utils._PARALLEL_CONFIG_ENGINE_FIELD_MAP
    )
    assert "fault_tolerance_config" not in projected
    assert "_omni_ft_config_was_typed" not in projected
