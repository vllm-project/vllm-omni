# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests that build_engine_args_dict does not mutate its input."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from vllm.config import FaultToleranceConfig

from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.utils import filter_dataclass_kwargs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_config(engine_args: dict | None = None):
    """Build a minimal stage config stub for build_engine_args_dict."""
    if engine_args is None:
        engine_args = {
            "tensor_parallel_size": None,
            "model_subdir": None,
            "tokenizer_subdir": None,
        }
    return SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=engine_args,
        runtime={},
        final_output=True,
        final_output_type=None,
        is_comprehension=False,
        default_sampling_params={},
        default_pooling_params={},
        engine_input_source=[],
        custom_process_input_func=None,
        prompt_transform_func=None,
        prompt_expand_func=None,
        cfg_kv_collect_func=None,
    )


@pytest.fixture(autouse=True)
def _patch_platform(monkeypatch):
    """Stub out platform checks so tests run on any host."""
    monkeypatch.setattr(
        "vllm_omni.engine.stage_init_utils.current_omni_platform",
        SimpleNamespace(
            is_rocm=lambda: False,
            get_omni_ar_worker_cls=lambda: "stub",
            get_omni_generation_worker_cls=lambda: "stub",
            device_control_env_var="CUDA_VISIBLE_DEVICES",
            get_device_count=lambda: 1,
        ),
    )


class TestBuildEngineArgsDictNoMutate:
    """build_engine_args_dict must never modify stage_config.engine_args."""

    def test_original_dict_unchanged_after_call(self):
        engine_args = {
            "tensor_parallel_size": None,
            "omni_kv_config": {"need_send_cache": True},
        }
        original = copy.deepcopy(engine_args)
        stage_cfg = _make_stage_config(engine_args)

        from vllm_omni.engine.stage_init_utils import build_engine_args_dict

        build_engine_args_dict(stage_cfg, model="dummy-model")

        assert stage_cfg.engine_args == original, "build_engine_args_dict mutated stage_config.engine_args"

    def test_returned_dict_drops_none_tp(self):
        engine_args = {"tensor_parallel_size": None}
        stage_cfg = _make_stage_config(engine_args)

        from vllm_omni.engine.stage_init_utils import build_engine_args_dict

        result = build_engine_args_dict(stage_cfg, model="dummy-model")

        assert "tensor_parallel_size" not in result

    def test_returned_dict_keeps_explicit_tp(self):
        engine_args = {"tensor_parallel_size": 4}
        stage_cfg = _make_stage_config(engine_args)

        from vllm_omni.engine.stage_init_utils import build_engine_args_dict

        result = build_engine_args_dict(stage_cfg, model="dummy-model")

        assert result["tensor_parallel_size"] == 4

    def test_idempotency(self):
        engine_args = {
            "tensor_parallel_size": None,
            "omni_kv_config": {"stage_id": 0},
        }
        stage_cfg = _make_stage_config(engine_args)

        from vllm_omni.engine.stage_init_utils import build_engine_args_dict

        result1 = build_engine_args_dict(stage_cfg, model="dummy-model")
        result2 = build_engine_args_dict(stage_cfg, model="dummy-model")

        assert result1 == result2, "Calling build_engine_args_dict twice with the same input produced different results"

    def test_nested_dict_not_shared(self):
        """Mutating the returned dict's nested values must not affect the original."""
        engine_args = {
            "omni_kv_config": {"need_send_cache": True},
        }
        original_kv = copy.deepcopy(engine_args["omni_kv_config"])
        stage_cfg = _make_stage_config(engine_args)

        from vllm_omni.engine.stage_init_utils import build_engine_args_dict

        result = build_engine_args_dict(stage_cfg, model="dummy-model")

        if "omni_kv_config" in result:
            result["omni_kv_config"]["injected_key"] = "injected_value"

        assert stage_cfg.engine_args["omni_kv_config"] == original_kv, (
            "Mutating the returned dict's nested value propagated back to stage_config.engine_args"
        )

    def test_inject_omni_kv_does_not_leak_back(self):
        """inject_omni_kv_connector_config on the result must not pollute stage_config."""
        engine_args = {
            "omni_kv_config": {"need_send_cache": True},
        }
        original = copy.deepcopy(engine_args)
        stage_cfg = _make_stage_config(engine_args)

        from vllm_omni.engine.stage_init_utils import (
            build_engine_args_dict,
            inject_omni_kv_connector_config,
        )

        result = build_engine_args_dict(stage_cfg, model="dummy-model")
        connector = ({"host": "localhost", "port": 1234}, "stage-0", "stage-1")
        inject_omni_kv_connector_config(result, connector, stage_id=0)

        assert stage_cfg.engine_args == original, (
            "inject_omni_kv_connector_config on the returned dict mutated stage_config.engine_args"
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("timeout", [120, 30])
def test_typed_engine_args_preserve_ft_intent_and_config_types(tmp_path, enabled, timeout):
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    source = OmniEngineArgs(
        model=str(tmp_path),
        enable_fault_tolerance=enabled,
        fault_tolerance_config=FaultToleranceConfig(engine_recovery_timeout_sec=timeout),
    )
    stage = _make_stage_config()
    stage.engine_args = source
    projected = build_engine_args_dict(stage, model=str(tmp_path))

    assert isinstance(projected["fault_tolerance_config"], FaultToleranceConfig)
    assert projected["fault_tolerance_config"] is not source.fault_tolerance_config
    assert type(projected["compilation_config"]) is type(source.compilation_config)
    rebuilt = OmniEngineArgs(**filter_dataclass_kwargs(OmniEngineArgs, projected))
    assert rebuilt.enable_fault_tolerance is enabled
    assert rebuilt.fault_tolerance_config.engine_recovery_timeout_sec == timeout

    rebuilt.fault_tolerance_config.engine_recovery_timeout_sec = 60
    assert source.fault_tolerance_config.engine_recovery_timeout_sec == timeout


@pytest.mark.parametrize("timeout", [120, 30])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_raw_ft_config_keeps_native_opt_in(tmp_path, timeout, enabled):
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    raw = {"engine_recovery_timeout_sec": timeout}
    inputs = {"fault_tolerance_config": raw}
    if enabled is not None:
        inputs["enable_fault_tolerance"] = enabled
    source = copy.deepcopy(inputs)
    stage = _make_stage_config(inputs)
    projected = build_engine_args_dict(stage, model=str(tmp_path))
    native = OmniEngineArgs(model=str(tmp_path), **inputs)
    rebuilt = OmniEngineArgs(**filter_dataclass_kwargs(OmniEngineArgs, projected))

    assert rebuilt.enable_fault_tolerance is native.enable_fault_tolerance is True
    assert rebuilt.fault_tolerance_config.engine_recovery_timeout_sec == timeout
    assert inputs == source
