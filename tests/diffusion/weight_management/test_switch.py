# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selector propagation, argument admission and process ownership (RFC 01-2)."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.weight_management.errors import ConfigError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def clean_prompt_cache_environment(monkeypatch):
    monkeypatch.delenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", raising=False)
    monkeypatch.delenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", raising=False)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("diffusion_offload_config", {}),
        ("enable_cpu_offload", True),
        ("enable_layerwise_offload", True),
        ("enable_distributed_layerwise_offload", True),
        ("dlo_host_registration_limit_gib", 1.0),
        ("host_weight_runtime_mode", "preferred"),
        ("host_weight_runtime_root", "/tmp/host-weights"),
    ],
)
def test_unified_rejects_legacy_offload_settings(setting, value):
    with pytest.raises(ConfigError, match=setting) as error:
        OmniDiffusionConfig(diffusion_weight_stack="unified", **{setting: value})
    assert "--diffusion-weight-stack legacy" in str(error.value)


@pytest.mark.parametrize("projected", [False, True], ids=["direct", "projection"])
@pytest.mark.parametrize(("limit", "accepted"), [(0, True), ("0", True), (1, False), ("1", False)])
def test_unified_normalizes_registration_limit_before_admission(projected, limit, accepted):
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    factory = _DiffusionConfigProjection.from_kwargs if projected else OmniDiffusionConfig
    options = {"diffusion_weight_stack": "unified", "dlo_host_registration_limit_gib": limit}
    if accepted:
        config = factory(**options)
        assert config.dlo_host_registration_limit_gib == 0.0
    else:
        with pytest.raises(ConfigError, match="dlo_host_registration_limit_gib"):
            factory(**options)


@pytest.mark.parametrize("setting", ["OMNI_DIFFUSION_PROMPT_EMBED_CACHE", "OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE"])
@pytest.mark.parametrize("value", ["", "0", "1"])
def test_unified_rejects_prompt_cache_environment(monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(ConfigError, match=setting):
        OmniDiffusionConfig(diffusion_weight_stack="unified")


@pytest.mark.parametrize("value", [None, "legacy", "unified"])
def test_parse_and_config_defaults(value):
    from vllm_omni.diffusion.weight_management.startup import WeightStack, is_unified, parse_weight_stack

    expected = WeightStack.UNIFIED if value == "unified" else WeightStack.LEGACY
    assert parse_weight_stack(value) is expected
    config = OmniDiffusionConfig(diffusion_weight_stack=value)
    assert config.diffusion_weight_stack == expected.value
    assert is_unified(config) is (expected is WeightStack.UNIFIED)
    assert not is_unified(object())


@pytest.mark.parametrize("value", ["", "invalid", "UNIFIED", 1, True, [], {}])
def test_invalid_selector(value):
    from vllm_omni.diffusion.weight_management.startup import parse_weight_stack

    with pytest.raises(ConfigError, match="diffusion_weight_stack"):
        parse_weight_stack(value)
    with pytest.raises(ConfigError, match="diffusion_weight_stack"):
        OmniDiffusionConfig(diffusion_weight_stack=value)


def test_process_owners_share_value_and_reject_conflicts():
    from vllm_omni.diffusion.weight_management.startup import (
        WeightStack,
        clear_process_weight_stack,
        process_weight_stack,
        set_process_weight_stack,
    )

    first, second, conflicting = [SimpleNamespace(stage_id=index) for index in range(3)]
    try:
        assert process_weight_stack() is WeightStack.LEGACY
        set_process_weight_stack(WeightStack.UNIFIED, owner=first)
        set_process_weight_stack(WeightStack.UNIFIED, owner=first)
        set_process_weight_stack(WeightStack.UNIFIED, owner=second)
        with pytest.raises(ConfigError) as error:
            set_process_weight_stack(WeightStack.LEGACY, owner=conflicting)
        assert "stage 0" in str(error.value) and "stage 2" in str(error.value)
        assert "legacy" in str(error.value) and "unified" in str(error.value)
        clear_process_weight_stack(owner=conflicting)
        clear_process_weight_stack(owner=first)
        clear_process_weight_stack(owner=first)
        assert process_weight_stack() is WeightStack.UNIFIED
        clear_process_weight_stack(owner=second)
        assert process_weight_stack() is WeightStack.LEGACY
        set_process_weight_stack(WeightStack.LEGACY, owner=first)
        with pytest.raises(ConfigError):
            set_process_weight_stack(WeightStack.UNIFIED, owner=conflicting)
    finally:
        for owner in (first, second, conflicting):
            clear_process_weight_stack(owner=owner)


@pytest.mark.parametrize("value", [None, "legacy", "unified"])
def test_python_propagation(value):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.engine.arg_utils import OrchestratorArgs

    args = OrchestratorArgs(diffusion_weight_stack=value)
    stage = StageConfigFactory.create_default_diffusion(
        {"model": "test-model", "diffusion_weight_stack": args.diffusion_weight_stack}
    )[0]
    config = OmniDiffusionConfig.from_kwargs(**dict(stage["engine_args"]))
    assert config.diffusion_weight_stack == (value or "legacy")


@pytest.mark.parametrize("value", [None, "legacy", "unified"])
def test_cli_propagation(value):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers(dest="command"))
    argv = ["serve", "test-model", "--omni"]
    if value is not None:
        argv += ["--diffusion-weight-stack", value]
    args = parser.parse_args(argv)
    assert args.diffusion_weight_stack == value
    stage = StageConfigFactory.create_default_diffusion(args.get_explicit_kwargs_dict())[0]
    config = OmniDiffusionConfig.from_kwargs(**dict(stage["engine_args"]))
    assert config.diffusion_weight_stack == (value or "legacy")


@pytest.mark.parametrize("value", ["legacy", "unified"])
def test_yaml_propagation_and_cli_precedence(tmp_path, value):
    from vllm_omni.config.omni_config import (
        VllmOmniConfig,
        VllmOmniDiffusionStageConfig,
        extract_diffusion_stage_config_kwargs,
    )
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config

    deploy = tmp_path / "weight-stack.yaml"
    deploy.write_text(f"pipeline: dreamzero\nstages:\n  - stage_id: 0\n    diffusion_weight_stack: {value}\n")
    pipeline = resolve_pipeline_config("dreamzero")
    config = VllmOmniConfig.from_pipeline_config(pipeline, deploy_config_path=str(deploy))
    stage = config.stage_by_id(0)
    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.diffusion_config.diffusion_weight_stack == value
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config

    engine_args = build_engine_args_dict_from_omni_stage_config(stage, model="test-model")
    diffusion_kwargs = extract_diffusion_stage_config_kwargs(
        engine_args, stage_id=stage.stage_id, include_engine_adapter_metadata=True
    )
    assert OmniDiffusionConfig.from_kwargs(**diffusion_kwargs).diffusion_weight_stack == value
    override = "legacy" if value == "unified" else "unified"
    config = VllmOmniConfig.from_pipeline_config(
        pipeline, deploy_config_path=str(deploy), cli_overrides={"diffusion_weight_stack": override}
    )
    stage = config.stage_by_id(0)
    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.diffusion_config.diffusion_weight_stack == override


def test_unified_rejects_legacy_policy_config_from_kwargs():
    with pytest.raises(ConfigError, match="diffusion_offload_config"):
        OmniDiffusionConfig.from_kwargs(diffusion_weight_stack="unified", diffusion_offload_config={})


@pytest.fixture
def worker_environment(monkeypatch):
    from vllm_omni.diffusion.weight_management.startup import process_weight_stack
    from vllm_omni.diffusion.worker import diffusion_worker as module

    seen = []

    def runner(**kwargs):
        seen.append(process_weight_stack())
        return SimpleNamespace(offload_backend=None, kv_transfer_manager=None)

    monkeypatch.setattr(module.DiffusionWorker, "init_device", lambda self: None)
    monkeypatch.setattr(module.DiffusionWorker, "_create_profiler", lambda self: None)
    monkeypatch.setattr(module.DiffusionWorker, "init_lora_manager", lambda self: None)
    monkeypatch.setattr(module, "resolve_obj_by_qualname", lambda name: runner)
    monkeypatch.setattr(module, "destroy_distributed_env", lambda: None)
    return module, seen


def _worker_config(stack="unified"):
    return OmniDiffusionConfig(
        stage_id=1,
        diffusion_weight_stack=stack,
        diffusion_model_runner_cls="test.Runner",
        diffusion_load_format="default",
    )


@pytest.mark.parametrize("stack", ["legacy", "unified"])
def test_worker_sets_selector_before_runner_and_clears_on_shutdown(worker_environment, stack):
    from vllm_omni.diffusion.weight_management.startup import WeightStack, process_weight_stack

    module, seen = worker_environment
    worker = module.DiffusionWorker(0, 0, _worker_config(stack), skip_load_model=True)
    try:
        assert seen == [WeightStack(stack)]
        assert process_weight_stack() is WeightStack(stack)
    finally:
        worker.shutdown()
    worker.shutdown()
    assert process_weight_stack() is WeightStack.LEGACY


@pytest.mark.parametrize("phase", ["init_device", "runner", "load_model"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_worker_initialization_releases_selector(worker_environment, monkeypatch, phase, error_type):
    from vllm_omni.diffusion.weight_management.startup import WeightStack, process_weight_stack

    module, _ = worker_environment
    original = error_type("initialization failed")

    def fail(*args, **kwargs):
        assert process_weight_stack() is WeightStack.UNIFIED
        raise original

    if phase == "runner":
        monkeypatch.setattr(module, "resolve_obj_by_qualname", fail)
    else:
        monkeypatch.setattr(module.DiffusionWorker, phase, fail)
    with pytest.raises(error_type) as error:
        module.DiffusionWorker(0, 0, _worker_config())
    assert error.value is original
    assert process_weight_stack() is WeightStack.LEGACY


@pytest.mark.parametrize("phase", ["offload", "distributed", "lora"])
def test_shutdown_failure_still_releases_selector(worker_environment, monkeypatch, phase):
    from vllm_omni.diffusion.weight_management.startup import WeightStack, process_weight_stack

    module, _ = worker_environment
    worker = module.DiffusionWorker(0, 0, _worker_config(), skip_load_model=True)

    error_type = RuntimeError if phase == "lora" else KeyboardInterrupt

    def fail():
        raise error_type("cleanup failed")

    if phase == "offload":
        worker.model_runner.offload_backend = SimpleNamespace(shutdown=fail)
    elif phase == "lora":
        worker.lora_manager = SimpleNamespace(close=fail)
    else:
        monkeypatch.setattr(module, "destroy_distributed_env", fail)
    with pytest.raises(error_type, match="cleanup failed"):
        worker.shutdown()
    assert process_weight_stack() is WeightStack.LEGACY


@pytest.mark.parametrize(
    "setting", ["enable_cpu_offload", "enable_layerwise_offload", "enable_distributed_layerwise_offload"]
)
def test_legacy_keeps_offload_flags_and_accepts_prompt_cache_env(monkeypatch, setting):
    monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", "1")
    config = OmniDiffusionConfig(**{setting: True})
    assert config.diffusion_weight_stack == "legacy"
    assert getattr(config, setting) is True


def test_failed_worker_does_not_clear_another_owner(worker_environment, monkeypatch):
    from vllm_omni.diffusion.weight_management.startup import WeightStack, process_weight_stack

    module, _ = worker_environment
    first = module.DiffusionWorker(0, 0, _worker_config(), skip_load_model=True)
    destroyed = []
    monkeypatch.setattr(module, "destroy_distributed_env", lambda: destroyed.append(True))

    def unexpected_init_device(self):
        pytest.fail("a conflicting worker must be rejected before device initialization")

    monkeypatch.setattr(module.DiffusionWorker, "init_device", unexpected_init_device)
    try:
        with pytest.raises(ConfigError):
            module.DiffusionWorker(0, 0, _worker_config("legacy"), skip_load_model=True)
        assert destroyed == []
        assert process_weight_stack() is WeightStack.UNIFIED
    finally:
        first.shutdown()
    assert process_weight_stack() is WeightStack.LEGACY


def test_unified_skips_legacy_flag_materialization(monkeypatch):
    from vllm_omni.diffusion.offloader import config as offload_config

    def fail(config):
        raise AssertionError("unified must not materialize legacy offload flags")

    monkeypatch.setattr(offload_config, "materialize_legacy_offload_flags", fail)
    config = OmniDiffusionConfig(diffusion_weight_stack="unified", diffusion_compile_granularity="full")
    assert config.diffusion_compile_granularity == "full"
    assert config.enable_cpu_offload is False


def test_yaml_rejects_legacy_policy_before_parsing(tmp_path):
    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config

    deploy = tmp_path / "unified-offload.yaml"
    deploy.write_text(
        "pipeline: dreamzero\nstages:\n  - stage_id: 0\n    diffusion_weight_stack: unified\n    diffusion_offload_config: {}\n"
    )
    with pytest.raises(ConfigError, match="diffusion_offload_config"):
        VllmOmniConfig.from_pipeline_config(resolve_pipeline_config("dreamzero"), deploy_config_path=str(deploy))


def test_package_root_exports_weight_stack():
    from vllm_omni.diffusion.weight_management import WeightStack

    assert WeightStack.LEGACY.value == "legacy"
    assert WeightStack.UNIFIED.value == "unified"


def test_projection_normalizes_boolean_strings_before_unified_admission():
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    accepted = _DiffusionConfigProjection.from_kwargs(diffusion_weight_stack="unified", enable_cpu_offload="false")
    assert accepted.enable_cpu_offload is False
    with pytest.raises(ConfigError, match="enable_cpu_offload"):
        _DiffusionConfigProjection.from_kwargs(diffusion_weight_stack="unified", enable_cpu_offload="true")


def test_failed_second_worker_in_init_device_preserves_first_worker_groups(worker_environment, monkeypatch):
    from vllm_omni.diffusion.weight_management.startup import WeightStack, process_weight_stack

    module, _ = worker_environment
    first = module.DiffusionWorker(0, 0, _worker_config("unified"), skip_load_model=True)
    destroyed = []
    monkeypatch.setattr(module, "destroy_distributed_env", lambda: destroyed.append(True))

    def fail_init_device(self):
        raise RuntimeError("model parallel state must be empty")

    monkeypatch.setattr(module.DiffusionWorker, "init_device", fail_init_device)
    try:
        with pytest.raises(RuntimeError, match="model parallel state must be empty"):
            module.DiffusionWorker(0, 0, _worker_config("unified"), skip_load_model=True)
        assert destroyed == []
        assert process_weight_stack() is WeightStack.UNIFIED
    finally:
        first.shutdown()
    assert process_weight_stack() is WeightStack.LEGACY


def test_unclosed_owner_released_on_gc():
    import gc

    from vllm_omni.diffusion.weight_management.startup import (
        WeightStack,
        process_weight_stack,
        set_process_weight_stack,
    )

    class _Owner:
        stage_id = 7

    owner = _Owner()
    set_process_weight_stack(WeightStack.UNIFIED, owner=owner)
    assert process_weight_stack() is WeightStack.UNIFIED
    del owner
    gc.collect()
    assert process_weight_stack() is WeightStack.LEGACY
