# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for diffusion engine and runner selection."""

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from vllm.utils.import_utils import resolve_obj_by_qualname

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.engine.stage_init_utils import build_diffusion_config, extract_stage_metadata_from_omni_stage_config
from vllm_omni.experimental.ar_diffusion.engine import (
    AR_DIFFUSION_MODEL_RUNNER_CLS,
    ARDiffusionEngine,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _cfg(backend):
    return SimpleNamespace(engine_backend=backend)


def test_default_resolves_to_diffusion_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("default")) is DiffusionEngine


def test_subclass_type_is_returned():
    assert DiffusionEngine.resolve_engine_class(_cfg(ARDiffusionEngine)) is ARDiffusionEngine


def test_ar_diffusion_qualname_resolves():
    # Legacy deploy configs used an engine import path.
    cls = DiffusionEngine.resolve_engine_class(_cfg("vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"))
    assert cls is ARDiffusionEngine


def test_non_engine_type_raises():
    with pytest.raises(TypeError):
        DiffusionEngine.resolve_engine_class(_cfg(dict))


def test_qualname_not_an_engine_raises():
    with pytest.raises(TypeError):
        DiffusionEngine.resolve_engine_class(_cfg("builtins.dict"))


def test_bad_qualname_raises():
    with pytest.raises(ValueError):
        DiffusionEngine.resolve_engine_class(_cfg("not.a.real.module.NoSuchEngine"))


def test_ar_diffusion_engine_is_diffusion_engine_subclass():
    assert issubclass(ARDiffusionEngine, DiffusionEngine)


def test_config_engine_backend_field_defaults_to_default():
    field = {f.name: f for f in fields(OmniDiffusionConfig)}["engine_backend"]
    assert field.default == "default"


# --- AR-Diffusion worker/runner wiring -----------------------------------------------


def test_ar_diffusion_model_runner_cls_resolves():
    cls = resolve_obj_by_qualname(AR_DIFFUSION_MODEL_RUNNER_CLS)
    assert cls.__name__ == "ARDiffusionModelRunner"


def test_engine_class_declares_its_runner():
    """Routing lives on the engine class (review: hsliuustc0106) — engines
    never mutate od_config; the worker resolves the runner off the class."""
    assert ARDiffusionEngine.default_diffusion_model_runner_cls == AR_DIFFUSION_MODEL_RUNNER_CLS
    assert DiffusionEngine.default_diffusion_model_runner_cls is None


_AR_ENGINE = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
_DEFAULT_RUNNER = "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner"

_AR_DEPLOYS = [
    ("abot_world_stepwise", True),
    ("causvid_wan_latest_kv", False),
    ("causvid_wan_self_forcing", False),
    ("dreamzero", False),
    ("dreamzero_tp1_cfg2", False),
    ("hy_nomas_ar", True),
    ("hy_nomas_ar_recent", True),
    ("lingbot_world_v2_stepwise", True),
    ("physis_h3", False),
    ("physis_ltx", False),
    ("physis_wan", False),
    ("taomate_h3_usp2_realtime", True),
    ("taomate_h3_usp4_full_request", False),
    ("taomate_h3_usp4_realtime", True),
]


@pytest.fixture
def resolve_deploy(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import REQUIRED_FILES
    from vllm_omni.platforms import current_omni_platform

    # Use the real resolver with a local model layout and simulated device
    # inventory so multi-device deployments also resolve on a CPU CI agent.
    for relative in REQUIRED_FILES:
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{"_class_name": "HyNomasModularPipeline"}' if relative == "modular_model_index.json" else "{}")
    if current_omni_platform.device_control_env_var:
        monkeypatch.delenv(current_omni_platform.device_control_env_var, raising=False)
    monkeypatch.setattr(current_omni_platform, "get_device_count", lambda: 16)

    def resolve(deploy_path: str | Path) -> OmniDiffusionConfig:
        model = str(tmp_path)
        resolution = resolve_omni_config(
            model,
            trust_remote_code=False,
            deploy_config_path=str(deploy_path),
            cli_overrides=None,
            stage_overrides=None,
            strategy_config_path=None,
        )
        [stage] = resolution.stage_configs
        metadata = extract_stage_metadata_from_omni_stage_config(stage)
        return build_diffusion_config(model, stage, metadata)

    return resolve


@pytest.mark.parametrize(("name", "step_execution"), _AR_DEPLOYS, ids=[name for name, _ in _AR_DEPLOYS])
def test_shipped_deploy_resolves_default_engine_with_explicit_ar_runner(resolve_deploy, name, step_execution):
    config = resolve_deploy(get_deploy_config_path(f"{name}.yaml"))
    assert config.diffusion_model_runner_cls == AR_DIFFUSION_MODEL_RUNNER_CLS
    assert DiffusionEngine.resolve_engine_class(config) is DiffusionEngine
    assert DiffusionEngine.resolve_model_runner_class(config) is ARDiffusionModelRunner
    assert config.step_execution is step_execution
    engine = object.__new__(DiffusionEngine)
    expected_mode = DiffusionExecutionMode.STEP_BATCH if step_execution else DiffusionExecutionMode.REQUEST_BATCH
    assert engine._resolve_execution_mode(config) is expected_mode


@pytest.mark.parametrize(
    ("runner_override", "expected_runner"),
    [(None, ARDiffusionModelRunner), (_DEFAULT_RUNNER, DiffusionModelRunner)],
    ids=["legacy", "explicit-override"],
)
def test_legacy_deploy_runner_selection(tmp_path, resolve_deploy, runner_override, expected_runner):
    raw = yaml.safe_load(Path(get_deploy_config_path("abot_world_stepwise.yaml")).read_text())
    [stage] = raw["stages"]
    stage["engine_backend"] = _AR_ENGINE
    stage.pop("diffusion_model_runner_cls")
    if runner_override:
        stage["diffusion_model_runner_cls"] = runner_override
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(yaml.safe_dump(raw))

    config = resolve_deploy(legacy)
    assert DiffusionEngine.resolve_engine_class(config) is ARDiffusionEngine
    assert DiffusionEngine.resolve_model_runner_class(config) is expected_runner


def test_default_engine_uses_platform_runner(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "get_diffusion_model_runner_cls", lambda: _DEFAULT_RUNNER)
    assert DiffusionEngine.resolve_model_runner_class(OmniDiffusionConfig()) is DiffusionModelRunner


@pytest.mark.parametrize("backend", ["default", _AR_ENGINE])
def test_dreamzero_rejects_non_ar_effective_runner(backend):
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    config = OmniDiffusionConfig(engine_backend=backend, diffusion_model_runner_cls=_DEFAULT_RUNNER)
    with pytest.raises(ValueError, match="requires ARDiffusionModelRunner"):
        DreamZeroPipeline(od_config=config)


def test_runner_rejects_multi_sequence_config():
    """Review (hsliuustc0106): the paged path writes batch index 0 only, so
    max_num_seqs > 1 must be rejected at init."""
    # Non-cell pipeline stub: exercise the paged max_num_seqs=1 guard.
    fake = SimpleNamespace(od_config=SimpleNamespace(max_num_seqs=4), pipeline=object())
    with pytest.raises(ValueError, match="max_num_seqs=1"):
        ARDiffusionModelRunner._preallocate_kv_cache(fake)


def test_execute_model_accepts_kv_prefetch_job_kwarg():
    """Regression: the worker passes ``kv_prefetch_job`` to every runner
    (added with the KV prefetch path, #4448); the AR-Diffusion override must
    accept and forward it or DreamZero crashes on the warm-up dummy run."""
    import inspect

    for cls in (DiffusionModelRunner, ARDiffusionModelRunner):
        params = inspect.signature(cls.execute_model).parameters
        assert "kv_prefetch_job" in params, cls
