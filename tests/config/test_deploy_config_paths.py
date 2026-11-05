# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deploy config lookup precedence for platform plugins and bundled defaults."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm_omni import platforms
from vllm_omni.config import stage_config
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import _get_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def deploy_dirs(tmp_path, monkeypatch):
    platform_dir = tmp_path / "plugin"
    bundled_dir = tmp_path / "bundled"
    cwd = tmp_path / "cwd"
    for directory in (platform_dir, bundled_dir, cwd):
        directory.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(stage_config, "_DEPLOY_DIR", bundled_dir)
    platform = Mock()
    platform.get_default_stage_config_path.return_value = str(platform_dir)
    monkeypatch.setattr(platforms, "current_omni_platform", platform)
    return platform_dir, bundled_dir, cwd


def write_deploy(directory, name="test.yaml", devices="0"):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'stages:\n  - stage_id: 0\n    devices: "{devices}"\n', encoding="utf-8")
    return path


@pytest.mark.parametrize("source", ["plugin", "bundled", "cwd"])
def test_explicit_bare_name_precedence(deploy_dirs, source):
    platform_dir, bundled_dir, cwd = deploy_dirs
    write_deploy(bundled_dir)
    if source in ("plugin", "cwd"):
        write_deploy(platform_dir)
    if source == "cwd":
        write_deploy(cwd)
    expected = {"plugin": platform_dir / "test.yaml", "bundled": bundled_dir / "test.yaml", "cwd": Path("test.yaml")}
    assert stage_config.resolve_deploy_config_path("test.yaml") == expected[source]


@pytest.mark.parametrize("platform_has_config", [True, False])
def test_pipeline_default_ignores_cwd(deploy_dirs, platform_has_config):
    platform_dir, bundled_dir, cwd = deploy_dirs
    write_deploy(cwd)
    write_deploy(bundled_dir)
    if platform_has_config:
        write_deploy(platform_dir)
    expected_dir = platform_dir if platform_has_config else bundled_dir
    assert stage_config.get_default_deploy_config_path("test.yaml") == expected_dir / "test.yaml"


def test_unimplemented_platform_hook_falls_back(deploy_dirs):
    _, bundled_dir, _ = deploy_dirs
    platforms.current_omni_platform.get_default_stage_config_path.side_effect = NotImplementedError
    expected = write_deploy(bundled_dir)
    assert stage_config.resolve_deploy_config_path("test.yaml") == expected


def test_directory_is_not_a_platform_config(deploy_dirs):
    platform_dir, bundled_dir, _ = deploy_dirs
    (platform_dir / "test.yaml").mkdir()
    expected = write_deploy(bundled_dir)
    assert stage_config.resolve_deploy_config_path("test.yaml") == expected


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("exists", [True, False])
def test_explicit_directory_paths_are_preserved(deploy_dirs, absolute, exists):
    platform_dir, _, cwd = deploy_dirs
    write_deploy(platform_dir)
    path = cwd / "custom" / "test.yaml" if absolute else Path("custom/test.yaml")
    if exists:
        write_deploy(path.parent)
    assert stage_config.resolve_deploy_config_path(path) == path
    platforms.current_omni_platform.get_default_stage_config_path.assert_not_called()


def test_missing_bare_name_is_preserved(deploy_dirs):
    assert stage_config.resolve_deploy_config_path("missing.yaml") == Path("missing.yaml")


def test_factory_loads_platform_bare_name(deploy_dirs):
    platform_dir, bundled_dir, _ = deploy_dirs
    write_deploy(bundled_dir, devices="0")
    write_deploy(platform_dir, devices="7")
    deploy = StageConfigFactory._load_user_deploy_config("test.yaml")
    assert deploy is not None
    assert deploy.stages[0].devices == "7"


@pytest.mark.parametrize("explicit", [True, False])
def test_structured_config_retains_platform_source(deploy_dirs, explicit):
    platform_dir, bundled_dir, _ = deploy_dirs
    write_deploy(bundled_dir, devices="0")
    expected = write_deploy(platform_dir, devices="7")
    pipeline = stage_config.PipelineConfig(
        model_type="test",
        default_deploy_config_name="test.yaml",
        stages=(stage_config.StagePipelineConfig(stage_id=0, model_stage="test", final_output=True),),
    )
    deploy, source = _get_deploy_config(pipeline, None, "test.yaml" if explicit else None)
    assert deploy.stages[0].devices == "7"
    assert source == str(expected)


def test_relative_platform_directory_returns_absolute_source(deploy_dirs):
    _, _, cwd = deploy_dirs
    expected = write_deploy(cwd / "profiles")
    platforms.current_omni_platform.get_default_stage_config_path.return_value = "profiles"
    assert stage_config.get_default_deploy_config_path("test.yaml") == expected


def test_platform_hook_errors_are_not_hidden(deploy_dirs):
    platforms.current_omni_platform.get_default_stage_config_path.side_effect = RuntimeError("broken plugin")
    with pytest.raises(RuntimeError, match="broken plugin"):
        stage_config.get_default_deploy_config_path("test.yaml")


def test_invalid_platform_yaml_does_not_fall_back(deploy_dirs):
    platform_dir, bundled_dir, _ = deploy_dirs
    write_deploy(bundled_dir)
    (platform_dir / "test.yaml").write_text("stage_args: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="removed `stage_args` schema"):
        StageConfigFactory._load_user_deploy_config("test.yaml")
