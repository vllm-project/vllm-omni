# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Weekly CI selects a runner by rewriting deploy ``model_runner``."""

from pathlib import Path

import pytest
import yaml

from tests.helpers.runtime import _resolve_server_deploy_args
from tests.helpers.stage_config import (
    get_deploy_config_path,
    stage_config_path_for_run_level,
)
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    resolve_stage_model_runner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = """\
model_runner: v1
stages:
  - stage_id: 0
  - stage_id: 1
    model_runner: v1
platforms:
  cuda:
    model_runner: v1
    stages:
      - stage_id: 0
        max_num_seqs: 1
"""


def test_ci_model_runner_env_rewrites_deploy_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "demo.yaml"
    source.write_text(_DEPLOY, encoding="utf-8")
    monkeypatch.setenv("VLLM_OMNI_CI_MODEL_RUNNER", "v2")

    rewritten = stage_config_path_for_run_level(str(source), "full_model")
    assert rewritten is not None
    loaded = yaml.safe_load(Path(rewritten).read_text(encoding="utf-8"))
    assert loaded["model_runner"] == "v2"
    assert loaded["platforms"]["cuda"]["model_runner"] == "v2"
    assert loaded["platforms"]["cuda"]["stages"] == [{"stage_id": 0, "max_num_seqs": 1}]
    assert loaded["stages"][1]["model_runner"] == "v1"

    deploy = _apply_platform_overrides(load_deploy_config(rewritten), "cuda")
    assert resolve_stage_model_runner(deploy, deploy.stages[0]) == "v2"
    assert resolve_stage_model_runner(deploy, deploy.stages[1]) == "v1"


def test_ci_model_runner_env_overrides_inherited_cuda_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "base.yaml").write_text(
        "model_runner: v1\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "platforms:\n"
        "  cuda:\n"
        "    model_runner: v1\n"
        "    stages:\n"
        "      - stage_id: 0\n"
        "        max_num_seqs: 2\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("base_config: base.yaml\nstages:\n  - stage_id: 0\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_OMNI_CI_MODEL_RUNNER", "v2")

    rewritten = stage_config_path_for_run_level(str(overlay), "full_model")
    assert rewritten is not None
    loaded = yaml.safe_load(Path(rewritten).read_text(encoding="utf-8"))
    assert loaded["model_runner"] == "v2"
    assert Path(loaded["base_config"]) == (tmp_path / "base.yaml").resolve()
    assert loaded["platforms"]["cuda"] == {"model_runner": "v2"}

    deploy = _apply_platform_overrides(load_deploy_config(rewritten), "cuda")
    assert resolve_stage_model_runner(deploy, deploy.stages[0]) == "v2"
    assert deploy.stages[0].max_num_seqs == 2


def test_voxtral_direct_async_ci_config_selects_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_CI_MODEL_RUNNER", "v2")

    rewritten = stage_config_path_for_run_level(get_deploy_config_path("voxtral_tts.yaml"), "full_model")
    assert rewritten is not None
    deploy = load_deploy_config(rewritten)

    assert deploy.model_runner == "v2"
    assert all(resolve_stage_model_runner(deploy, stage) == "v2" for stage in deploy.stages)


def test_ci_model_runner_env_unset_keeps_the_launch_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "demo.yaml"
    source.write_text(_DEPLOY, encoding="utf-8")
    monkeypatch.delenv("VLLM_OMNI_CI_MODEL_RUNNER", raising=False)

    assert stage_config_path_for_run_level(str(source), "full_model") == str(source)
    assert stage_config_path_for_run_level(None, "full_model") is None


def test_ci_model_runner_env_rejects_unknown_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "demo.yaml"
    source.write_text("stages: []\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_OMNI_CI_MODEL_RUNNER", "v3")

    with pytest.raises(ValueError, match="VLLM_OMNI_CI_MODEL_RUNNER"):
        stage_config_path_for_run_level(str(source), "full_model")


def test_ci_model_runner_env_requires_a_deploy_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_CI_MODEL_RUNNER", "v2")

    with pytest.raises(ValueError, match="did not provide a deploy config"):
        stage_config_path_for_run_level(None, "full_model")


@pytest.mark.parametrize("equals_form", [False, True])
def test_server_args_deploy_config_is_rewritten_only_for_the_weekly_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, equals_form: bool
) -> None:
    source = tmp_path / "demo.yaml"
    source.write_text(_DEPLOY, encoding="utf-8")
    original = source.read_text(encoding="utf-8")
    if equals_form:
        server_args = [f"--deploy-config={source}", "--host", "127.0.0.1"]
    else:
        server_args = ["--deploy-config", str(source), "--host", "127.0.0.1"]

    monkeypatch.delenv("VLLM_OMNI_CI_MODEL_RUNNER", raising=False)
    untouched, launch_path, replaced = _resolve_server_deploy_args(server_args, None, "full_model")
    assert untouched == server_args
    assert launch_path is None
    assert not replaced

    monkeypatch.setenv("VLLM_OMNI_CI_MODEL_RUNNER", "v2")
    rewritten_args, launch_path, replaced = _resolve_server_deploy_args(server_args, None, "full_model")
    assert replaced
    assert launch_path is not None
    assert Path(launch_path) != source
    if equals_form:
        assert rewritten_args == [f"--deploy-config={launch_path}", "--host", "127.0.0.1"]
    else:
        assert rewritten_args == ["--deploy-config", launch_path, "--host", "127.0.0.1"]
    loaded = yaml.safe_load(Path(launch_path).read_text(encoding="utf-8"))
    assert loaded["model_runner"] == "v2"
    assert source.read_text(encoding="utf-8") == original
