import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/online_serving/text_to_speech/indextts2/run_server.sh"
README = ROOT / "examples/online_serving/text_to_speech/README.md"
MACHINE_SPECIFIC_HOME_PATH = re.compile(r"(?:/home|/Users)/[^/\s`]+/")


def run_server(
    tmp_path: Path,
    *,
    model_version: str | None = None,
    model: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    argv_file = tmp_path / "vllm.argv"
    stub = tmp_path / "vllm"
    stub.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@" > "$VLLM_ARGV_FILE"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    env = os.environ.copy()
    for name in ("MODEL_VERSION", "MODEL", "DEPLOY_CONFIG", "PORT"):
        env.pop(name, None)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["VLLM_ARGV_FILE"] = str(argv_file)

    if model_version is not None:
        env["MODEL_VERSION"] = model_version
    if model is not None:
        env["MODEL"] = model
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, argv_file


def read_argv(argv_file: Path) -> list[str]:
    return argv_file.read_bytes().decode().rstrip("\0").split("\0")


@pytest.mark.parametrize("model", [None, ""])
def test_version_2_5_requires_model_before_invoking_vllm(
    tmp_path: Path,
    model: str | None,
) -> None:
    result, argv_file = run_server(tmp_path, model_version="2.5", model=model)

    assert result.returncode == 2
    assert "MODEL=/path/to/native/bundle" in result.stderr
    assert not argv_file.exists()


def test_version_2_5_selects_deploy_config(
    tmp_path: Path,
) -> None:
    model = "/models/indextts-2.5"
    result, argv_file = run_server(
        tmp_path,
        model_version="2.5",
        model=model,
    )

    assert result.returncode == 0, result.stderr
    argv = read_argv(argv_file)
    assert argv[:2] == ["serve", model]
    deploy_config_index = argv.index("--deploy-config") + 1
    assert Path(argv[deploy_config_index]) == ROOT / "vllm_omni/deploy/indextts2_5.yaml"


@pytest.mark.parametrize("model_version", [None, "2.0"])
def test_default_and_version_2_0_use_huggingface_model(
    tmp_path: Path,
    model_version: str | None,
) -> None:
    result, argv_file = run_server(tmp_path, model_version=model_version)

    assert result.returncode == 0, result.stderr
    argv = read_argv(argv_file)
    assert argv[:2] == ["serve", "IndexTeam/IndexTTS-2"]
    deploy_config_index = argv.index("--deploy-config") + 1
    assert Path(argv[deploy_config_index]) == ROOT / "vllm_omni/deploy/indextts2.yaml"


def test_invalid_model_version_exits_before_invoking_vllm(tmp_path: Path) -> None:
    result, argv_file = run_server(tmp_path, model_version="3.0")

    assert result.returncode == 2
    assert "MODEL_VERSION must be 2.0 or 2.5" in result.stderr
    assert not argv_file.exists()


@pytest.mark.parametrize("path", [SCRIPT, README])
def test_public_example_does_not_contain_machine_specific_home_path(path: Path) -> None:
    assert MACHINE_SPECIFIC_HOME_PATH.search(path.read_text(encoding="utf-8")) is None
