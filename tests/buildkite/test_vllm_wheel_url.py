# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the Docker install commands without downloading packages or building images."""

import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dockerfile", ["Dockerfile.ci", "Dockerfile.cuda"])
@pytest.mark.parametrize("version", ["0.33.0rc3.dev3+g4ba787ec9", "0.33.0"])
def test_cuda_wheel_url_encodes_local_version(tmp_path: Path, dockerfile: str, version: str) -> None:
    text = (REPO_ROOT / "docker" / dockerfile).read_text().replace("\\\n", "")
    install = next(line.removeprefix("RUN ") for line in text.splitlines() if "vllm @ https://wheels.vllm.ai/" in line)
    # Run the real shell expansion, replacing only the installer with an argv recorder.
    # A literal '+' is decoded as a space by the wheel CDN and returned HTTP404
    # for 4ba787ec99, while the percent-encoded path returned HTTP200.
    stub = tmp_path / "uv"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$WHEEL_ARGS"\n')
    stub.chmod(0o755)
    (tmp_path / "vllm-omni").mkdir()
    args_file = tmp_path / "args.txt"
    commit = "4ba787ec9959e566d3b587a796dff07673324da5"
    env = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        WHEEL_ARGS=str(args_file),
        COMMON_WORKDIR=str(tmp_path),
        VLLM_PRECOMPILED_WHEEL_COMMIT=commit,
        VLLM_VERSION=version,
    )
    subprocess.run(["/bin/sh", "-eu", "-c", install], cwd=tmp_path, env=env, check=True)
    requirement = next(arg for arg in args_file.read_text().splitlines() if arg.startswith("vllm @ "))
    wheel_url = requirement.removeprefix("vllm @ ")
    arch = os.uname().machine
    expected_filename = f"vllm-{quote(version, safe='')}-cp38-abi3-manylinux_2_28_{arch}.whl"
    assert wheel_url == f"https://wheels.vllm.ai/{commit}/{expected_filename}"
    assert unquote(wheel_url).split("/")[-1] == f"vllm-{version}-cp38-abi3-manylinux_2_28_{arch}.whl"
    if dockerfile == "Dockerfile.ci":
        args = args_file.read_text().splitlines()
        assert args[args.index("-r") + 1] == "/tmp/vllm-otel-roots.txt"


def test_bootstrap_telemetry_solve_keeps_unselected_exporters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The newer common exporter can force an SDK upgrade while an older,
    # unselected Prometheus exporter still requires the old SDK. Both reverse
    # constraints must participate in the normal combined dependency solve.
    telemetry = {
        "opentelemetry-api",
        "opentelemetry-sdk",
        "OpenTelemetry-Exporter-OTLP-Common",
        "opentelemetry-exporter-prometheus",
        "opentelemetry-instrumentation-fastapi",
        "opentelemetry-instrumentation-requests",
    }
    for index, name in enumerate(sorted(telemetry | {"torch", "gradio"})):
        metadata = tmp_path / f"fixture_{index}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    text = (REPO_ROOT / "docker/Dockerfile.ci").read_text().replace("\\\n", "")
    collection = next(
        shlex.split(line.removeprefix("RUN "))
        for line in text.splitlines()
        if line.startswith("RUN python3 tools/prepare_ci_telemetry.py")
    )
    assert collection[:6] == [
        "python3",
        "tools/prepare_ci_telemetry.py",
        "--roots",
        "/tmp/vllm-otel-roots.txt",
        "--orphans",
        "/tmp/vllm-otel-orphans.txt",
    ]
    prepare = runpy.run_path(str(REPO_ROOT / collection[1]))["prepare"]
    roots, orphans = tmp_path / "roots.txt", tmp_path / "orphans.txt"
    # Discovery reads only these real dist-info fixtures; installed developer
    # packages and editable finders cannot enter this probe.
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "path", [str(tmp_path)])
        prepare(roots, orphans)
    assert set(roots.read_text().splitlines()) == telemetry
    assert len(roots.read_text().splitlines()) == len(telemetry)
    assert orphans.read_text() == ""
    assert "RUN uv pip check --system" in text
