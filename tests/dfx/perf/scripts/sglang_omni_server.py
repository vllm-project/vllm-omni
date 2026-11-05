# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Launch the selected SGLang server in its own environment for DFX benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
SGLANG_SERVER_TYPES = {"sglang-omni", "sglang-diffusion"}


def _open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _executable() -> str:
    configured = os.environ.get("SGLANG_OMNI_EXECUTABLE")
    if configured:
        return configured
    sibling = Path(sys.executable).with_name("sgl-omni")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("sgl-omni") or "sgl-omni"


def _installed_revision(executable: str) -> str | None:
    # The server has its own environment; the benchmark client's distribution
    # metadata describes vLLM and cannot identify the SGLang installation.
    launcher = Path(shutil.which(executable) or executable)
    direct_url = subprocess.check_output(
        [
            str(launcher.with_name("python3")),
            "-c",
            "from importlib.metadata import distribution; "
            "print(distribution('sglang-omni').read_text('direct_url.json') or '{}')",
        ],
        text=True,
    )
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def _serve_args(values: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in values.items():
        flag = f"--{key.replace('_', '-')}"
        if key == "config":
            path = Path(str(value))
            value = path if path.is_absolute() else _REPO_ROOT / path
        if isinstance(value, bool):
            args.extend([flag, str(value).lower()])
        elif isinstance(value, dict):
            args.extend([flag, json.dumps(value, separators=(",", ":"))])
        elif value is not None:
            args.extend([flag, str(value)])
    return args


def sglang_server_entries(configs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    entries: list[tuple[dict[str, Any], str]] = []
    for config in configs:
        if config.get("server_type") not in SGLANG_SERVER_TYPES:
            continue
        params = config["server_params"]
        entries.append(
            (
                {
                    "test_name": config["test_name"],
                    "server_type": config["server_type"],
                    "model": params["model"],
                    "serve_args": _serve_args(params.get("serve_args", {})),
                    "env": params.get("env", {}),
                },
                config["test_name"],
            )
        )
    return entries


class SglangOmniServer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.server_type = config.get("server_type", "sglang-omni")
        self.model = config["model"]
        self.host = "127.0.0.1"
        self.port = _open_port()
        self._args = config["serve_args"]
        self._env = config["env"]
        self.server_revision: str | None = None
        self.config_path = self._args[self._args.index("--config") + 1] if "--config" in self._args else None
        self._proc: subprocess.Popen[Any] | None = None

    def __enter__(self) -> SglangOmniServer:
        executable = _executable()
        if self.server_type == "sglang-diffusion":
            launcher = Path(shutil.which(executable) or executable).with_name("sglang")
            executable = str(launcher)
            self.server_revision = subprocess.check_output(
                [
                    str(launcher.with_name("python3")),
                    "-c",
                    "from importlib.metadata import version; print(version('sglang'))",
                ],
                text=True,
            ).strip()
        else:
            self.server_revision = _installed_revision(executable)
        command = [
            executable,
            "serve",
            "--model-path",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self._args,
        ]
        env = os.environ.copy()
        env.update(self._env)
        print(f"Launching {self.server_type}: {' '.join(command)}")
        self._proc = subprocess.Popen(command, cwd=_REPO_ROOT, env=env, start_new_session=True)
        deadline = time.monotonic() + int(os.environ.get("SGLANG_OMNI_SERVER_TIMEOUT", "1800"))
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"{self.server_type} exited with code {self._proc.returncode}")
            with socket.socket() as sock:
                sock.settimeout(1)
                if sock.connect_ex((self.host, self.port)) == 0:
                    return self
            time.sleep(2)
        self.__exit__()
        raise TimeoutError(f"{self.server_type} did not become ready on {self.host}:{self.port}")

    def __exit__(self, *_: object) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        os.killpg(self._proc.pid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()
