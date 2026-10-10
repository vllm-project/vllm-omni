# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SGLang-Omni server adapter for DFX performance benchmarks."""

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
        if config.get("server_type") != "sglang-omni":
            continue
        params = config["server_params"]
        hardware: dict[str, Any] = next(
            (item["hardware_marks"] for item in config.get("mark", []) if isinstance(item, dict)),
            {},
        )
        entries.append(
            (
                {
                    "test_name": config["test_name"],
                    "model": params["model"],
                    "serve_args": _serve_args(params.get("serve_args", {})),
                    "env": params.get("env", {}),
                    "resource_label": hardware.get("res", {}).get("cuda", "na"),
                },
                config["test_name"],
            )
        )
    return entries


class SglangOmniServer:
    server_type = "sglang-omni"

    def __init__(self, config: dict[str, Any]) -> None:
        self.model = config["model"]
        self.host = "127.0.0.1"
        self.port = _open_port()
        self._args = config["serve_args"]
        self._env = config["env"]
        self.resource_label = config["resource_label"]
        self._proc: subprocess.Popen[Any] | None = None

    def __enter__(self) -> SglangOmniServer:
        command = [
            _executable(),
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
        print(f"Launching SGLang-Omni: {' '.join(command)}")
        self._proc = subprocess.Popen(command, cwd=_REPO_ROOT, env=env, start_new_session=True)
        deadline = time.monotonic() + int(os.environ.get("SGLANG_OMNI_SERVER_TIMEOUT", "1800"))
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"SGLang-Omni exited with code {self._proc.returncode}")
            with socket.socket() as sock:
                sock.settimeout(1)
                if sock.connect_ex((self.host, self.port)) == 0:
                    return self
            time.sleep(2)
        self.__exit__()
        raise TimeoutError(f"SGLang-Omni did not become ready on {self.host}:{self.port}")

    def __exit__(self, *_: object) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        os.killpg(self._proc.pid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()
