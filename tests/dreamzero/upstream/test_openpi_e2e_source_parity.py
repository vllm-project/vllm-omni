# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Formal OpenPI end-to-end parity: upstream DreamZero server vs `vllm serve`.

This test uses DreamZero's own client-side observation builders from
`${DREAMZERO_REPO}/test_client_AR.py`, and client-side websocket protocol from
`${DREAMZERO_REPO}/eval_utils/policy_client.py`.

The only client-side adaptation for vLLM is the websocket path:
DreamZero's upstream server serves at `/`, while vLLM serves OpenPI at
`/v1/realtime/robot/openpi`.

Current scope for this test:
- default two-GPU run (`nproc_per_node=2` on upstream, `--cfg-parallel-size 2` on `vllm serve`)
- non-`torch.compile` (upstream launched through
  `upstream_socket_server_no_compile.py`, vLLM with `--enforce-eager`)
- non-DiT-cache / non-skip-schedule (`NUM_DIT_STEPS=16`)

Serving contract locked by this test:
- upstream DreamZero still boots from the local checkpoint directory
- vLLM boots from the official DreamZero HF repo name (`GEAR-Dreams/DreamZero-DROID`)
  rather than a prepared local bundle path
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.runtime import get_open_port

msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")

_DREAMZERO_REPO_ENV = os.environ.get("DREAMZERO_REPO")
DREAMZERO_REPO = Path(_DREAMZERO_REPO_ENV).expanduser() if _DREAMZERO_REPO_ENV else None
if DREAMZERO_REPO is not None and str(DREAMZERO_REPO) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_REPO))

try:
    import test_client_AR as dreamzero_client
    from eval_utils.policy_client import WebsocketClientPolicy
except Exception:  # pragma: no cover - guarded by pytest skip below
    dreamzero_client = None
    WebsocketClientPolicy = None
_BaseWebsocketClientPolicy = WebsocketClientPolicy if WebsocketClientPolicy is not None else object

CHECKPOINT_DIR = DREAMZERO_REPO / "checkpoints" / "dreamzero" if DREAMZERO_REPO is not None else None
VLLM_MODEL = os.environ.get("VLLM_DREAMZERO_MODEL", "GEAR-Dreams/DreamZero-DROID")
SERVICE_READY_TIMEOUT_S = int(os.environ.get("OPENPI_SERVICE_READY_TIMEOUT_S", "900"))
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
SESSION_ID = "openpi-e2e-parity-session"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif(
        dreamzero_client is None or WebsocketClientPolicy is None,
        reason="DreamZero client modules are required on PYTHONPATH",
    ),
    pytest.mark.skipif(
        DREAMZERO_REPO is None or not DREAMZERO_REPO.exists(),
        reason="DreamZero source repo is required at DREAMZERO_REPO",
    ),
    pytest.mark.skipif(
        CHECKPOINT_DIR is None or not CHECKPOINT_DIR.exists(), reason="DreamZero local checkpoint is required"
    ),
]


class OpenPIWebsocketClientPolicy(_BaseWebsocketClientPolicy):
    """DreamZero client protocol with an OpenPI websocket path suffix."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/v1/realtime/robot/openpi",
    ) -> None:
        self._uri = f"ws://{host}:{port}{path}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()


def _vllm_executable() -> str:
    fallback = Path(sys.executable).with_name("vllm")
    if fallback.exists():
        return str(fallback)
    exe = shutil.which("vllm")
    if exe:
        return exe
    raise FileNotFoundError("Unable to locate `vllm` executable in current environment.")


def _cfg_parallel_size() -> int:
    return int(os.environ.get("OPENPI_E2E_CFG_PARALLEL_SIZE", "2"))


def _pick_test_gpus() -> list[str]:
    cfg_parallel_size = _cfg_parallel_size()
    override = os.environ.get("OPENPI_E2E_GPUS") or os.environ.get("OPENPI_E2E_GPU")
    if override is not None:
        gpus = [part.strip() for part in override.split(",") if part.strip()]
        if not gpus:
            raise ValueError("OPENPI_E2E_GPUS is set but empty.")
        if len(gpus) < cfg_parallel_size:
            raise RuntimeError(f"Need {cfg_parallel_size} GPUs, but OPENPI_E2E_GPUS only provided {gpus}.")
        return gpus

    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpu_rows = []
    for line in query.strip().splitlines():
        gpu_index, used_mb = [part.strip() for part in line.split(",", maxsplit=1)]
        gpu_rows.append((int(used_mb), gpu_index))
    gpu_rows.sort()
    gpus = [gpu_index for _, gpu_index in gpu_rows[: max(cfg_parallel_size, 1)]]
    if len(gpus) < cfg_parallel_size:
        raise RuntimeError(
            f"Need {cfg_parallel_size} GPUs for cfg_parallel_size={cfg_parallel_size}, "
            f"but found only {len(gpus)} candidates."
        )
    return gpus


def _torchrun_argv(script: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(_cfg_parallel_size()),
        script,
        "--port",
        str(port),
        "--model_path",
        str(CHECKPOINT_DIR),
    ]


def _run_upstream_service(port: int, log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{Path.cwd()}:{DREAMZERO_REPO}:{env['PYTHONPATH']}".rstrip(":")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(_pick_test_gpus())
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env["ATTENTION_BACKEND"] = "torch"
    env.setdefault("ENABLE_TENSORRT", "false")
    env["ENABLE_DIT_CACHE"] = "false"
    env["NUM_DIT_STEPS"] = "16"
    env["DYNAMIC_CACHE_SCHEDULE"] = "false"
    argv = _torchrun_argv(
        str(Path("tests/dreamzero/upstream/upstream_socket_server_no_compile.py")),
        port,
    )
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        argv,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path.cwd()),
        env=env,
    )
    proc._codex_log_file = log_file  # type: ignore[attr-defined]
    return proc


def _run_vllm_service(port: int, log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    gpus = _pick_test_gpus()
    cfg_parallel_size = _cfg_parallel_size()
    if cfg_parallel_size > len(gpus):
        raise RuntimeError(
            f"cfg_parallel_size={cfg_parallel_size} requires at least {cfg_parallel_size} GPUs, but only got {gpus}."
        )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:cfg_parallel_size])
    env.setdefault("ATTENTION_BACKEND", "torch")
    env.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    env.setdefault("MASTER_PORT", str(get_open_port()))
    argv = [
        _vllm_executable(),
        "serve",
        VLLM_MODEL,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "dreamzero-droid",
        "--enforce-eager",
    ]
    if cfg_parallel_size > 1:
        argv.extend(["--cfg-parallel-size", str(cfg_parallel_size)])
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        argv,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(Path.cwd()),
    )
    proc._codex_log_file = log_file  # type: ignore[attr-defined]
    return proc


def _stop_process(proc: subprocess.Popen[str]) -> None:
    log_file = getattr(proc, "_codex_log_file", None)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - cleanup path
            proc.kill()
            proc.wait(timeout=10)
    if log_file is not None:
        log_file.close()


def _build_obs_sequence() -> tuple[dict, dict]:
    camera_frames = dreamzero_client.load_camera_frames()
    chunks = dreamzero_client.build_frame_schedule(
        min(v.shape[0] for v in camera_frames.values()),
        1,
    )
    obs0 = dreamzero_client._make_obs_from_video(camera_frames, [0], PROMPT, SESSION_ID)
    obs1 = dreamzero_client._make_obs_from_video(camera_frames, chunks[0], PROMPT, SESSION_ID)
    return obs0, obs1


def _wait_for_client_ready(client_factory, timeout_s: float, proc=None, log_path: Path | None = None):
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            details = ""
            if log_path is not None and log_path.exists():
                details = log_path.read_text(errors="replace")[-8000:]
            raise RuntimeError(f"Service exited before becoming ready with code {proc.returncode}.\n{details}")
        try:
            return client_factory()
        except Exception as exc:  # pragma: no cover - retry path
            last_err = exc
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for websocket service: {last_err}")


def _collect_outputs_with_client(client) -> tuple[dict, list[np.ndarray]]:
    metadata = client.get_server_metadata()
    obs0, obs1 = _build_obs_sequence()
    outputs = [
        client.infer(dict(obs0)),
        client.infer(dict(obs1)),
    ]
    assert _normalize_reset_response(client.reset({})) == "reset successful"
    outputs.append(client.infer(dict(obs0)))
    client._ws.close()
    return metadata, outputs


def _normalize_reset_response(response) -> str:
    if isinstance(response, str):
        return response
    decoded = msgpack_numpy.unpackb(response)
    if isinstance(decoded, dict):
        return str(decoded.get("status"))
    return str(decoded)


def _normalize_metadata(metadata: dict) -> dict:
    normalized = dict(metadata)
    if isinstance(normalized.get("image_resolution"), tuple):
        normalized["image_resolution"] = list(normalized["image_resolution"])
    return normalized


def _assert_logs_clean(log_path: Path) -> None:
    text = log_path.read_text(errors="replace")
    if "SignalException: Process" in text and "got signal: 15" in text:
        text = text.split("Traceback (most recent call last):", 1)[0]
    assert "Traceback" not in text, text
    assert "RuntimeError:" not in text, text


def _assert_upstream_log_matches_vllm_baseline(log_path: Path) -> None:
    text = log_path.read_text(errors="replace")
    assert "DIT Compute Steps 8 steps" not in text, text
    assert "DIT Compute Steps 16 steps" in text, text


def test_openpi_service_matches_upstream_server_noncompile(tmp_path: Path) -> None:
    expected_metadata = {
        "image_resolution": [180, 320],
        "n_external_cameras": 2,
        "needs_wrist_camera": True,
        "needs_stereo_camera": False,
        "needs_session_id": True,
        "action_space": "joint_position",
    }

    upstream_port = get_open_port()
    upstream_log = tmp_path / "dreamzero_upstream.log"
    upstream_proc = _run_upstream_service(upstream_port, upstream_log)
    try:
        upstream_client = _wait_for_client_ready(
            lambda: WebsocketClientPolicy(host="127.0.0.1", port=upstream_port),
            timeout_s=SERVICE_READY_TIMEOUT_S,
            proc=upstream_proc,
            log_path=upstream_log,
        )
        upstream_metadata, upstream_outputs = _collect_outputs_with_client(upstream_client)
    finally:
        _stop_process(upstream_proc)
    _assert_logs_clean(upstream_log)
    _assert_upstream_log_matches_vllm_baseline(upstream_log)

    vllm_port = get_open_port()
    vllm_log = tmp_path / "vllm_openpi.log"
    vllm_proc = _run_vllm_service(vllm_port, vllm_log)
    try:
        vllm_client = _wait_for_client_ready(
            lambda: OpenPIWebsocketClientPolicy(host="127.0.0.1", port=vllm_port),
            timeout_s=SERVICE_READY_TIMEOUT_S,
            proc=vllm_proc,
            log_path=vllm_log,
        )
        vllm_metadata, vllm_outputs = _collect_outputs_with_client(vllm_client)
    finally:
        _stop_process(vllm_proc)
    _assert_logs_clean(vllm_log)

    assert _normalize_metadata(upstream_metadata) == expected_metadata
    assert _normalize_metadata(vllm_metadata) == expected_metadata

    for idx, (actual, expected) in enumerate(zip(vllm_outputs, upstream_outputs, strict=True)):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-2,
            atol=1e-3,
            err_msg=f"OpenPI step {idx} output mismatch",
        )
