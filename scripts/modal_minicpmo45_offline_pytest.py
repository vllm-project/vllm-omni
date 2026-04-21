#!/usr/bin/env python3
"""Run the MiniCPM-o 4.5 offline pytest on Modal.

Usage:
    modal run scripts/modal_minicpmo45_offline_pytest.py --preload-only
    modal run scripts/modal_minicpmo45_offline_pytest.py
    modal run scripts/modal_minicpmo45_offline_pytest.py --pytest-k no_ref
    modal run scripts/modal_minicpmo45_offline_pytest.py --junitxml-output minicpmo45_offline_pytest.xml
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Final

import modal

APP_NAME: Final = "minicpmo45-offline-pytest"
MODEL_ID: Final = "openbmb/MiniCPM-o-4_5"
VLLM_VERSION: Final = "0.19.0"
GPU_REQUEST: Final = "L40S:2"
HF_CACHE_DIR: Final = "/root/.cache/huggingface"
VLLM_CACHE_DIR: Final = "/root/.cache/vllm"
REMOTE_OUTPUT_DIR: Final = Path("/root/minicpmo45-offline-pytest-output")
REMOTE_REPO_ROOT: Final = Path("/root/vllm-omni-local")
REMOTE_REQUIREMENTS_ROOT: Final = Path("/root/vllm-omni-requirements")
DEFAULT_PYTEST_TARGET: Final = "tests/e2e/offline_inference/test_minicpmo4_5.py"
ARTIFACT_DIR_ENV: Final = "MINICPMO45_E2E_OUTPUT_DIR"
REF_AUDIO_PATH_ENV: Final = "MINICPMO45_REF_AUDIO_PATH"
MINUTES: Final = 60

LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IGNORE = [
    ".git",
    ".cursor",
    ".venv",
    "build",
    "dist",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/*.pyc",
]


app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(f"{APP_NAME}-hf", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(f"{APP_NAME}-vllm", create_if_missing=True)
output_volume = modal.Volume.from_name(f"{APP_NAME}-output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "espeak-ng")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]==0.34.4",
        "pytest==8.4.1",
        "pytest-asyncio>=0.21.0",
        "opencv-python-headless",
        "pillow",
        "requests",
        "openai",
        "soundfile",
        "pyyaml",
        "psutil",
        "cloudpickle",
        "pyttsx3>=2.99",
    )
    .add_local_dir(
        str(LOCAL_REPO_ROOT / "requirements"),
        remote_path=str(REMOTE_REQUIREMENTS_ROOT),
        copy=True,
    )
    .run_commands(f"VLLM_OMNI_TARGET_DEVICE=cuda python -m pip install -r {REMOTE_REQUIREMENTS_ROOT / 'cuda.txt'}")
    .run_commands("python -m pip install --no-build-isolation 'minicpmo-utils[all]'")
    .run_commands("python -m pip install 'torchcodec'")
    .run_commands("python -m pip install --force-reinstall 'transformers==4.57.5'")
    .run_commands("python -m pip install --force-reinstall 'numpy==2.2.6' 'numba==0.61.2'")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": str(REMOTE_REPO_ROOT),
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
            "VLLM_TARGET_DEVICE": "cuda",
            "VLLM_OMNI_TARGET_DEVICE": "cuda",
        }
    )
    .add_local_dir(
        str(LOCAL_REPO_ROOT),
        remote_path=str(REMOTE_REPO_ROOT),
        copy=False,
        ignore=LOCAL_IGNORE,
    )
)


def _download_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        resume_download=True,
    )


def _tail_text(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _list_relative_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(REMOTE_OUTPUT_DIR)) for path in root.rglob("*") if path.is_file())


def _resolve_remote_repo_path(local_path_str: str) -> str:
    local_path = Path(local_path_str).expanduser()
    if not local_path.is_absolute():
        local_path = (LOCAL_REPO_ROOT / local_path).resolve()
    else:
        local_path = local_path.resolve()

    if not local_path.is_file():
        raise FileNotFoundError(f"Reference audio file not found: {local_path}")

    try:
        relative_path = local_path.relative_to(LOCAL_REPO_ROOT)
    except ValueError as e:
        raise ValueError(
            f"Reference audio path must be inside the repository for the Modal offline pytest runner. Got: {local_path}"
        ) from e

    return str(REMOTE_REPO_ROOT / relative_path)


@app.function(
    image=image,
    timeout=30 * MINUTES,
    volumes={HF_CACHE_DIR: hf_cache_volume},
)
def preload_model() -> str:
    snapshot_path = _download_model()
    hf_cache_volume.commit()
    return snapshot_path


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    timeout=45 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        VLLM_CACHE_DIR: vllm_cache_volume,
        str(REMOTE_OUTPUT_DIR): output_volume,
    },
)
def run_pytest(
    pytest_target: str = DEFAULT_PYTEST_TARGET,
    pytest_k: str = "",
    remote_stdout_name: str = "pytest_stdout.log",
    remote_stderr_name: str = "pytest_stderr.log",
    remote_junitxml_name: str = "",
    remote_artifact_dir_name: str = "",
    remote_ref_audio_path: str = "",
) -> dict[str, Any]:
    snapshot_path = _download_model()
    hf_cache_volume.commit()

    command = [
        "python",
        "-m",
        "pytest",
        pytest_target,
        "-s",
        "-v",
        "--maxfail=1",
    ]
    if pytest_k:
        command.extend(["-k", pytest_k])
    if remote_junitxml_name:
        command.extend(["--junitxml", str(REMOTE_OUTPUT_DIR / remote_junitxml_name)])

    artifact_dir = REMOTE_OUTPUT_DIR / remote_artifact_dir_name if remote_artifact_dir_name else None
    env = os.environ.copy()
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env[ARTIFACT_DIR_ENV] = str(artifact_dir)
    if remote_ref_audio_path:
        env[REF_AUDIO_PATH_ENV] = remote_ref_audio_path

    proc = subprocess.run(
        command,
        cwd=str(REMOTE_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    REMOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (REMOTE_OUTPUT_DIR / remote_stdout_name).write_text(proc.stdout, encoding="utf-8")
    (REMOTE_OUTPUT_DIR / remote_stderr_name).write_text(proc.stderr, encoding="utf-8")
    output_volume.commit()

    artifact_paths = _list_relative_files(artifact_dir) if artifact_dir is not None else []

    return {
        "snapshot_path": snapshot_path,
        "command": command,
        "exit_code": int(proc.returncode),
        "stdout_path": remote_stdout_name,
        "stderr_path": remote_stderr_name,
        "junitxml_path": remote_junitxml_name or None,
        "stdout_tail": _tail_text(proc.stdout),
        "stderr_tail": _tail_text(proc.stderr),
        "artifact_paths": artifact_paths,
    }


@app.local_entrypoint()
def main(
    pytest_target: str = DEFAULT_PYTEST_TARGET,
    pytest_k: str = "",
    preload_only: bool = False,
    junitxml_output: str = "",
    stdout_output: str = "",
    stderr_output: str = "",
    artifact_output_dir: str = "minicpmo45_offline_pytest_artifacts",
    ref_audio_path: str = "",
) -> None:
    snapshot_path = preload_model.remote()
    print(f"Model cached at: {snapshot_path}")

    if preload_only:
        return

    stdout_name = f"{uuid.uuid4().hex}_{Path(stdout_output or 'pytest_stdout.log').name}"
    stderr_name = f"{uuid.uuid4().hex}_{Path(stderr_output or 'pytest_stderr.log').name}"
    junitxml_name = f"{uuid.uuid4().hex}_{Path(junitxml_output).name}" if junitxml_output else ""
    artifact_dir_name = f"{uuid.uuid4().hex}_artifacts"
    remote_ref_audio_path = _resolve_remote_repo_path(ref_audio_path) if ref_audio_path else ""

    result = run_pytest.remote(
        pytest_target=pytest_target,
        pytest_k=pytest_k,
        remote_stdout_name=stdout_name,
        remote_stderr_name=stderr_name,
        remote_junitxml_name=junitxml_name,
        remote_artifact_dir_name=artifact_dir_name,
        remote_ref_audio_path=remote_ref_audio_path,
    )

    stdout_bytes = b"".join(output_volume.read_file(result["stdout_path"]))
    stderr_bytes = b"".join(output_volume.read_file(result["stderr_path"]))
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    if stdout_output:
        stdout_path = Path(stdout_output)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        print(f"Saved pytest stdout to: {stdout_path}")
    else:
        print(stdout_text)

    if stderr_output:
        stderr_path = Path(stderr_output)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(stderr_text, encoding="utf-8")
        print(f"Saved pytest stderr to: {stderr_path}")
    elif stderr_text.strip():
        print(stderr_text)

    if junitxml_output:
        junit_bytes = b"".join(output_volume.read_file(result["junitxml_path"]))
        junit_path = Path(junitxml_output)
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        junit_path.write_bytes(junit_bytes)
        print(f"Saved junit xml to: {junit_path}")

    local_artifact_paths: list[str] = []
    if artifact_output_dir and result.get("artifact_paths"):
        artifact_root = Path(artifact_output_dir)
        for remote_path in result["artifact_paths"]:
            local_path = artifact_root / remote_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_bytes = b"".join(output_volume.read_file(remote_path))
            local_path.write_bytes(artifact_bytes)
            local_artifact_paths.append(str(local_path.resolve()))
        if local_artifact_paths:
            print("Saved pytest artifacts:")
            for local_path in local_artifact_paths:
                print(f"  {local_path}")

    print(
        json.dumps(
            {
                "command": result["command"],
                "exit_code": result["exit_code"],
                "snapshot_path": result["snapshot_path"],
                "stdout_path": result["stdout_path"],
                "stderr_path": result["stderr_path"],
                "junitxml_path": result["junitxml_path"],
                "artifact_paths": result["artifact_paths"],
                "local_artifact_paths": local_artifact_paths,
                "remote_ref_audio_path": remote_ref_audio_path or None,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if result["exit_code"] != 0:
        raise SystemExit(result["exit_code"])
