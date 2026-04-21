#!/usr/bin/env python3
"""Serve MiniCPM-o 4.5 on Modal with an OpenAI-compatible vLLM endpoint.

Usage:
    modal run scripts/modal_minicpm_o45_vllm.py --preload-only
    modal serve scripts/modal_minicpm_o45_vllm.py
    modal deploy scripts/modal_minicpm_o45_vllm.py

This follows the official MiniCPM vLLM serving path. It is aimed at
text/image/video requests over the OpenAI-compatible `/v1` API. It does not
attempt to expose MiniCPM-o 4.5's full-duplex realtime audio stack.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Final

import modal

APP_NAME: Final = "minicpm-o45-vllm"
MODEL_ID: Final = "openbmb/MiniCPM-o-4_5"
MODEL_REVISION: Final[str | None] = None
SERVED_MODEL_NAME: Final = "cpmo"
GPU_TYPE: Final = "A100-40GB"
VLLM_PORT: Final = 8000
API_KEY_ENV_VAR: Final = "MINICPM_API_KEY"
DEFAULT_API_KEY: Final = "token-abc123"
HF_CACHE_DIR: Final = "/root/.cache/huggingface"
VLLM_CACHE_DIR: Final = "/root/.cache/vllm"
GPU_MEMORY_UTILIZATION: Final = 0.9
MAX_MODEL_LEN: Final = 4096
MAX_NUM_BATCHED_TOKENS: Final = 2048
LIMIT_MM_PER_PROMPT: Final = {"image": 5, "video": 2}
MINUTES: Final = 60


app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(f"{APP_NAME}-hf", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(f"{APP_NAME}-vllm", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "vllm[video]==0.17.1",
        "huggingface_hub[hf_transfer]==0.34.4",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
        }
    )
)


def _snapshot_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        "repo_id": MODEL_ID,
        "cache_dir": HF_CACHE_DIR,
    }
    if MODEL_REVISION is not None:
        kwargs["revision"] = MODEL_REVISION
    return kwargs


def _download_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**_snapshot_kwargs())


def _build_vllm_command() -> list[str]:
    api_key = os.environ.get(API_KEY_ENV_VAR, DEFAULT_API_KEY)
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--limit-mm-per-prompt",
        json.dumps(LIMIT_MM_PER_PROMPT),
        "--trust-remote-code",
        "--api-key",
        api_key,
        "--uvicorn-log-level",
        "info",
    ]
    if MODEL_REVISION is not None:
        cmd.extend(["--revision", MODEL_REVISION])
    return cmd


@app.function(
    image=image,
    timeout=30 * MINUTES,
    volumes={HF_CACHE_DIR: hf_cache_volume},
)
def preload_model() -> str:
    """Download MiniCPM-o 4.5 weights into the shared Hugging Face cache."""
    snapshot_path = _download_model()
    hf_cache_volume.commit()
    return snapshot_path


@app.function(
    image=image,
    gpu=GPU_TYPE,
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        VLLM_CACHE_DIR: vllm_cache_volume,
    },
)
@modal.concurrent(max_inputs=8, target_inputs=4)
@modal.web_server(VLLM_PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    """Start a vLLM OpenAI-compatible API server for MiniCPM-o 4.5."""
    snapshot_path = _download_model()
    hf_cache_volume.commit()
    print(f"Using model snapshot: {snapshot_path}")

    cmd = _build_vllm_command()
    print("Launching:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main(preload_only: bool = False) -> None:
    """CLI entrypoint for `modal run scripts/modal_minicpm_o45_vllm.py`."""
    snapshot_path = preload_model.remote()
    print(f"Model cached at: {snapshot_path}")

    if preload_only:
        return

    print("Serve locally on Modal with:")
    print("  modal serve scripts/modal_minicpm_o45_vllm.py")
    print("Deploy a persistent endpoint with:")
    print("  modal deploy scripts/modal_minicpm_o45_vllm.py")
    print(f"OpenAI base URL: <modal-url>/v1  model={SERVED_MODEL_NAME}")
    print(f"API key env var inside the container: {API_KEY_ENV_VAR} (default: {DEFAULT_API_KEY})")
