# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local asset/config construction for ZONOS2 single-card acceptance."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def model_path() -> str:
    path = Path(os.environ.get("ZONOS2_TEST_MODEL_PATH", ""))
    if "ZONOS2_TEST_MODEL_PATH" not in os.environ or not (path / "config.json").is_file():
        raise RuntimeError("Set ZONOS2_TEST_MODEL_PATH to the locally converted ZONOS2 safetensors checkpoint")
    return str(path)


def deployment(directory: Path, *, streaming: bool = True) -> Path:
    data = yaml.safe_load((ROOT / "vllm_omni/deploy/zonos2.yaml").read_text())
    data["async_chunk"] = streaming
    data["connectors"]["connector_of_shared_memory"]["extra"]["codec_streaming"] = streaming
    data["stages"][0].update(
        devices="0", gpu_memory_utilization=0.72, max_num_seqs=4, max_model_len=6144, max_num_batched_tokens=128
    )
    data["stages"][1].update(devices="0", gpu_memory_utilization=0.1, max_num_seqs=4)
    # Never inherit a dummy setting from a validation override.
    for stage in data["stages"]:
        stage["load_format"] = "auto"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "deploy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def environment(directory: Path) -> dict[str, str]:
    return {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_MODULES_CACHE": str(directory / "hf-modules"),
        "ZONOS2_E2E_TRACE_DIR": str(directory / "trace"),
        "VLLM_ZONOS2_TN_CACHE_DIR": os.environ.get("VLLM_ZONOS2_TN_CACHE_DIR", str(directory / "tn-cache")),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }


def reference_audio(second: bool = False) -> Path:
    relative = "glm_tts/jiayan_zh.wav" if second else "qwen3_tts/clone_2.wav"
    return ROOT / "tests/assets" / relative
