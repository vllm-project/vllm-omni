# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for VoxCPM offline inference."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import tests.helpers.runtime as omni_runtime
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.model_executor.models.voxcpm.voxcpm_runtime_utils import (
    prepare_voxcpm_hf_config_dir,
    resolve_voxcpm_model_dir,
)

VOXCPM_MODEL = os.environ.get("VOXCPM_MODEL", "OpenBMB/VoxCPM1.5")
STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"
)
SAMPLE_RATE = 24000


@pytest.fixture(autouse=True)
def _patch_npu_cleanup_for_voxcpm(monkeypatch: pytest.MonkeyPatch):
    """Limit the NPU cleanup workaround to this VoxCPM test module only."""
    original_cleanup = omni_runtime.cleanup_dist_env_and_memory

    def _safe_cleanup() -> None:
        try:
            original_cleanup()
        except RuntimeError as exc:
            if "Allocator for npu is not a DeviceAllocator" in str(exc):
                return
            raise

    monkeypatch.setattr(omni_runtime, "cleanup_dist_env_and_memory", _safe_cleanup)


def _build_prompt(text: str) -> dict[str, Any]:
    return {
        "prompt_token_ids": [1],
        "additional_information": {
            "text": [text],
            "cfg_value": [2.0],
            "inference_timesteps": [10],
            "min_len": [2],
            "max_new_tokens": [1024],
        },
    }


def _extract_audio_tensor(multimodal_output: dict[str, Any]) -> torch.Tensor:
    audio = multimodal_output.get("audio", multimodal_output.get("model_outputs"))
    assert audio is not None, f"No audio output found, keys={list(multimodal_output.keys())}"

    if isinstance(audio, list):
        parts: list[torch.Tensor] = []
        for item in audio:
            if item is None:
                continue
            tensor = torch.as_tensor(item)
            if tensor.numel() == 0:
                continue
            parts.append(tensor.float().cpu().reshape(-1))
        return torch.cat(parts, dim=-1) if parts else torch.zeros((0,), dtype=torch.float32)

    return torch.as_tensor(audio).float().cpu().reshape(-1)


def _extract_final_multimodal_output(outputs) -> dict[str, Any]:
    for item in reversed(outputs):
        request_output = getattr(item, "request_output", None)
        if request_output is not None:
            multimodal_output = getattr(request_output, "multimodal_output", None)
            if isinstance(multimodal_output, dict):
                return multimodal_output
            completions = getattr(request_output, "outputs", None) or []
            for completion in completions:
                multimodal_output = getattr(completion, "multimodal_output", None)
                if isinstance(multimodal_output, dict):
                    return multimodal_output

        multimodal_output = getattr(item, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            return multimodal_output

    raise AssertionError("No multimodal audio output found in VoxCPM generate results")


@pytest.fixture
def voxcpm_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    model_dir = resolve_voxcpm_model_dir(VOXCPM_MODEL)

    hf_config_env = os.environ.get("VLLM_OMNI_VOXCPM_HF_CONFIG_PATH")
    if hf_config_env:
        hf_config_dir = Path(hf_config_env).expanduser()
    else:
        hf_config_dir = tmp_path / "voxcpm_hf_config"

    if not (hf_config_dir / "config.json").exists():
        prepare_voxcpm_hf_config_dir(model_dir, hf_config_dir)

    monkeypatch.setenv("VLLM_OMNI_VOXCPM_HF_CONFIG_PATH", str(hf_config_dir))
    return str(model_dir)


def test_prepare_voxcpm_hf_config_dir(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 1024}), encoding="utf-8")
    (model_dir / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")

    hf_config_dir = prepare_voxcpm_hf_config_dir(model_dir, tmp_path / "voxcpm_hf_config")

    prepared_config = json.loads((hf_config_dir / "config.json").read_text(encoding="utf-8"))
    assert prepared_config["model_type"] == "voxcpm"
    assert prepared_config["architectures"] == ["VoxCPMForConditionalGeneration"]
    assert (hf_config_dir / "generation_config.json").exists()


def test_resolve_voxcpm_model_dir_local_path(tmp_path: Path):
    model_dir = tmp_path / "OpenBMB" / "VoxCPM1.5"
    model_dir.mkdir(parents=True)

    assert resolve_voxcpm_model_dir(str(model_dir)) == model_dir


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxcpm_zero_shot_001(voxcpm_model_path: str):
    with OmniRunner(voxcpm_model_path, stage_configs_path=STAGE_CONFIG) as runner:
        outputs = list(runner.omni.generate(_build_prompt("Hello, this is a VoxCPM offline inference test.")))

    assert outputs, "No outputs returned"

    multimodal_output = _extract_final_multimodal_output(outputs)
    audio = _extract_audio_tensor(multimodal_output)
    assert audio.numel() > SAMPLE_RATE // 2, f"Audio too short: {audio.numel()} samples"

    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"

    peak = float(torch.max(torch.abs(audio)).item()) if audio.numel() > 0 else 0.0
    assert peak > 0.01, "Generated audio appears to be silence"

    audio_np = audio.numpy()
    rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
    assert rms > 1e-4, "Generated audio RMS too low"
