# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end regressions for YuE2 offline WAV export."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import get_model_prefix
from vllm_omni.transformers_utils.repo_utils import hf_api

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    *hardware_marks(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1),
]

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def vae_path() -> Path:
    path = Path(os.environ.get("YUE2_VAE", get_model_prefix() + "m-a-p/YuE2-Vae"))
    if not path.is_dir():
        path = Path(hf_api().snapshot_download(str(path), allow_patterns=["*.json", "*.safetensors"]))
    return path


@pytest.mark.parametrize("max_frames,truncated", [(200, True), (9000, False)], ids=["budget", "natural-eos"])
def test_offline_example_saves_audio(vae_path: Path, tmp_path: Path, max_frames: int, truncated: bool) -> None:
    model = Path(os.environ.get("YUE2_MODEL_DIR", get_model_prefix() + "m-a-p/YuE2-3B"))
    if not model.is_dir():
        model = Path(
            hf_api().snapshot_download(str(model), allow_patterns=["*.json", "*.safetensors", "qwen.tiktoken"])
        )
    output = tmp_path / "song.wav"
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples/offline_inference/yue2/end2end.py"),
            "--model",
            str(model),
            "--vae",
            str(vae_path),
            "--style",
            "gentle acoustic folk, clear English vocals, 90 BPM",
            "--lyrics",
            "[Verse]\nMorning light across the bay\nCarry all my dreams away\n[Outro]",
            "--cot",
            "off",
            "--seed",
            "831001",
            "--max-frames",
            str(max_frames),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"truncated={truncated}" in completed.stdout
    audio, sample_rate = sf.read(output, always_2d=True)
    assert sample_rate == 48000 and audio.shape[1] == 2
    assert np.isfinite(audio).all() and np.abs(audio).max() > 0
    duration = len(audio) / sample_rate
    if truncated:
        assert duration == pytest.approx(max_frames / 25, abs=0.1)
    else:
        assert 8 <= duration < max_frames / 25
