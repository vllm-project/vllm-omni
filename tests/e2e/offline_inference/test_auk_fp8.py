# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E check of AuK with the DiT block linears as FP8 GEMMs.

The stage-level ``diffusion_quantization_config: fp8`` has to reach the
pipeline through the deploy config, and the FP8 linears have to survive the
regional compile and the per-step CUDA graph that the default deploy runs the
DiT under. Needs an assembled checkpoint, like the other AuK E2E tests:

    VLLM_OMNI_AUK_MODEL_DIR=/path/to/auk-omni python -m pytest tests/e2e/offline_inference/test_auk_fp8.py

Fidelity of the FP8 path is covered by ``tests/diffusion/models/auk/test_auk_fp8_linear.py``;
the checks here are structural.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.diffusion.models.auk import pipeline_auk
from vllm_omni.diffusion.models.auk.fp8_linear import Fp8Linear, fp8_supported
from vllm_omni.model_extras.auk import HOP, SAMPLE_RATE, auk_prompt, auk_sampling_params

MODEL_DIR_ENV = "VLLM_OMNI_AUK_MODEL_DIR"
REFERENCE_WAV_PATH = get_asset_path("cosyvoice3/zero_shot_prompt.wav")
TEXT = "The quick brown fox jumps over the lazy dog."
GEN_SECONDS = 3.0
# 8 Euler steps keep the test short; the base recipe is 32.
NFE = 8

_model_dir = os.environ.get(MODEL_DIR_ENV)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skipif(not _model_dir, reason=f"set {MODEL_DIR_ENV} to an assembled AuK directory"),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and fp8_supported(torch.device("cuda"))),
        reason="FP8 GEMMs need an Ada or Hopper CUDA device",
    ),
]


def _fp8_deploy(tmp_path: Path) -> str:
    """``auk.yaml`` with FP8 on the diffusion stage and a startup trimmed to what the test uses."""
    with open(get_deploy_config_path("auk.yaml")) as f:
        deploy = yaml.safe_load(f)
    stage = next(s for s in deploy["stages"] if s["stage_id"] == 1)
    stage["diffusion_quantization_config"] = "fp8"
    # Compile and capture the one DiT shape on its first request instead of
    # warming the 3/6/12 s shapes, and warm a single codec bucket.
    stage["model_config"] = {
        **stage.get("model_config", {}),
        "auk_dit_warmup_frames": [],
        "auk_vae_compile_shapes": [160],
    }
    path = tmp_path / "auk_fp8.yaml"
    path.write_text(yaml.safe_dump(deploy))
    return str(path)


def _waveform(output) -> np.ndarray:
    audio = output.multimodal_output["audio"]
    if isinstance(audio, list):
        audio = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in audio])
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_auk_fp8_zero_shot_tts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    swapped: list[tuple[int, int]] = []
    original = pipeline_auk.quantize_block_linears

    def recording_quantize(dit: torch.nn.Module) -> int:
        count = original(dit)
        swapped.append((count, sum(isinstance(module, Fp8Linear) for module in dit.modules())))
        return count

    wav, sr = sf.read(str(REFERENCE_WAV_PATH), dtype="float32", always_2d=False)
    prompt = auk_prompt(f"Say the following with the same voice: '{TEXT}'", (wav, int(sr)), gen_seconds=GEN_SECONDS)
    # The diffusion stage is built in this process, so the patch must precede Omni.
    with monkeypatch.context() as patch:
        patch.setattr(pipeline_auk, "quantize_block_linears", recording_quantize)
        with OmniRunner(str(Path(_model_dir or ".").resolve()), deploy_config=_fp8_deploy(tmp_path)) as runner:
            # The first request compiles the blocks and captures the step graph; the second replays it.
            audios = [
                _waveform(runner.omni.generate(prompt, auk_sampling_params(nfe=NFE, cfg=2.0, seed=0))[0])
                for _ in range(2)
            ]

    # Every block linear of the released checkpoint was swapped, once.
    assert len(swapped) == 1 and swapped[0][0] == swapped[0][1] > 0, swapped
    expected = math.ceil(GEN_SECONDS * SAMPLE_RATE / HOP) * HOP
    for audio in audios:
        assert audio.size == expected, f"expected {expected} samples, got {audio.size}"
        assert np.isfinite(audio).all(), "non-finite samples"
        assert np.abs(audio).max() <= 1.0 + 1e-6, "audio exceeds full scale"
        assert float(np.sqrt(np.mean(audio**2))) > 1e-3, "audio is silent"
    # Same seed, same graph: the replay reproduces the captured request.
    np.testing.assert_allclose(audios[1], audios[0], atol=1e-4)
