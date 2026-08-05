# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Expanded end-to-end test for SenseNova-U1 in offline mode.

This test file primarily covers end-to-end tests for LoRA support.

Validates that LoRA adapters are correctly loaded, applied with controllable
scale, and cleanly deactivated. Uses a synthetic rank-4 adapter targeting the
first decoder layer's gen-path (``_mot_gen``) output projection, so the
perturbation is exercised during the denoising loop.

Assertions:
  (a) LoRA at scale=1.0 visibly changes the output  (diff > 0.5)
  (b) scale=2.0 produces a different delta than scale=1.0  (scale sensitivity)
  (c) The delta is bounded  (not corrupted)
  (d) Deactivating LoRA exactly restores the baseline  (diff == 0)

Observed on H100 (seed=42, 8 steps, 512x512): diff_1x~62.2, diff_2x~76.9,
diff_restored=0.0. Thresholds match test_bagel_expansion.py and hold with margin.
"""

import json
import os
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

MODEL = "SenseNova/SenseNova-U1-8B-MoT"
DEFAULT_PROMPT = "a photo of a cat sitting on a laptop keyboard"
SIZE = (512, 512)  # (width, height); small for fast E2E

# SenseNova-U1-8B-MoT hidden_size (Qwen3-based). o_proj is square (hidden x hidden)
# regardless of GQA, so this is the robust LoRA target. Verify against config.json.
_LORA_DIM = 4096
_LORA_MODULE = "language_model.model.layers.0.self_attn.o_proj_mot_gen"
_LORA_RANK = 4


# ---------------------------------------------------------------------------
# Helpers (reused from test_sensenova_u1_text2img.py patterns)
# ---------------------------------------------------------------------------


def _build_sampling_params(
    lora_request: LoRARequest | None = None,
    lora_scale: float = 1.0,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=SIZE[1],
        width=SIZE[0],
        seed=42,
        num_inference_steps=8,
        extra_args={
            "cfg_scale": 4.0,
            "cfg_norm": "none",
            "timestep_shift": 3.0,
            "cfg_interval": (0.0, 1.0),
            "batch_size": 1,
            "think": False,
            "t_eps": 0.02,
        },
        lora_request=lora_request,
        lora_scale=lora_scale,
    )


def _extract_generated_image(omni_outputs: list) -> Image.Image | None:
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
    return None


def _generate(omni: Omni, lora_request: LoRARequest | None = None, lora_scale: float = 1.0) -> Image.Image:
    outputs = list(
        omni.generate(
            prompts={"prompt": DEFAULT_PROMPT, "modalities": ["image"]},
            sampling_params_list=_build_sampling_params(lora_request, lora_scale),
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated"
    return img


def _make_file_lora_request(adapter_dir: Path) -> LoRARequest:
    """Write synthetic adapter to disk and return a file-backed LoRARequest."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(42)
    lora_a = torch.randn((_LORA_RANK, _LORA_DIM), dtype=torch.float32, generator=gen) * 0.1
    lora_b = torch.randn((_LORA_DIM, _LORA_RANK), dtype=torch.float32, generator=gen) * 0.5
    save_file(
        {
            f"base_model.model.{_LORA_MODULE}.lora_A.weight": lora_a,
            f"base_model.model.{_LORA_MODULE}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": _LORA_RANK, "lora_alpha": _LORA_RANK, "target_modules": [_LORA_MODULE]}),
        encoding="utf-8",
    )
    lora_dir = str(adapter_dir)
    return LoRARequest(lora_name="test_file", lora_int_id=stable_lora_int_id(lora_dir), lora_path=lora_dir)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_lora_scale_and_deactivation(tmp_path) -> None:
    """Validate LoRA effect, scale sensitivity, bounded perturbation, and clean deactivation."""
    lora_request = _make_file_lora_request(tmp_path / "sensenova_lora")

    with OmniRunner(MODEL, stage_configs_path=None) as runner:
        omni = runner.omni
        baseline = _generate(omni)
        img_1x = _generate(omni, lora_request, lora_scale=1.0)
        img_2x = _generate(omni, lora_request, lora_scale=2.0)
        restored = _generate(omni)

    baseline_arr = np.array(baseline, dtype=np.int16)
    diff_1x = np.abs(baseline_arr - np.array(img_1x, dtype=np.int16)).mean()
    diff_2x = np.abs(baseline_arr - np.array(img_2x, dtype=np.int16)).mean()
    diff_restored = np.abs(baseline_arr - np.array(restored, dtype=np.int16)).mean()

    # (a) Adapter has visible effect at both scales
    assert diff_1x > 0.5, f"LoRA scale=1.0 had no visible effect: diff={diff_1x}"
    assert diff_2x > 0.5, f"LoRA scale=2.0 had no visible effect: diff={diff_2x}"

    # (b) Different scales produce different outputs
    assert not np.isclose(diff_1x, diff_2x, atol=1.0), (
        f"LoRA scale has no effect: diff_1x={diff_1x:.2f}, diff_2x={diff_2x:.2f}"
    )

    # (c) Output is not corrupted
    assert diff_1x < 80, f"LoRA output looks corrupted: diff_1x={diff_1x}"
    assert diff_2x < 120, f"LoRA output looks corrupted: diff_2x={diff_2x}"

    # (d) Deactivation fully restores base model
    assert diff_restored == 0.0, f"Base model not restored after LoRA deactivation: diff={diff_restored}"
