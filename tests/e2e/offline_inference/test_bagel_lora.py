# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for BAGEL LoRA support (Stage 1 / DiT).

Validates that LoRA adapters are correctly loaded, applied with controllable
scale, and cleanly deactivated.  Uses a synthetic rank-1 adapter targeting the
first decoder layer's QKV projection.

Assertions:
  (a) LoRA at scale=1.0 visibly changes the output  (diff > 0.5)
  (b) scale=2.0 produces a larger delta than scale=1.0  (linearity)
  (c) The delta is bounded  (diff < 80, not corrupted)
  (d) Deactivating LoRA exactly restores the baseline  (diff == 0)
"""

import json
import os
from pathlib import Path

from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
BAGEL_STAGE_CONFIG = get_deploy_config_path("ci/bagel.yaml")
DEFAULT_PROMPT = "<|im_start|>A cute cat<|im_end|>"


# ---------------------------------------------------------------------------
# Helpers (reused from test_bagel_text2img.py patterns)
# ---------------------------------------------------------------------------


def _resolve_deploy_config(config_path: str, run_level: str) -> str:
    if run_level == "advanced_model":
        return modify_stage_config(
            config_path,
            deletes={
                "stages": {
                    0: ["load_format"],
                    1: ["load_format"],
                }
            },
        )
    return config_path


def _configure_sampling_params(omni: Omni, num_inference_steps: int = 10) -> list[OmniSamplingParams]:
    params_list = omni.default_sampling_params_list
    if len(params_list) > 1:
        params_list[1].num_inference_steps = num_inference_steps
        params_list[1].extra_args = {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
        }
    return params_list


def _extract_generated_image(omni_outputs: list[OmniRequestOutput]) -> Image.Image | None:
    for req_output in omni_outputs:
        if req_output.images:
            return req_output.images[0]
    return None


def _generate_bagel_image(omni: Omni) -> Image.Image:
    params_list = _configure_sampling_params(omni)
    params_list[1].lora_request = None
    outputs = list(
        omni.generate(
            prompts=[{"prompt": DEFAULT_PROMPT, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated"
    return img


def _generate_bagel_image_with_lora(
    omni: Omni,
    lora_request: LoRARequest,
    lora_scale: float = 1.0,
) -> Image.Image:
    params_list = _configure_sampling_params(omni)
    params_list[1].lora_request = lora_request
    params_list[1].lora_scale = lora_scale
    outputs = list(
        omni.generate(
            prompts=[{"prompt": DEFAULT_PROMPT, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated with LoRA"
    return img


# BAGEL uses GQA: hidden_size=3584, 28 Q heads, 4 KV heads, head_dim=128
# QKV packed dim = 28*128 + 4*128 + 4*128 = 3584 + 512 + 512 = 4608
_LORA_DIM = 3584
_LORA_QKV_DIM = 4608
_LORA_MODULE = "bagel.language_model.model.layers.0.self_attn.qkv_proj"
_LORA_RANK = 4


def _make_file_lora_request(adapter_dir: Path) -> LoRARequest:
    """Write synthetic adapter to disk and return a file-backed LoRARequest."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(42)
    lora_a = torch.randn((_LORA_RANK, _LORA_DIM), dtype=torch.float32, generator=gen) * 0.1
    lora_b = torch.randn((_LORA_QKV_DIM, _LORA_RANK), dtype=torch.float32, generator=gen) * 0.5
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


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_lora_scale_and_deactivation(run_level, tmp_path):
    """Validate LoRA effect, bounded perturbation, and clean deactivation."""
    config_path = _resolve_deploy_config(BAGEL_STAGE_CONFIG, run_level)
    with OmniRunner(MODEL, stage_configs_path=config_path) as runner:
        omni = runner.omni
        lora_request = _make_file_lora_request(tmp_path / "bagel_lora")

        # 1) Baseline (no LoRA)
        baseline = _generate_bagel_image(omni)

        # 2) LoRA with scale=1.0
        img_1x = _generate_bagel_image_with_lora(omni, lora_request, lora_scale=1.0)

        # 3) LoRA with scale=2.0
        img_2x = _generate_bagel_image_with_lora(omni, lora_request, lora_scale=2.0)

        # 4) No LoRA again (deactivation)
        restored = _generate_bagel_image(omni)

        baseline_arr = np.array(baseline, dtype=np.int16)
        img_1x_arr = np.array(img_1x, dtype=np.int16)
        img_2x_arr = np.array(img_2x, dtype=np.int16)
        restored_arr = np.array(restored, dtype=np.int16)

        diff_1x = np.abs(baseline_arr - img_1x_arr).mean()
        diff_2x = np.abs(baseline_arr - img_2x_arr).mean()
        diff_restored = np.abs(baseline_arr - restored_arr).mean()

        # (a) Adapter has visible effect at both scales
        assert diff_1x > 0.5, f"LoRA scale=1.0 had no visible effect: diff={diff_1x}"
        assert diff_2x > 0.5, f"LoRA scale=2.0 had no visible effect: diff={diff_2x}"

        # (b) Different scales produce different outputs
        assert not np.isclose(diff_1x, diff_2x, atol=1.0), (
            f"LoRA scale has no effect: diff_1x={diff_1x:.2f}, diff_2x={diff_2x:.2f}"
        )

        # (c) Output is not corrupted (scale=2.0 can produce ~2x the diff of scale=1.0)
        assert diff_1x < 80, f"LoRA output looks corrupted: diff_1x={diff_1x}"
        assert diff_2x < 120, f"LoRA output looks corrupted: diff_2x={diff_2x}"

        # (d) Deactivation fully restores base model
        assert diff_restored == 0.0, f"Base model not restored after LoRA deactivation: diff={diff_restored}"
