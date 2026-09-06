# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end LoRA inference test for Cosmos3-Nano."""

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

MODEL = "nvidia/Cosmos3-Nano"
PROMPT = "A small warehouse robot moves a blue box across a clean floor."
SIZE = (256, 256)
LORA_DIM = 4096
LORA_RANK = 4
LORA_MODULE = "transformer.gen_layers.0.cross_attention.to_out"


def _build_sampling_params(
    lora_request: LoRARequest | None = None,
    lora_scale: float = 1.0,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=SIZE[1],
        width=SIZE[0],
        seed=42,
        num_inference_steps=2,
        guidance_scale=1.0,
        extra_args={
            "flow_shift": 3.0,
            "guardrails": False,
        },
        lora_request=lora_request,
        lora_scale=lora_scale,
    )


def _generate(
    omni: Omni,
    lora_request: LoRARequest | None = None,
    lora_scale: float = 1.0,
) -> Image.Image:
    outputs = omni.generate(
        prompts={
            "prompt": PROMPT,
            "modalities": ["image"],
        },
        sampling_params_list=_build_sampling_params(lora_request, lora_scale),
        use_tqdm=False,
    )
    for output in outputs:
        if output.images:
            return output.images[0]
    raise AssertionError("No image generated")


def _make_lora_request(adapter_dir: Path) -> LoRARequest:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    lora_a = torch.randn((LORA_RANK, LORA_DIM), dtype=torch.float32, generator=generator) * 0.1
    lora_b = torch.randn((LORA_DIM, LORA_RANK), dtype=torch.float32, generator=generator) * 0.5
    save_file(
        {
            f"base_model.model.{LORA_MODULE}.lora_A.weight": lora_a,
            f"base_model.model.{LORA_MODULE}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": LORA_RANK,
                "lora_alpha": LORA_RANK,
                "target_modules": [LORA_MODULE],
            }
        ),
        encoding="utf-8",
    )
    lora_dir = str(adapter_dir)
    return LoRARequest(
        lora_name="cosmos3-test",
        lora_int_id=stable_lora_int_id(lora_dir),
        lora_path=lora_dir,
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize(
    "omni_runner",
    [
        (
            MODEL,
            None,
            {
                "model_class_name": "Cosmos3OmniDiffusersPipeline",
                "model_config": {"guardrails": False},
            },
        )
    ],
    indirect=True,
)
def test_cosmos3_lora_scale_and_deactivation(omni_runner: OmniRunner, tmp_path: Path) -> None:
    lora_request = _make_lora_request(tmp_path / "cosmos3_lora")
    omni = omni_runner.omni
    baseline = _generate(omni)
    image_1x = _generate(omni, lora_request, lora_scale=1.0)
    image_2x = _generate(omni, lora_request, lora_scale=2.0)
    restored = _generate(omni)

    baseline_array = np.asarray(baseline, dtype=np.int16)
    image_1x_array = np.asarray(image_1x, dtype=np.int16)
    image_2x_array = np.asarray(image_2x, dtype=np.int16)
    restored_array = np.asarray(restored, dtype=np.int16)

    diff_1x = np.abs(baseline_array - image_1x_array).mean()
    diff_2x = np.abs(baseline_array - image_2x_array).mean()
    diff_restored = np.abs(baseline_array - restored_array).mean()

    assert diff_1x > 0.5, f"LoRA scale=1.0 had no visible effect: diff={diff_1x}"
    assert diff_2x > 0.5, f"LoRA scale=2.0 had no visible effect: diff={diff_2x}"
    assert not np.isclose(diff_1x, diff_2x, atol=1.0), (
        f"LoRA scale had no effect: diff_1x={diff_1x:.2f}, diff_2x={diff_2x:.2f}"
    )
    assert diff_restored < 5.0, f"LoRA did not deactivate cleanly: diff_restored={diff_restored:.2f}"
