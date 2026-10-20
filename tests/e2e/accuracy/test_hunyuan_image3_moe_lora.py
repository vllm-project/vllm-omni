# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate HunyuanImage3 routed-expert LoRA with a synthetic PEFT adapter.

Check activation, suspension, reactivation and zero-scale output.
Run with --run-level full_model on four GPUs with at least 80 GiB each.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

# Gated MoE projections in upstream [w1, w2, w3] order.
_EXPERT_PROJECTIONS = ("gate_proj", "down_proj", "up_proj")


@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=4)
def test_hunyuan_image3_dit_moe_lora_generation(tmp_path: Path):
    # Set HUNYUAN_IMAGE3_MODEL to use local weights instead of the HF repository.
    model = os.environ.get("HUNYUAN_IMAGE3_MODEL", "tencent/HunyuanImage-3.0-Instruct")
    config_path = Path(model) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
    else:
        config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    hidden = config["hidden_size"]
    # moe_intermediate_size may be a list (per-layer); use the first entry.
    moe_inter = config.get("moe_intermediate_size") or config["intermediate_size"]
    if isinstance(moe_inter, list):
        moe_inter = moe_inter[0]
    num_experts = config["num_experts"]
    if isinstance(num_experts, list):
        num_experts = num_experts[0]
    rank = 8
    generator = torch.Generator().manual_seed(8182)

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    tensors = {}
    # Double "model" combines the PEFT prefix and the pipeline component name.
    for layer in (0, 1):
        for ei in range(num_experts):
            for proj in _EXPERT_PROJECTIONS:
                # gate_proj/up_proj: hidden -> moe_inter (W: [moe_inter, hidden])
                # down_proj:          moe_inter -> hidden (W: [hidden, moe_inter])
                if proj in ("gate_proj", "up_proj"):
                    a_rows, b_rows = rank, moe_inter
                    a_cols, b_cols = hidden, rank
                else:  # down_proj
                    a_rows, b_rows = rank, hidden
                    a_cols, b_cols = moe_inter, rank
                prefix = f"base_model.model.model.layers.{layer}.mlp.experts.{ei}.{proj}"
                tensors[f"{prefix}.lora_A.weight"] = torch.randn(a_rows, a_cols, generator=generator) * 0.02
                tensors[f"{prefix}.lora_B.weight"] = torch.randn(b_rows, b_cols, generator=generator) * 0.1
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": rank,
                "target_modules": list(_EXPERT_PROJECTIONS),
            }
        )
    )

    tp = 4
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "pipeline": "hunyuan_image3_dit",
                "async_chunk": False,
                "trust_remote_code": True,
                "stages": [
                    {
                        "stage_id": 0,
                        "devices": ",".join(map(str, range(tp))),
                        "max_num_seqs": 1,
                        "enforce_eager": True,
                        "trust_remote_code": True,
                        "parallel_config": {"tensor_parallel_size": tp},
                    }
                ],
                # Exercise expert parallelism on NPU.
                "platforms": {
                    "npu": {
                        "stages": [
                            {
                                "stage_id": 0,
                                "gpu_memory_utilization": 0.65,
                                "moe_backend": "auto",
                                "devices": ",".join(map(str, range(tp))),
                                "parallel_config": {
                                    "tensor_parallel_size": tp,
                                    "enable_expert_parallel": True,
                                },
                            }
                        ]
                    }
                },
            }
        )
    )
    request = LoRARequest(lora_name="hi3_moe", lora_int_id=8182, lora_path=str(adapter_dir))
    with OmniRunner(
        model,
        trust_remote_code=True,
        deploy_config=str(deploy),
        stage_init_timeout=1800,
        init_timeout=2400,
    ) as runner:

        def generate(label, lora_request=None, scale=1.0):
            outputs = runner.omni.generate(
                "A red ceramic teapot on a wooden table.",
                OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    seed=42,
                    num_inference_steps=2,
                    guidance_scale=1.0,
                    lora_request=lora_request,
                    lora_scale=scale,
                ),
            )
            image = outputs[0].images[0]
            assert image.size == (512, 512)
            image.save(tmp_path / f"{label}.png")
            return np.asarray(image).copy()

        baseline = generate("baseline")
        adapted = generate("adapted", request)
        restored = generate("restored")
        repeated = generate("repeated", request)
        zero_scale = generate("zero_scale", request, scale=0.0)
        # Verify that routed-expert LoRA changes the generated image.
        assert np.abs(adapted.astype(float) - baseline).mean() > 0.1
        np.testing.assert_array_equal(restored, baseline)
        np.testing.assert_array_equal(repeated, adapted)
        np.testing.assert_array_equal(zero_scale, baseline)
