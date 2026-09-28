# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HI3 DiT generation with an on-disk PEFT adapter (issue #6411).

Run with --run-level full_model on four GPUs with at least 80 GiB each.
"""

from __future__ import annotations

import json
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


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=4)
def test_hunyuan_image3_dit_lora_generation(tmp_path: Path):
    model = "tencent/HunyuanImage-3.0-Instruct"
    config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["attention_head_dim"])
    rows = (heads + 2 * kv_heads) * head_dim
    rank = 8
    generator = torch.Generator().manual_seed(6411)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    tensors = {}
    # Exercise both affected projections in several layers, with different
    # updates for every interleaved Q/K/V row, rather than a constant delta
    # that would be invariant under a row permutation.
    for layer in (0, 1):
        for projection, out_dim in (("qkv_proj", rows), ("o_proj", hidden)):
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden, generator=generator) * 0.02
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(out_dim, rank, generator=generator) * 0.1
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": ["qkv_proj", "o_proj"]})
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
            }
        )
    )
    request = LoRARequest(lora_name="hi3", lora_int_id=6411, lora_path=str(adapter_dir))
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
        assert np.abs(adapted.astype(float) - baseline).mean() > 0.1
        np.testing.assert_array_equal(restored, baseline)
        np.testing.assert_array_equal(repeated, adapted)
        np.testing.assert_array_equal(zero_scale, baseline)
