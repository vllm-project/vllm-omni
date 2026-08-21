import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.lora.types import registered_lora_request
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL = "Tongyi-MAI/Z-Image-Turbo"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_diffusion_model(tmp_path: Path):
    def _extract_images(outputs: list[OmniRequestOutput]):
        if not outputs:
            raise ValueError("Empty outputs from Omni.generate()")
        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        if not isinstance(first_output, OmniRequestOutput) or not first_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            raise ValueError("Invalid request_output structure or missing 'images' key")
        return req_out.images

    def _write_zimage_lora(adapter_dir: Path) -> str:
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Z-Image transformer uses dim=3840 by default (see ZImageTransformer2DModel).
        dim = 3840
        module_name = "transformer.layers.0.attention.to_qkv"
        rank = 1
        lora_a = torch.zeros((rank, dim), dtype=torch.float32)
        lora_a[0, 0] = 1.0

        # QKVParallelLinear packs (Q, K, V). With tp=1 and n_kv_heads==n_heads in Z-Image,
        # each slice is `dim`, so total out dim is `3 * dim`.
        lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
        # Apply a visible delta to the Q slice only to keep the perturbation bounded.
        lora_b[:dim, 0] = 0.1

        save_file(
            {
                f"base_model.model.{module_name}.lora_A.weight": lora_a,
                f"base_model.model.{module_name}.lora_B.weight": lora_b,
            },
            str(adapter_dir / "adapter_model.safetensors"),
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": rank,
                    "lora_alpha": rank,
                    "target_modules": [module_name],
                }
            ),
            encoding="utf-8",
        )
        return str(adapter_dir)

    lora_dir = _write_zimage_lora(tmp_path / "zimage_lora")
    with OmniRunner(MODEL, dynamic_lora=[json.dumps({"path": lora_dir, "name": "test"})]) as runner:
        m = runner.omni
        # High resolution may cause OOM on L4.
        height = 256
        width = 256
        prompt = "a photo of a cat sitting on a laptop keyboard"

        outputs = m.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,
                lora_request=(),
                lora_scale=(),
            ),
        )
        images = _extract_images(outputs)

        assert len(images) == 1
        # check image size
        assert images[0].width == width
        assert images[0].height == height

        # Generate again with the startup-registered adapter and verify that
        # request activation changes the output.
        lora_request = registered_lora_request("test")
        outputs_lora = m.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,
                lora_request=lora_request,
                lora_scale=2.0,
            ),
        )
        images_lora = _extract_images(outputs_lora)
        assert len(images_lora) == 1
        assert images_lora[0].width == width
        assert images_lora[0].height == height

        import numpy as np

        diff = np.abs(np.array(images[0], dtype=np.int16) - np.array(images_lora[0], dtype=np.int16)).mean()
        assert diff > 0.0
