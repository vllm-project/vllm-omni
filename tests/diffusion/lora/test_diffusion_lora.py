import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# This test is specific to Z-Image LoRA behavior. Keep it focused on a single
# model to reduce runtime and avoid extra downloads.
models = ["Tongyi-MAI/Z-Image-Turbo"]
omni_runner_params = [(model_name, None) for model_name in models]
# 3-tuple form lets us set max_loras=2 for the multi-LoRA composition test.
omni_runner_params_multi_lora = [(model_name, None, {"max_loras": 2}) for model_name in models]


def _extract_images(outputs: list[OmniRequestOutput]):
    if not outputs:
        raise ValueError("Empty outputs from Omni.generate()")
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")
    return req_out.images


def _write_zimage_lora(
    adapter_dir: Path,
    *,
    lora_b_value: float = 0.1,
    lora_b_slice: str = "q",
) -> str:
    """Write a fake Z-Image PEFT adapter to disk.

    Args:
        adapter_dir: Directory to write the adapter files.
        lora_b_value: Value to fill in the active lora_b slice.
        lora_b_slice: Which QKV slice to perturb ("q", "k", or "v").
    """
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
    slice_offsets = {"q": 0, "k": dim, "v": 2 * dim}
    offset = slice_offsets[lora_b_slice]
    lora_b[offset : offset + dim, 0] = lora_b_value

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


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_runner", omni_runner_params, indirect=True)
def test_diffusion_model(tmp_path: Path, omni_runner):
    runner = omni_runner
    m = runner.omni
    # high resolution may cause OOM on L4
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
        ),
    )
    images = _extract_images(outputs)

    assert len(images) == 1
    # check image size
    assert images[0].width == width
    assert images[0].height == height

    # Real LoRA E2E: generate again with a real on-disk PEFT adapter and
    # verify that output changes.
    from vllm_omni.lora.request import LoRARequest
    from vllm_omni.lora.utils import stable_lora_int_id

    lora_dir = _write_zimage_lora(tmp_path / "zimage_lora")
    lora_request = LoRARequest(
        lora_name="test",
        lora_int_id=stable_lora_int_id(lora_dir),
        lora_path=lora_dir,
    )
    outputs_lora = m.generate(
        prompt,
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_inference_steps=2,
            guidance_scale=0.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            num_outputs_per_prompt=1,
            lora_requests=[lora_request],
            lora_scales=[2.0],
        ),
    )
    images_lora = _extract_images(outputs_lora)
    assert len(images_lora) == 1
    assert images_lora[0].width == width
    assert images_lora[0].height == height

    import numpy as np

    diff = np.abs(np.array(images[0], dtype=np.int16) - np.array(images_lora[0], dtype=np.int16)).mean()
    assert diff > 0.0


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_runner", omni_runner_params_multi_lora, indirect=True)
def test_diffusion_multi_lora_composition(tmp_path: Path, omni_runner):
    """Test that composing two LoRA adapters produces different output than either alone."""
    m = omni_runner.omni
    from vllm_omni.lora.request import LoRARequest
    from vllm_omni.lora.utils import stable_lora_int_id

    height = 256
    width = 256
    prompt = "a photo of a cat sitting on a laptop keyboard"

    # Create two adapters that perturb different QKV slices
    lora_dir_a = _write_zimage_lora(tmp_path / "lora_a", lora_b_value=0.1, lora_b_slice="q")
    lora_dir_b = _write_zimage_lora(tmp_path / "lora_b", lora_b_value=0.1, lora_b_slice="k")

    req_a = LoRARequest("lora_a", stable_lora_int_id(lora_dir_a), lora_dir_a)
    req_b = LoRARequest("lora_b", stable_lora_int_id(lora_dir_b), lora_dir_b)

    def _gen(**lora_kwargs):
        return _extract_images(
            m.generate(
                prompt,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                    generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                    num_outputs_per_prompt=1,
                    **lora_kwargs,
                ),
            )
        )

    import numpy as np

    # Baseline: no LoRA
    img_base = np.array(_gen()[0], dtype=np.int16)

    # Single LoRA A
    img_a = np.array(_gen(lora_requests=[req_a], lora_scales=[2.0])[0], dtype=np.int16)

    # Single LoRA B
    img_b = np.array(_gen(lora_requests=[req_b], lora_scales=[2.0])[0], dtype=np.int16)

    # Composed: A + B
    img_ab = np.array(
        _gen(lora_requests=[req_a, req_b], lora_scales=[2.0, 2.0])[0],
        dtype=np.int16,
    )

    # All four outputs should differ from each other
    diff_base_a = np.abs(img_base - img_a).mean()
    diff_base_b = np.abs(img_base - img_b).mean()
    diff_base_ab = np.abs(img_base - img_ab).mean()
    diff_a_ab = np.abs(img_a - img_ab).mean()
    diff_b_ab = np.abs(img_b - img_ab).mean()

    assert diff_base_a > 0.0, "LoRA A should differ from baseline"
    assert diff_base_b > 0.0, "LoRA B should differ from baseline"
    assert diff_base_ab > 0.0, "Composed A+B should differ from baseline"
    assert diff_a_ab > 0.0, "Composed A+B should differ from A alone"
    assert diff_b_ab > 0.0, "Composed A+B should differ from B alone"
