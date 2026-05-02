import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# Match unprefixed HF id even when MODEL_PREFIX is set (omni_runner resolves full path).
_QWEN_IMAGE_RANDOM_ID = "riverclouds/qwen_image_random"


def _is_qwen_image_random(model_path: str) -> bool:
    return model_path.rstrip("/").endswith(_QWEN_IMAGE_RANDOM_ID)


models = ["Tongyi-MAI/Z-Image-Turbo", "riverclouds/qwen_image_random"]

# Modelscope can't find riverclouds/qwen_image_random
# TODO: When NPU support is ready, remove this branch.
if current_omni_platform.is_npu():
    models = ["Tongyi-MAI/Z-Image-Turbo", "Qwen/Qwen-Image"]

# omni_runner expects (model, stage_configs_path); single-stage diffusion has no YAML.
test_params = [(m, None) for m in models]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 1, "rocm": 1, "xpu": 2})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_diffusion_model(omni_runner, run_level):
    resolved = omni_runner.model_name
    if run_level == "core_model" and not _is_qwen_image_random(resolved):
        pytest.skip()

    if run_level == "advanced_model" and _is_qwen_image_random(resolved):
        pytest.skip()

    # high resolution may cause OOM on L4
    height = 256
    width = 256
    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
        num_outputs_per_prompt=2,
    )

    # OmniRunner.generate() is typed for list[TextPrompt]; diffusion uses Omni.generate(str, ...).
    outputs = omni_runner.omni.generate(
        "a photo of a cat sitting on a laptop keyboard",
        sampling,
    )

    # Extract images from request_output['images']
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images

    assert len(images) == 2
    # check image size
    assert images[0].width == width
    assert images[0].height == height
    images[0].save("image_output.png")
