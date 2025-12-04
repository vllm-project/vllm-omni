import pytest
import torch

from vllm_omni import Omni

models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(model_name: str):
    m = Omni(model=model_name)
    # high resolution may cause OOM on L4
    height = 256
    width = 256
    images = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=2,
    )
    assert len(images) == 2
    # check image size
    assert images[0].width == width
    assert images[0].height == height
    images[0].save("z_image_output.png")
