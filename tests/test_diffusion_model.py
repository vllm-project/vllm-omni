import pytest

from vllm_omni import Omni

models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(model_name: str):
    m = Omni(model=model_name)
    height = 256
    width = 256
    image = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=9,
    )
    # check image size
    assert image[0].width == width
    assert image[0].height == height
    image[0].save("z_image_output.png")
