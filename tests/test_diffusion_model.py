import pytest

from vllm_omni import Omni

models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(model_name: str):
    m = Omni(model=model_name)
    image = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=1024,
        width=1024,
        num_inference_steps=9,
    )
    image[0].save("z_image_output.png")
