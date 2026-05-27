import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PROMPT = "Two anthropomorphic cats in boxing gear on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_text_to_video_001(omni_runner_handler: OmniRunnerHandler):
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_frames=8,
        fps=8,
        num_inference_steps=2,
        guidance_scale=4.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "sampling_params": sampling,
    }
    omni_runner_handler.send_diffusion_request(request_config)
