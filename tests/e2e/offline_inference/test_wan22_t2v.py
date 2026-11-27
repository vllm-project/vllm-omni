# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient, OmniRunner
from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PROMPT = "Two anthropomorphic cats in boxing gear on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
# Cold ROCm starts can exceed 30 minutes. Keep the inner deadline five
# minutes below the merge job's 40-minute process deadline for cleanup.
WAN22_INIT_TIMEOUT_S = 35 * 60
WAN22_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "init_timeout": WAN22_INIT_TIMEOUT_S,
        "stage_init_timeout": WAN22_INIT_TIMEOUT_S,
    },
)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 1})
@pytest.mark.parametrize("omni_runner", [WAN22_RUNNER_PARAM], indirect=True)
def test_structured_diffusion_config_reaches_runtime(omni_runner) -> None:
    """The default WAN resolver launches the typed diffusion config itself."""
    omni = omni_runner.omni
    resolved_stages = omni.engine.stage_configs
    stage_clients = omni.engine.stage_clients
    assert len(resolved_stages) == 1
    assert len(stage_clients) == 1
    stage_config = resolved_stages[0]
    od_config = stage_clients[0].od_config

    assert isinstance(stage_config, VllmOmniDiffusionStageConfig)
    assert isinstance(od_config, OmniDiffusionConfig)
    assert stage_config.stage_id == 0
    assert stage_config.model_stage == "dit"
    assert stage_config.final_output_type == "video"

    assert omni.engine.stage_vllm_configs == [None]
    assert od_config.stage_id == 0
    assert od_config.model == MODEL
    assert od_config.model_class_name == "WanPipeline"
    assert od_config.output_type == "pil"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 1})
@pytest.mark.parametrize("omni_runner", [WAN22_RUNNER_PARAM], indirect=True)
def test_text_to_video_001(offline_client: OfflineOmniClient):
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
    offline_client.send_diffusion_request(request_config)


def _generate_ti2v_5b(backend: str) -> np.ndarray:
    model = os.environ.get("WAN_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    config: dict[str, object] = {"default": {"backend": "FLASH_ATTN"}}
    if backend == "SOL_ATTN":
        config["per_role"] = {"self": {"backend": backend, "sol_attn": {"dense_steps": 1}}}
    with OmniRunner(
        model,
        deploy_config="vllm_omni/deploy/wan2_2_ti2v.yaml",
        diffusion_attention_config=config,
    ) as runner:
        sampling = OmniDiffusionSamplingParams(
            width=832, height=480, num_frames=17, num_inference_steps=2, guidance_scale=5.0, seed=1024
        )
        response = OfflineOmniClient(runner).send_diffusion_request({"prompt": PROMPT, "sampling_params": sampling})
        frames = np.asarray((response.videos if response.videos is not None else response.images)[0])
        assert frames.shape[-4:] == (17, 480, 832, 3)
        assert np.isfinite(frames).all() and frames.max() > frames.min()
        return frames


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"npu": "A2"})
def test_ti2v_5b_sol_attn():
    from vllm_omni.diffusion.attention.backends.sol_attn import _sol_unavailable_reason

    # Cross the dense-to-Sol boundary in one short generation.
    if reason := _sol_unavailable_reason():
        pytest.skip(reason)
    _generate_ti2v_5b("SOL_ATTN")


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"npu": "A2"})
def test_ti2v_5b_sol_fallback_matches_flash():
    from vllm_omni.diffusion.attention.backends.sol_attn import _sol_unavailable_reason

    if _sol_unavailable_reason() is None:
        pytest.skip("requires an environment without the Sol fused operator")
    np.testing.assert_array_equal(_generate_ti2v_5b("SOL_ATTN"), _generate_ti2v_5b("FLASH_ATTN"))
