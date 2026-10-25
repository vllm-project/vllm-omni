# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.clean import cleanup_test_environment
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import DECODER_FILE, DIT_FILE, MODEL_ID, REQUIRED_FILES
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]


@pytest.fixture(scope="module")
def model_cleanup():
    cleanup_test_environment()
    try:
        yield
    finally:
        # Collect offload-hook cycles after the pipeline fixture releases its references.
        cleanup_test_environment()


@pytest.fixture(scope="module")
def pipelines(model_cleanup):
    from configs import config as reference_config
    from hydra import compose, initialize_config_dir
    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline

    root = Path(os.environ.get("HY_NOMAS_MODEL", MODEL_ID))
    if not root.is_dir():
        root = Path(
            download_weights_from_hf_specific(
                str(root),
                cache_dir=None,
                allow_patterns=list(REQUIRED_FILES),
                require_all=True,
            )
        )
    text_encoder = os.environ.get("HY_NOMAS_TEXT_ENCODER", "Qwen/Qwen2.5-VL-7B-Instruct")
    with initialize_config_dir(config_dir=str(Path(reference_config.__file__).parent), version_base="1.3"):
        config = compose(config_name="config", overrides=["+experiment=inference"])
    config.dit_weight = str(root / DIT_FILE)
    config.model.vae.checkpoint = str(root / DECODER_FILE)
    config.model.vae.parallel_mode = "none"
    config.model.text_encoder.model_name = text_encoder
    config.model.dit.seg_ar_attention_backend = "flex"
    config.parallelism.sp_size = 1
    config.parallelism.shard_text_encoder = False
    device = get_local_device()
    omni_config = OmniDiffusionConfig(
        model=str(root),
        dtype=torch.bfloat16,
        enforce_eager=True,
        custom_pipeline_args={"text_encoder": text_encoder},
    )
    reference = candidate = None
    try:
        reference = CPTCausalInferencePipeline(reference_config._flatten_cfg(config), device, config.dit_weight).eval()
        with set_default_torch_dtype(torch.bfloat16):
            candidate = HyNomasPipeline(od_config=omni_config)
        yield reference, candidate, device
    finally:
        if candidate is not None:
            candidate.disable_omni_model_cpu_offload()
        reference = candidate = None


@pytest.mark.parametrize("offload", [False, True], ids=["resident", "cpu_offload"])
@pytest.mark.parametrize("scenario", ["text", "segments", "camera"])
@torch.no_grad()
def test_reference_decoded_video(pipelines, scenario, offload):
    reference, candidate, device = pipelines
    prompts = ["A forest beside a clear river, photorealistic, daylight."]
    boundaries = [0, 49]
    if scenario != "text":
        prompts.append("The camera moves along the river toward a wooden bridge.")
        boundaries = [0, 24, 49]
    camera = "pan_left" if scenario == "camera" else None
    camera_flags = [True, True] if camera else None
    # pytest has imported vLLM; restore the SDPA backends used by a standalone reference process.
    with sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        expected = reference.inference(
            prompts=prompts,
            num_frames=49,
            height=384,
            width=640,
            guidance_scale=1.0,
            num_inference_steps=8,
            flow_shift=5.0,
            seed=42,
            segment_frame_boundaries=boundaries,
            camera_trajectory=camera,
            segment_has_camera=camera_flags,
            preserve_kv_cache=False,
        ).cpu()
    params = OmniDiffusionSamplingParams(
        num_frames=49,
        height=384,
        width=640,
        num_inference_steps=8,
        guidance_scale=1.0,
        seed=42,
        extra_args={
            "segment_frame_boundaries": boundaries,
            "camera_trajectory": camera,
            "segment_has_camera": camera_flags,
        },
    )
    if offload:
        candidate.enable_omni_model_cpu_offload(device=device)
    try:
        output = candidate(
            DiffusionRequestBatch(
                [
                    OmniDiffusionRequest(
                        prompt="|||".join(prompts), sampling_params=params, request_id=f"parity-{scenario}"
                    )
                ]
            )
        )
        actual = output.media.video.tensor.cpu()
        assert actual.shape == expected.shape == (1, 3, 49, 384, 640)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        if offload:
            candidate.disable_omni_model_cpu_offload()
            candidate.to(device)
