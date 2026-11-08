# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.config import LoadConfig
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.e2e.accuracy.helpers import assert_video_pixel_close
from tests.helpers.clean import cleanup_test_environment
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import MODEL_ID
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]

# The native PixelUMM decoder is not bitwise to the reference's BF16 kernels
# (stacked GEMMs, vLLM RMSNorm, per-tile attention). On real latents its error
# against an FP32 reference decode is within 1.11x of the reference's own BF16
# error (mean 2.8e-3, p99 2.6e-2, worst frame 32.8 dB), and native vs
# reference measured mean 2.7e-3, p99 2.4e-2, worst frame 39.6 dB. The bounds
# allow about twice that BF16 rounding noise; a window or tile bug is far larger.
VIDEO_TOLERANCE = {"mean_threshold": 6e-3, "p99_threshold": 5.5e-2, "frame_psnr_threshold": 32.0}

REFERENCE_FILES = (
    "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
    "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors",
)


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

    reference_root = Path(os.environ.get("HY_NOMAS_REFERENCE_MODEL", "tencent/Hy-Nomas"))
    if not reference_root.is_dir():
        reference_root = Path(
            download_weights_from_hf_specific(
                str(reference_root),
                cache_dir=None,
                allow_patterns=list(REFERENCE_FILES),
                revision="bac719a80c8d02f91c485fca804bb8221e149b70",
                require_all=True,
            )
        )
    text_encoder = os.environ.get("HY_NOMAS_REFERENCE_TEXT_ENCODER", "Qwen/Qwen2.5-VL-7B-Instruct")
    with initialize_config_dir(config_dir=str(Path(reference_config.__file__).parent), version_base="1.3"):
        config = compose(config_name="config", overrides=["+experiment=inference"])
    config.dit_weight = str(reference_root / REFERENCE_FILES[0])
    config.model.vae.checkpoint = str(reference_root / REFERENCE_FILES[1])
    config.model.vae.parallel_mode = "none"
    config.model.text_encoder.model_name = text_encoder
    config.model.dit.seg_ar_attention_backend = "flex"
    config.parallelism.sp_size = 1
    config.parallelism.shard_text_encoder = False
    device = get_local_device()
    custom_args = {}
    if "HY_NOMAS_TEXT_ENCODER" in os.environ:
        custom_args["text_encoder"] = os.environ["HY_NOMAS_TEXT_ENCODER"]
    omni_config = OmniDiffusionConfig(
        model=os.environ.get("HY_NOMAS_MODEL", MODEL_ID),
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        enforce_eager=True,
        custom_pipeline_args=custom_args,
    )
    reference = candidate = None
    try:
        reference = CPTCausalInferencePipeline(reference_config._flatten_cfg(config), device, config.dit_weight).eval()
        with set_default_torch_dtype(torch.bfloat16):
            candidate = HyNomasPipeline(od_config=omni_config)
        loader = DiffusersPipelineLoader(LoadConfig(), omni_config)
        candidate.load_weights(loader.get_all_weights(candidate))
        yield reference, candidate, device
    finally:
        if candidate is not None:
            candidate.disable_omni_model_cpu_offload()
        reference = candidate = None


@pytest.mark.parametrize("offload", [False, True], ids=["resident", "cpu_offload"])
@pytest.mark.parametrize("scenario", ["text", "segments", "camera"])
@pytest.mark.parametrize("steps", [4, 8])
@torch.no_grad()
def test_reference_decoded_video(pipelines, monkeypatch, scenario, offload, steps):
    reference, candidate, device = pipelines
    prompts = ["A forest beside a clear river, photorealistic, daylight."]
    boundaries = [0, 49]
    if scenario != "text":
        prompts.append("The camera moves along the river toward a wooden bridge.")
        boundaries = [0, 24, 49]
    camera = "pan_left" if scenario == "camera" else None
    camera_flags = [True, True] if camera else None
    expected_latents = []
    actual_latents = []
    decode = reference._decode_latents
    post_decode = candidate.post_decode

    def record_reference(value, num_frames):
        expected_latents.append(value.detach().cpu())
        return decode(value, num_frames)

    def record_candidate(state, **kwargs):
        actual_latents.append(state.latents.detach().cpu())
        return post_decode(state, **kwargs)

    monkeypatch.setattr(reference, "_decode_latents", record_reference)
    monkeypatch.setattr(candidate, "post_decode", record_candidate)
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
            num_inference_steps=steps,
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
        num_inference_steps=steps,
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
        # The DiT matches the reference exactly; only the native decoder's BF16 kernels differ.
        assert len(expected_latents) == 1
        torch.testing.assert_close(torch.cat(actual_latents, dim=2), expected_latents[0], rtol=0, atol=0)
        assert_video_pixel_close(
            model_name=f"Hy-Nomas {scenario} {steps} steps",
            actual=actual[0].transpose(0, 1),
            expected=expected[0].transpose(0, 1),
            **VIDEO_TOLERANCE,
        )
    finally:
        if offload:
            candidate.disable_omni_model_cpu_offload()
            candidate.to(device)
