# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
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
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
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


@pytest.mark.parametrize("offload", [False, True], ids=["resident", "cpu_offload"])
@torch.no_grad()
def test_ar_continuation_latents_match_reference(pipelines, monkeypatch, offload):
    reference, candidate, device = pipelines
    config = deepcopy(reference.args)
    OmegaConf.set_struct(config, False)
    config.inference.rollout_type = "chunk_ar"
    config.inference_chunk_size = 16
    config.inference.interactive_noise_max_seconds = 32 / 6
    config.inference.interactive_noise_hard_max_seconds = 32 / 6
    monkeypatch.setattr(reference, "args", config)
    reference.kv_cache = None
    candidate.initialize_ar({"ar_height": 384, "ar_width": 640, "ar_max_latent_frames": 32, "ar_max_segments": 4})
    spec = candidate.ar_diffusion_kv_cache_spec()
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True),
        num_layers=spec.num_layers,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=torch.bfloat16,
        block_size=spec.segmented.block_size,
        max_model_len=spec.segmented.initial_staging_tokens,
        available_bytes=16 * 1024**3,
        kv_branches=spec.kv_branches,
        session_capacity=1,
        device=device,
        segmented=spec.segmented,
    )
    owner = ARDiffusionKVState(
        cache, "ar-parity", {"cond": cache.begin_request("ar-parity")}, num_layers=spec.num_layers
    )
    expected_latents, actual_latents = [], []
    reference_decode, candidate_decode = reference._decode_latents, candidate.post_decode

    def record_reference(latents, num_frames):
        expected_latents.append(latents.detach().cpu().clone())
        return reference_decode(latents, num_frames)

    def record_candidate(state, **kwargs):
        actual_latents.append(state.latents.detach().cpu().clone())
        return candidate_decode(state, **kwargs)

    monkeypatch.setattr(reference, "_decode_latents", record_reference)
    monkeypatch.setattr(candidate, "post_decode", record_candidate)
    prompts = ["A forest beside a clear river, daylight.", "The camera approaches a wooden bridge beside the river."]
    outputs = []
    if offload:
        candidate.enable_omni_model_cpu_offload(device=device)
    try:
        for index, prompt in enumerate(prompts):
            with sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                reference.inference(
                    prompts=[prompt],
                    num_frames=49,
                    height=384,
                    width=640,
                    guidance_scale=1,
                    num_inference_steps=8,
                    flow_shift=5,
                    seed=42,
                    segment_frame_boundaries=[0, 49],
                    camera_trajectory="pan_left",
                    preserve_kv_cache=True,
                )
            tick = ARDiffusionTickRequest("ar-parity", f"turn-{index}", index, prompt=prompt, close_session=index == 1)
            params = OmniDiffusionSamplingParams(
                num_frames=49,
                height=384,
                width=640,
                num_inference_steps=8,
                guidance_scale=1,
                seed=42,
                extra_args={**tick.to_extra_args(), "camera_trajectory": "pan_left"},
            )
            with candidate.bind_ar_diffusion_state("ar-parity", owner):
                output = candidate(
                    DiffusionRequestBatch(
                        [OmniDiffusionRequest(prompt=prompt, sampling_params=params, request_id=tick.request_id)]
                    )
                )
            outputs.append(output.output["payload"]["video"])
            torch.testing.assert_close(actual_latents[-1], expected_latents[-1], rtol=0, atol=0)
        assert len(outputs[0]) > 0
        assert sum(len(video) for video in outputs) == 98
        assert output.output["metadata"]["hy_nomas"]["withheld_frames"] == 0
        reference_stream = reference.vae.create_streaming_decode_session(to_cpu=True, retain_history=False)
        expected_pixels = []
        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        ):
            for index, latents in enumerate(expected_latents):
                reference_stream.final_segment = index + 1 == len(expected_latents)
                pixels = reference_stream.push(
                    reference._prepare_vae_decode_latents(latents.to(device)), pixel_frames=49
                )
                if pixels.numel():
                    expected_pixels.append(pixels)
            tail = reference_stream.finish()
            if tail.numel():
                expected_pixels.append(tail)
        expected = reference._postprocess_vae_video(torch.cat(expected_pixels, 2))
        expected = expected[0].permute(1, 2, 3, 0).float().mul(255).round().to(torch.uint8).numpy()
        np.testing.assert_array_equal(np.concatenate(outputs), expected)
    finally:
        owner.close()
        candidate.close_ar_diffusion_session("ar-parity")
        reference.kv_cache = None
        reference._reset_interactive_noise_state()
        if offload:
            candidate.disable_omni_model_cpu_offload()
            candidate.to(device)
