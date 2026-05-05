# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Run vLLM-Omni with diffusers backend, then run diffusers directly. Compare the output similarity.
Also, do a quick performance check on the end-to-end latency.
The latency check is not put under /tests/dfx/perf/ because this is not a complete performance check to be monitored.
"""

import base64
import gc
import io
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import requests
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import export_to_video  # pyright: ignore[reportPrivateImportUsage]
from PIL import Image

from benchmarks.accuracy.common import pil_to_base64
from tests.e2e.accuracy.helpers import (
    apply_ftfy_mock,
    assert_similarity,
    env_to_apply_ftfy_mock_in_subproc,
    model_output_dir,
)
from tests.e2e.accuracy.wan22_i2v.test_wan22_i2v_video_similarity import (
    _parse_psnr_score,
    _parse_ssim_score,
    _run_ffmpeg_similarity,
)
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 30.0

VIDEO_PROMPT = "The bear in the image dances happily"
VIDEO_WIDTH = 832
VIDEO_HEIGHT = 480
VIDEO_NUM_INFERENCE_STEPS = 10
NUM_FRAMES = 9
FPS = 4
GUIDANCE_SCALE = 4.0
GUIDANCE_SCALE_2 = 1.0
BOUNDARY_RATIO = 0.875
FLOW_SHIFT = 12.0
VIDEO_SSIM_THRESHOLD = 0.95
VIDEO_PSNR_THRESHOLD = 30.0


def _run_vllm_omni_wan22_i2v(
    *,
    model: str,
    output_path: Path,
    conditioning_image: Image.Image,
) -> float:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "400",
        "--init-timeout",
        "900",
        "--diffusion-load-format",
        "diffusers",
        "--boundary-ratio",
        str(BOUNDARY_RATIO),
        "--flow-shift",
        str(FLOW_SHIFT),
    ]
    form_data = {
        "prompt": VIDEO_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": VIDEO_HEIGHT,
        "width": VIDEO_WIDTH,
        "num_inference_steps": VIDEO_NUM_INFERENCE_STEPS,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_scale_2": GUIDANCE_SCALE_2,
        "seed": SEED,
    }
    with OmniServer(model, server_args, env_dict=env_to_apply_ftfy_mock_in_subproc(), use_omni=True) as omni_server:
        client = OpenAIClientHandler(
            host=omni_server.host,
            port=omni_server.port,
            run_level="full_model",
        )
        request_config = {
            "model": omni_server.model,
            "form_data": form_data,
            "image_reference": f"data:image/png;base64,{pil_to_base64(conditioning_image, 'png')}",
        }
        result = client.send_video_diffusion_request(request_config)[0]
        video_bytes = result.videos[0]  # pyright: ignore[reportOptionalSubscript] # Guaranteed not None
        output_path.write_bytes(video_bytes)
        return result.e2e_latency  # pyright: ignore[reportReturnType] # Guaranteed not None


def _run_diffusers_wan22_i2v(*, model: str, output_path: Path, conditioning_image: Image.Image) -> float:
    from diffusers import WanImageToVideoPipeline  # pyright: ignore[reportPrivateImportUsage]
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    run_pre_test_cleanup(enable_force=True)
    apply_ftfy_mock()
    pipe: WanImageToVideoPipeline | None = None
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            boundary_ratio=BOUNDARY_RATIO,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)
        pipe.to("cuda")

        _diffusers_dummy_run(pipe)

        generator = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            start_time = time.perf_counter()
            result = pipe(
                image=conditioning_image,
                prompt=VIDEO_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                height=VIDEO_HEIGHT,
                width=VIDEO_WIDTH,
                num_inference_steps=VIDEO_NUM_INFERENCE_STEPS,
                num_frames=NUM_FRAMES,
                guidance_scale=GUIDANCE_SCALE,
                guidance_scale_2=GUIDANCE_SCALE_2,
                generator=generator,
            )
            end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        frames = result.frames[0]  # pyright: ignore[reportAttributeAccessIssue]
        export_to_video(frames, str(output_path), fps=FPS)  # pyright: ignore[reportArgumentType]
        return e2e_latency
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()
        run_post_test_cleanup(enable_force=True)


def _run_vllm_omni_qwen_image(*, model: str, output_path: Path) -> tuple[Image.Image, float]:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "400",
        "--init-timeout",
        "900",
        "--diffusion-load-format",
        "diffusers",
        # "--enable-diffusion-pipeline-profiler",
    ]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        start_time = time.perf_counter()
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": TRUE_CFG_SCALE,
                "seed": SEED,
            },
            timeout=600,
        )
        end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image, e2e_latency


def _run_diffusers_qwen_image(*, model: str, output_path: Path) -> tuple[Image.Image, float]:
    run_pre_test_cleanup(enable_force=True)
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

        _diffusers_dummy_run(pipe)

        generator = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            start_time = time.perf_counter()
            result = pipe(  # pyright: ignore[reportCallIssue]
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=NUM_INFERENCE_STEPS,
                true_cfg_scale=TRUE_CFG_SCALE,
                generator=generator,
            )
            end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        output_image = result.images[0].convert("RGB")
        output_image.save(output_path)
        return output_image, e2e_latency
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()
        run_post_test_cleanup(enable_force=True)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen-Image"])
def test_diffusers_backend_t2i_matches_diffusers(model_id: str, accuracy_artifact_root: Path) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, model_id + "-diffusers-backend")

    vllm_output, vllm_latency = _run_vllm_omni_qwen_image(model=model_id, output_path=output_dir / "vllm_omni.png")
    diffusers_output, diffusers_latency = _run_diffusers_qwen_image(
        model=model_id, output_path=output_dir / "diffusers.png"
    )
    vllm_latency = vllm_latency * 1000
    diffusers_latency = diffusers_latency * 1000
    latency_threshold_factor = 0.2
    latency_threshold = diffusers_latency * (1 + latency_threshold_factor)
    print(f"{model_id} latency metrics:")
    print(
        f"  Latency={vllm_latency:.2f}ms, threshold<={latency_threshold:.2f}ms, diffusers latency={diffusers_latency:.2f}ms, lower is better"
    )

    assert_similarity(
        model_name=model_id,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
    assert vllm_latency <= latency_threshold, (
        f"VLLM latency ({vllm_latency:.2f}ms) is greater than {latency_threshold_factor * 100}% more than Diffusers latency ({diffusers_latency:.2f}ms)."
    )


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("model_id", ["Wan-AI/Wan2.2-I2V-A14B-Diffusers"])
def test_diffusers_backend_i2v_matches_diffusers(
    model_id: str,
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, model_id + "-diffusers-backend")

    resized_image = qwen_bear_image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)

    vllm_path = output_dir / "vllm_omni.mp4"
    vllm_latency = _run_vllm_omni_wan22_i2v(model=model_id, output_path=vllm_path, conditioning_image=resized_image)
    vllm_latency = vllm_latency * 1000

    diffusers_path = output_dir / "diffusers.mp4"
    diffusers_latency = _run_diffusers_wan22_i2v(
        model=model_id, output_path=diffusers_path, conditioning_image=resized_image
    )
    diffusers_latency = diffusers_latency * 1000
    latency_threshold_factor = 0.3
    latency_threshold = diffusers_latency * (1 + latency_threshold_factor)

    ssim_output = _run_ffmpeg_similarity("ssim", vllm_path, diffusers_path)
    psnr_output = _run_ffmpeg_similarity("psnr", vllm_path, diffusers_path)
    ssim_score = _parse_ssim_score(ssim_output)
    psnr_score = _parse_psnr_score(psnr_output)
    print(f"{model_id} latency metrics:")
    print(
        f"  Latency={vllm_latency:.2f}ms, threshold<={latency_threshold:.2f}ms, diffusers latency={diffusers_latency:.2f}ms, lower is better"
    )
    print(f"{model_id} similarity metrics:")
    print(f"  SSIM: value={ssim_score:.6f}, threshold>={VIDEO_SSIM_THRESHOLD:.6f}, range=[-1, 1], higher is better")
    print(
        f"  PSNR: value={psnr_score:.6f} dB, threshold>={VIDEO_PSNR_THRESHOLD:.6f} dB, range=[0, +inf), higher is better"
    )

    assert ssim_score >= VIDEO_SSIM_THRESHOLD, (
        f"SSIM below threshold for {model_id}: got {ssim_score:.6f}, expected >= {VIDEO_SSIM_THRESHOLD:.6f}."
    )
    assert psnr_score >= VIDEO_PSNR_THRESHOLD, (
        f"PSNR below threshold for {model_id}: got {psnr_score:.6f}, expected >= {VIDEO_PSNR_THRESHOLD:.6f}."
    )
    assert vllm_latency <= latency_threshold, (
        f"VLLM latency ({vllm_latency:.2f}ms) is greater than {latency_threshold_factor * 100}% more than Diffusers latency ({diffusers_latency:.2f}ms)."
    )


def _diffusers_dummy_run(pipe: DiffusionPipeline) -> None:
    from vllm_omni.diffusion.diffusion_engine import supports_multimodal_input

    supports_image_input, supports_audio_input = supports_multimodal_input(
        SimpleNamespace(
            diffusion_load_format="diffusers",
            diffusers_pipeline_cls=pipe.__class__,
            model_class_name="DiffusersAdapterPipeline",
        ),  # pyright: ignore[reportArgumentType]
    )
    height = 512
    width = 512
    kwargs: dict[str, Any] = {
        "prompt": "dummy run",
        "height": height,
        "width": width,
        "num_inference_steps": 1,
    }
    if supports_image_input:
        # Provide a dummy image input if the model supports it
        dummy_image = Image.new("RGB", (width, height))
        kwargs["image"] = dummy_image
    if supports_audio_input:
        audio_sr = 16000
        dummy_audio = np.random.randn(audio_sr * 2).astype(np.float32)
        kwargs["audio"] = dummy_audio
    with torch.inference_mode():
        pipe(**kwargs)  # pyright: ignore[reportCallIssue]
