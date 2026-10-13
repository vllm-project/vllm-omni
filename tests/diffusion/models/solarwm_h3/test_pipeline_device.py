# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in generation with the local published SolarWM-H3 base and Stage2 EMA.

Run on a GPU with enough memory for the unquantized model and conditioners:
    VLLM_TEST_SOLARWM_H3_MODEL=/models/SolarWM-H3-33B \
        pytest tests/diffusion/models/solarwm_h3/test_pipeline_device.py -q
No model assets are downloaded by this test.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


def _published_pipeline_worker(rank: int, model_root: str, rendezvous: str) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    config = OmniDiffusionConfig(
        model=model_root,
        model_class_name="SolarWMH3Pipeline",
        dtype=torch.bfloat16,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(),
        model_config={"ar_diffusion_height": 32, "ar_diffusion_width": 32},
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=rank,
        distributed_init_method=f"file://{rendezvous}",
    )
    try:
        initialize_model_parallel()
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))),
            set_current_diffusion_config(config),
            set_forward_context(omni_diffusion_config=config),
            torch.inference_mode(),
        ):
            pipeline = DiffusersPipelineLoader(LoadConfig(), config).load_model("cuda", device=device)
            pixels = np.zeros((64, 64, 3), dtype=np.uint8)
            pixels[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 4
            pixels[..., 1] = np.arange(64, dtype=np.uint8)[:, None] * 4
            pixels[..., 2] = 127
            camera = {
                "schema": "solarwm.camera_trajectory.v1",
                "c2w": torch.eye(4).repeat(22, 1, 1).tolist(),
                "intrinsics": [[0.8, 0, 0.5], [0, 0.8, 0.5], [0, 0, 1]],
                "intrinsics_normalized": True,
            }
            for output_type in ("np", "latent"):
                sampling = OmniDiffusionSamplingParams(
                    height=32,
                    width=32,
                    num_frames=22,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    seed=42,
                    fps=24,
                    output_type=output_type,
                    extra_args={"camera": camera},
                )
                request = DiffusionRequestBatch(
                    [
                        OmniDiffusionRequest(
                            prompt={
                                "prompt": "A colorful geometric scene with a stationary camera.",
                                "multi_modal_data": {"image": Image.fromarray(pixels)},
                            },
                            sampling_params=sampling,
                            request_id=f"published-pipeline-{output_type}",
                        )
                    ]
                )
                result = pipeline(request)
                payload = result.output["payload"]
                if output_type == "np":
                    video = payload["video"][0]
                    assert video.shape == (22, 32, 32, 3)
                    assert video.dtype == np.uint8 and np.isfinite(video).all()
                else:
                    latent = payload["latents"]
                    # Two generated five-latent blocks are trimmed to the
                    # seven latents required for exactly 22 decoded frames.
                    assert latent.shape == (1, 24, 7, 2, 2)
                    assert torch.isfinite(latent).all()
                assert not pipeline._sessions
    finally:
        destroy_distributed_env()


def test_published_pipeline_video_and_latent_generation(tmp_path, monkeypatch):
    model_root = os.environ.get("VLLM_TEST_SOLARWM_H3_MODEL", "")
    if not model_root:
        pytest.skip("set VLLM_TEST_SOLARWM_H3_MODEL to opt into the published-checkpoint test")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model_path = Path(model_root).expanduser().resolve()
    assert model_path.is_dir(), f"local SolarWM-H3 model root does not exist: {model_path}"
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    process = mp.spawn(
        _published_pipeline_worker,
        args=(str(model_path), str(tmp_path / "rendezvous")),
        nprocs=1,
        join=False,
    )
    try:
        if not process.join(timeout=600):
            pytest.fail("published SolarWM-H3 pipeline worker timed out")
    finally:
        for worker in process.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
