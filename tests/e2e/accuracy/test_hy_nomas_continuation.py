# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]

PROMPTS = [
    "A forest beside a clear river, daylight.",
    "The camera approaches a wooden bridge beside the river.",
    "Birds fly above the river as the camera moves toward the bridge.",
]
CAMERAS = ["pan_left", None, "pan_left"]


def run_worker(args):
    from vllm_omni.diffusion.distributed.utils import get_local_device

    device = get_local_device()
    prompts = PROMPTS if args.frames == 49 else PROMPTS[:1]
    cameras = CAMERAS if args.frames == 49 else [None]
    if args.worker == "reference":
        from configs import config as reference_config
        from hydra import compose, initialize_config_dir
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
        from omegaconf import OmegaConf

        root = Path(os.environ["HY_NOMAS_REFERENCE_MODEL"])
        with initialize_config_dir(config_dir=str(Path(reference_config.__file__).parent), version_base="1.3"):
            config = compose(config_name="config", overrides=["+experiment=inference"])
        OmegaConf.set_struct(config, False)
        config.dit_weight = str(root / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors")
        config.model.vae.checkpoint = str(root / "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors")
        config.model.vae.parallel_mode = "none"
        config.model.text_encoder.model_name = os.environ.get(
            "HY_NOMAS_REFERENCE_TEXT_ENCODER", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        config.model.dit.seg_ar_attention_backend = "flex"
        config.parallelism.sp_size = 1
        config.parallelism.shard_text_encoder = False
        config.inference.rollout_type = "chunk_ar"
        config.inference.inference_chunk_size = 16
        config.inference.interactive_noise_max_seconds = 8
        config.inference.interactive_noise_hard_max_seconds = 8
        pipeline = CPTCausalInferencePipeline(reference_config._flatten_cfg(config), device, config.dit_weight).eval()
        latents = []
        cache_chunks = []
        turn_chunks = []
        decode = pipeline._decode_latents
        forward = pipeline.transformer.forward_with_cache

        def record_chunk(*values, **kwargs):
            if kwargs["update_cache"]:
                cache_chunks.append(kwargs["hidden_states"].shape[2])
            return forward(*values, **kwargs)

        def record_reference(value, num_frames):
            latents.append(value.detach().cpu().contiguous())
            return decode(value, num_frames)

        pipeline._decode_latents = record_reference
        pipeline.transformer.forward_with_cache = record_chunk
        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        ):
            for prompt, camera in zip(prompts, cameras):
                start = len(cache_chunks)
                pipeline.inference(
                    prompts=[prompt],
                    num_frames=args.frames,
                    height=704,
                    width=1280,
                    guidance_scale=1.0,
                    num_inference_steps=args.steps,
                    flow_shift=5.0,
                    seed=42,
                    camera_trajectory=camera,
                    preserve_kv_cache=True,
                )
                turn_chunks.append(cache_chunks[start:])
            assert len(latents) == len(prompts)
            (args.output / "reference-chunks.json").write_text(json.dumps(turn_chunks) + "\n")
            decoder = pipeline.vae.create_streaming_decode_session(to_cpu=True, retain_history=False)
            for index, latent in enumerate(latents):
                parts = []
                left = 0
                for chunk_index, length in enumerate(turn_chunks[index]):
                    right = left + length
                    decoder.final_segment = index + 1 == len(latents) and chunk_index + 1 == len(turn_chunks[index])
                    pixels = decoder.push(
                        pipeline._prepare_vae_decode_latents(latent[:, :, left:right].to(device)),
                        pixel_frames=min(args.frames, right * 4) - left * 4,
                    )
                    if pixels.numel():
                        parts.append(pixels)
                    left = right
                assert left == latent.shape[2]
                if index + 1 == len(latents):
                    tail = decoder.finish()
                    if tail.numel():
                        parts.append(tail)
                pixels = torch.cat(parts, 2)
                video = pipeline._postprocess_vae_video(pixels)[0].permute(1, 2, 3, 0)
                video = video.float().mul(255).round().to(torch.uint8).contiguous()
                save_file({"latents": latent, "video": video}, str(args.output / f"turn-{index}.safetensors"))
        return

    from vllm.config import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
    from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    config = OmniDiffusionConfig(
        model=os.environ["HY_NOMAS_MODEL"],
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        enforce_eager=True,
        custom_pipeline_args={"ar_max_latent_frames": 48, "ar_max_segments": 4},
    )
    with set_default_torch_dtype(torch.bfloat16):
        pipeline = HyNomasPipeline(od_config=config)
    pipeline.load_weights(DiffusersPipelineLoader(LoadConfig(), config).get_all_weights(pipeline))
    if args.offload:
        pipeline.enable_omni_model_cpu_offload(device=device)
    spec = pipeline.ar_diffusion_kv_cache_spec()
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=spec.tokens_per_frame, window_chunks=1, warmup_cudagraph=False),
        num_layers=spec.num_layers,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=torch.bfloat16,
        block_size=spec.segmented.block_size,
        max_model_len=spec.segmented.initial_staging_tokens,
        available_bytes=current_omni_platform.mem_get_info()[0],
        kv_branches=spec.kv_branches,
        session_capacity=1,
        device=device,
        segmented=spec.segmented,
        model_owned_state_bytes_per_session=spec.model_owned_state_bytes_per_session,
    )
    owner = ARDiffusionKVState(cache, "parity", {"cond": cache.begin_request("parity")}, num_layers=spec.num_layers)
    actual_latents = []
    post_decode = pipeline.post_decode

    def record_candidate(state, **kwargs):
        actual_latents.append(state.latents.detach().cpu().contiguous())
        return post_decode(state, **kwargs)

    pipeline.post_decode = record_candidate
    records = []
    try:
        for index, (prompt, camera) in enumerate(zip(prompts, cameras)):
            tick = ARDiffusionTickRequest(
                "parity",
                f"turn-{index}",
                index,
                prompt=prompt,
                close_session=index + 1 == len(prompts),
                controls=(ARDiffusionControlInput("camera", "hy_nomas.camera.v1", {"camera_trajectory": camera}),),
            )
            params = OmniDiffusionSamplingParams(
                height=704,
                width=1280,
                num_frames=args.frames,
                seed=42,
                num_inference_steps=args.steps,
                guidance_scale=1,
                extra_args=(
                    tick.to_extra_args() if args.frames == 49 else {"session_id": "parity", "close_session": True}
                ),
            )
            start = len(actual_latents)
            with pipeline.bind_ar_diffusion_state("parity", owner):
                output = pipeline(
                    DiffusionRequestBatch(
                        [OmniDiffusionRequest(prompt=prompt, sampling_params=params, request_id=tick.request_id)]
                    )
                )
            expected = load_file(str(args.output / f"turn-{index}.safetensors"))
            torch.testing.assert_close(torch.cat(actual_latents[start:], 2), expected["latents"], rtol=0, atol=0)
            video = output.output["payload"]["video"]
            np.testing.assert_array_equal(video, expected["video"].numpy())
            records.append({"turn": index, "frames": len(video), "latent_max_abs": 0, "uint8_max_abs": 0})
        assert sum(record["frames"] for record in records) == args.frames * len(prompts)
        assert output.output["metadata"]["hy_nomas"]["withheld_frames"] == 0
    finally:
        owner.close()
        pipeline.close_ar_diffusion_session("parity")
    (args.output / f"candidate-{'offload' if args.offload else 'resident'}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


@pytest.mark.parametrize("steps", [pytest.param(4, id="four_steps"), pytest.param(8, id="eight_steps")])
@pytest.mark.parametrize("frames", [pytest.param(49, id="three_turns"), pytest.param(129, id="short_tail")])
def test_reference_continuation_at_704x1280(tmp_path, steps, frames):
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--steps",
        str(steps),
        "--frames",
        str(frames),
        "--output",
        str(tmp_path),
    ]
    commands = [
        base + ["--worker", "reference"],
        base + ["--worker", "candidate"],
        base + ["--worker", "candidate", "--offload"],
    ]
    for index, command in enumerate(commands):
        with (tmp_path / f"worker-{index}.log").open("w") as log:
            subprocess.run(command, check=True, timeout=1800, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("reference", "candidate"), required=True)
    parser.add_argument("--steps", type=int, choices=(4, 8), required=True)
    parser.add_argument("--frames", type=int, choices=(49, 129), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload", action="store_true")
    with torch.no_grad():
        run_worker(parser.parse_args())
