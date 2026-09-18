# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One ABot generate() call, emitting statefully decoded chunks.

Camera actions are fixed at request start. For actions arriving during a
rollout, use the experimental typed ARDiffusionSessionManager API instead.
"""

import argparse
import asyncio
import json
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


async def main(args: argparse.Namespace) -> None:
    if args.ticks < 1:
        raise ValueError("--ticks must be positive.")
    extra_args = {"flow_shift": 5.0}
    if args.actions:
        # One item per tick; each item has three lists of W/A/S/D/I/J/K/L keys.
        extra_args["camera_action_script"] = json.loads(Path(args.actions).read_text())
    engine = AsyncOmni(
        model=args.model,
        model_class_name="ABotWorldCausalPipeline",
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        step_execution=True,
        diffusion_streaming_output=True,
        model_config={
            "abot_vae": args.vae,
            "ar_diffusion_height": 512,
            "ar_diffusion_width": 832,
            "ar_diffusion_kv_config": {"warmup_cudagraph": False},
        },
    )
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=832,
        num_frames=12 * args.ticks - 3,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=args.seed,
        output_type="pt",
        extra_args=extra_args,
    )
    chunks = []
    try:
        async for output in engine.generate(
            {"prompt": args.prompt, "multi_modal_data": {"image": args.image}},
            sampling,
            request_id="abot-stepwise",
        ):
            if output.error:
                raise RuntimeError(output.error)
            if not output.images:
                continue
            frames = output.images[0]
            if frames.ndim == 5:
                frames = frames[0]
            expected = 9 if not chunks else 12
            if frames.shape != (expected, 3, 512, 832):
                raise RuntimeError(f"Unexpected chunk shape: {tuple(frames.shape)}")
            chunks.append(frames.cpu())
            print(f"chunk {len(chunks) - 1}: {expected} frames", flush=True)
        if len(chunks) != args.ticks:
            raise RuntimeError(f"Expected {args.ticks} chunks, received {len(chunks)}")
        video = torch.cat(chunks).float().permute(0, 2, 3, 1).numpy()
        export_to_video(video, args.output, fps=args.fps)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local ABot checkpoint directory")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="The camera moves slowly forward through the scene.")
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", choices=("wan", "taew2_2"), default="wan")
    parser.add_argument("--actions", help="JSON camera_action_script, fixed before generation starts")
    parser.add_argument("--output", default="abot-stepwise.mp4")
    parser.add_argument("--fps", type=int, default=24)
    asyncio.run(main(parser.parse_args()))
