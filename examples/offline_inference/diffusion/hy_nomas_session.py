# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
from vllm_omni.experimental.ar_diffusion.session import ARDiffusionWorkerLifecycle
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


async def main():
    parser = argparse.ArgumentParser(description="Continue a Hy-Nomas AR session and save each stable video chunk.")
    parser.add_argument("--model", default="tencent/Hy-Nomas")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--camera", default=None, help="Reference camera trajectory or generator, such as pan_left.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=49, help="Frames per turn, at most 61 for a 16-latent chunk.")
    parser.add_argument("--num-inference-steps", type=int, choices=(4, 6, 8), default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument("--text-encoder", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    if args.num_frames < 5 or (args.num_frames - 1) % 4 or args.num_frames > 61:
        parser.error("--num-frames must be 4*n+1 in [5, 61].")
    if len(args.prompts) * ((args.num_frames - 1) // 4 + 1) > 120 or len(args.prompts) > 16:
        parser.error("The example session must fit 120 latent frames and 16 prompt segments.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = AsyncOmni(
        model=args.model,
        enforce_eager=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        max_num_seqs=1,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        step_execution=False,
        parallel_config={"ulysses_degree": args.ulysses_degree},
        custom_pipeline_args={"text_encoder": args.text_encoder, "ar_height": args.height, "ar_width": args.width},
        model_config={"ar_diffusion_kv_config": {"gpu_memory_fraction": 0.02, "warmup_cudagraph": False}},
    )
    lifecycle = ARDiffusionWorkerLifecycle(engine, stage_ids=[0], timeout=180)
    session_id = "hy-nomas-example"
    frames, records = [], []
    try:
        consumer = ARDiffusionOmniTickConsumer(
            engine,
            prompt_provider=lambda tick: tick.prompt,
            sampling_params_list=[
                OmniDiffusionSamplingParams(
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    seed=args.seed,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=1,
                    output_type="np",
                )
            ],
            diffusion_stage_id=0,
            output_modalities=["video"],
        )
        controls = (
            ()
            if args.camera is None
            else (ARDiffusionControlInput("camera", "hy_nomas.camera.v1", {"camera_trajectory": args.camera}),)
        )
        for index, prompt in enumerate(args.prompts):
            tick = ARDiffusionTickRequest(
                session_id=session_id,
                request_id=f"hy-turn-{index}",
                chunk_index=index,
                prompt=prompt,
                controls=controls,
                applied_event_ids=(index,),
                reset=index == 0,
                close_session=index + 1 == len(args.prompts),
            )
            started = time.perf_counter()
            output = await consumer.execute_tick(tick)
            elapsed = time.perf_counter() - started
            if len(output.images) != 1:
                raise RuntimeError("Expected one video chunk from Hy-Nomas.")
            video = output.images[0]
            if not isinstance(video, np.ndarray) or video.ndim != 4 or video.dtype != np.uint8:
                raise RuntimeError("Expected a THWC uint8 video chunk from Hy-Nomas.")
            metadata = output.multimodal_output["metadata"]
            record = {"turn": index, "frames": len(video), "generation_seconds": elapsed, "metadata": metadata}
            records.append(record)
            print(json.dumps(record), flush=True)
            if len(video):
                export_to_video(video.astype(np.float32) / 255, str(args.output_dir / f"chunk_{index:03d}.mp4"), fps=24)
                frames.append(video)
        export_to_video(np.concatenate(frames).astype(np.float32) / 255, str(args.output_dir / "session.mp4"), fps=24)
        (args.output_dir / "chunks.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        try:
            await lifecycle.close_session(session_id)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
