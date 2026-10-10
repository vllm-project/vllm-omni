# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK stage-1 request-batch benchmark with real weights and fixed conditioning.

Run the same script in the base and PR checkouts. This measures DiT + VAE,
excluding the text encoder, scheduler and transport; it makes no quality claim.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.auk.pipeline_auk import AuKPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.repeats < 1 or args.warmups < 1:
        parser.error("batch sizes, repeats and warmups must be positive")
    frames = round(args.seconds * 50)
    model_config = {
        "auk_dit_warmup_frames": [frames],
        "auk_dit_warmup_nfe": args.steps,
        "auk_vae_compile_shapes": [frames],
        "auk_vae_tile_frames": 0,
        "auk_dit_warmup_batches": args.batch_sizes,
        "auk_vae_warmup_batches": args.batch_sizes,
    }
    pipeline = AuKPipeline(
        od_config=OmniDiffusionConfig(
            model=args.model,
            model_class_name="AuKPipeline",
            dtype=torch.bfloat16,
            enforce_eager=False,
            max_num_seqs=max(args.batch_sizes),
            model_config=model_config,
        )
    )
    pipeline.eval()
    pipeline.setup_compile()
    text = torch.randn(64, pipeline.text_hidden_dim, generator=torch.Generator().manual_seed(42))
    results = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:

            def run() -> None:
                requests = [
                    OmniDiffusionRequest(
                        request_id=f"auk-{i}",
                        prompt={
                            "prompt_embeds": text,
                            "additional_information": {"auk": {"gen_seconds": args.seconds}},
                        },
                        sampling_params=OmniDiffusionSamplingParams(
                            num_inference_steps=args.steps,
                            guidance_scale=2.0,
                            seed=100 + i,
                            output_type="pt",
                        ),
                    )
                    for i in range(batch_size)
                ]
                if pipeline.supports_request_batch:
                    outputs = pipeline.forward(DiffusionRequestBatch(requests))
                else:
                    outputs = [pipeline.forward(DiffusionRequestBatch([request]))[0] for request in requests]
                assert len(outputs) == batch_size
                assert all(output.output.shape == (frames * pipeline.hop_size,) for output in outputs)

            for _ in range(args.warmups):
                run()
            torch.accelerator.synchronize()
            torch.accelerator.memory.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                run()
                torch.accelerator.synchronize()
                times.append(time.perf_counter() - start)
            median = statistics.median(times)
            results.append(
                {
                    "batch_size": batch_size,
                    "wall_seconds": times,
                    "median_seconds": median,
                    "rtf": median / args.seconds,
                    "audio_seconds_per_second": batch_size * args.seconds / median,
                    "peak_allocated_gib": torch.accelerator.memory.max_memory_allocated() / 2**30,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "stage 1 only, synthetic text conditioning, real checkpoint weights",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "platform": current_omni_platform.device_type,
                "model": args.model,
                "request_batch": pipeline.supports_request_batch,
                "steps": args.steps,
                "seconds": args.seconds,
                "warmups": args.warmups,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
