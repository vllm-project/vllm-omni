# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replay a recorded real AR payload through the native diffusion engine.

One observation-only warmup, then three uninstrumented measured requests.
The native model, loader, collective implementation, scheduler, CFG, random
generator and VAE are unchanged. Source/backend/SP are explicit CLI inputs.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from diffusers.image_processor import VaeImageProcessor
from safetensors.torch import load_file, save_file

from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    # Match the shared Omni entrypoint before the executor creates its queues.
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conditioning", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--degree", type=int, choices=[1, 2], required=True)
    parser.add_argument("--backend", choices=["TORCH_SDPA", "FLASH_ATTN"], required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=4.0)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    condition_dir = Path(args.conditioning)
    metadata = json.loads((condition_dir / "conditioning.json").read_text())
    hidden = load_file(str(condition_dir / "conditioning.safetensors"))["full_hidden_states"]
    assert list(hidden.shape) == metadata["hidden_shape"]
    config = OmniDiffusionConfig(
        model=args.model,
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(json.loads((Path(args.model) / "config.json").read_text())),
        dtype=getattr(torch, args.dtype),
        enforce_eager=True,
        max_num_seqs=1,
        parallel_config=DiffusionParallelConfig(ulysses_degree=args.degree, ulysses_mode="advanced_uaa"),
        diffusion_attention_config={"default": {"backend": args.backend}},
        distributed_executor_backend="mp",
        num_gpus=args.degree,
        worker_extension_cls="qualification_worker.QualificationWorkerExtension",
    )
    setup = {
        **vars(args),
        "conditioning_sha256": hashlib.sha256((condition_dir / "conditioning.safetensors").read_bytes()).hexdigest(),
        "warmup": 1,
        "measured": 3,
        "timing_unit": "ms",
        "allocation": "DiT-only engine; SP=1 reserves one GPU and SP=2 two GPUs; no AR worker",
    }
    (root / "protocol.json").write_text(json.dumps(setup, indent=2))
    engine = DiffusionEngine(config)
    records = []
    try:
        for rank in range(args.degree):
            engine.collective_rpc(
                "qualification_observe",
                args=(str(root / "warmup-observation"),),
                unique_reply_rank=rank,
                timeout=120,
            )
        for index in range(4):
            request = OmniDiffusionRequest(
                request_id=f"checkpoint-replay-{index}",
                prompt={
                    "prompt": "",
                    "height": metadata["height"],
                    "width": metadata["width"],
                    "additional_information": {
                        "full_hidden_states": hidden,
                        "full_token_ids": metadata["full_token_ids"],
                        "answer_start_index": metadata["answer_start_index"],
                    },
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=metadata["height"],
                    width=metadata["width"],
                    seed=args.seed,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    extra_args={"cfg_range": [0.0, 1.0]},
                ),
            )
            for rank in range(args.degree):
                engine.collective_rpc(
                    "qualification_memory",
                    kwargs={"reset": True},
                    unique_reply_rank=rank,
                    timeout=120,
                )
            start = time.perf_counter()
            result = engine.add_req_and_wait_for_response(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert result.error is None and not result.aborted and result.finished, result.error
            image = result.output.detach().cpu().contiguous()
            assert image.shape == (1, 3, metadata["height"], metadata["width"])
            assert torch.isfinite(image).all()
            memory = [
                engine.collective_rpc("qualification_memory", unique_reply_rank=rank, timeout=120)
                for rank in range(args.degree)
            ]
            save_file({"decoded": image}, str(root / f"decoded-{index}.safetensors"))
            VaeImageProcessor().postprocess(image, output_type="pil")[0].save(root / f"output-{index}.png")
            record = {
                "iteration": index,
                "warmup": index == 0,
                "elapsed_ms": elapsed_ms,
                "stage_durations": result.stage_durations,
                "memory": memory,
                "gpu_ms_per_image": args.degree * elapsed_ms,
            }
            records.append(record)
            (root / "results.json").write_text(json.dumps(records, indent=2))
            print("REPLAY_RESULT " + json.dumps(record), flush=True)
            if index == 0:
                for rank in range(args.degree):
                    removed = engine.collective_rpc(
                        "qualification_remove_observers",
                        unique_reply_rank=rank,
                        timeout=120,
                    )
                    assert removed["hooks"] == 0
        measured = [record["elapsed_ms"] for record in records[1:]]
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "mean_ms": statistics.mean(measured),
                    "stdev_ms": statistics.stdev(measured),
                    "min_ms": min(measured),
                    "max_ms": max(measured),
                    "gpu_ms_per_image": args.degree * statistics.mean(measured),
                },
                indent=2,
            )
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
