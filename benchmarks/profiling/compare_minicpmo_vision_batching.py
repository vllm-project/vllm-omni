#!/usr/bin/env python3
"""Compare full-checkpoint MiniCPM-o vision embeddings with and without batching."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, SiglipVisionConfig
from vllm.multimodal.utils import fetch_video

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    Resampler,
    SiglipVisionTransformer,
)


@dataclass
class VisionConfig:
    vision_batch_size: int


class VisionInputs(TypedDict):
    pixel_values: list[torch.Tensor]
    tgt_sizes: torch.Tensor


class VisionModelHarness:
    """Minimum typed model surface needed by ``get_vision_hidden_states``."""

    def __init__(
        self,
        *,
        vision_batch_size: int,
        vpm: SiglipVisionTransformer,
        resampler: Resampler,
    ) -> None:
        self.config = VisionConfig(vision_batch_size=vision_batch_size)
        self.vpm = vpm
        self.resampler = resampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--patch-tokens", type=int, default=1032)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Vision module and input dtype.",
    )
    media = parser.add_mutually_exclusive_group()
    media.add_argument(
        "--image",
        type=Path,
        help="Use the preprocessing output from this real image instead of synthetic pixels.",
    )
    media.add_argument(
        "--video",
        type=Path,
        help="Use the frames and preprocessing output from this real video instead of synthetic pixels.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--latency-repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.vision_batch_size < 1 or args.patch_tokens < 1 or args.latency_repetitions < 1:
        parser.error("batch sizes and patch token count must be positive")
    return args


def factor_grid(num_patches: int) -> tuple[int, int]:
    height = math.isqrt(num_patches)
    while num_patches % height:
        height -= 1
    return height, num_patches // height


def prepare_image_inputs(
    model_path: Path,
    image_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[VisionInputs, dict[str, object]]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processed = processor.image_processor.preprocess(
        [image],
        return_tensors=None,
    )
    pixel_values = [torch.from_numpy(item).to(device=device, dtype=dtype) for item in processed["pixel_values"][0]]
    tgt_sizes = torch.as_tensor(
        processed["tgt_sizes"][0],
        dtype=torch.int32,
        device=device,
    )
    patch_grids = sorted({tuple(int(value) for value in row) for row in tgt_sizes.cpu().tolist()})
    patch_tokens = sorted({height * width for height, width in patch_grids})
    input_report: dict[str, object] = {
        "source": "image",
        "image_path": str(image_path.resolve()),
        "image_size": list(image.size),
        "batch_size": len(pixel_values),
        "patch_tokens_per_item": patch_tokens,
        "patch_grids": [list(grid) for grid in patch_grids],
    }
    return {"pixel_values": pixel_values, "tgt_sizes": tgt_sizes}, input_report


def prepare_video_inputs(
    model_path: Path,
    video_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[VisionInputs, dict[str, object]]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    frames, metadata = fetch_video(video_path.resolve().as_uri())
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processed = processor.image_processor.preprocess(
        [list(frames)],
        max_slice_nums=1,
        return_tensors=None,
    )
    pixel_values = [torch.from_numpy(item).to(device=device, dtype=dtype) for item in processed["pixel_values"][0]]
    tgt_sizes = torch.as_tensor(
        processed["tgt_sizes"][0],
        dtype=torch.int32,
        device=device,
    )
    patch_grids = sorted({tuple(int(value) for value in row) for row in tgt_sizes.cpu().tolist()})
    patch_tokens = sorted({height * width for height, width in patch_grids})
    input_report: dict[str, object] = {
        "source": "video",
        "video_path": str(video_path.resolve()),
        "decoded_frame_count": int(frames.shape[0]),
        "decoded_frame_shape": list(frames.shape[1:]),
        "video_backend": metadata.get("video_backend"),
        "sampled_frame_indices": metadata.get("frames_indices"),
        "batch_size": len(pixel_values),
        "patch_tokens_per_item": patch_tokens,
        "patch_grids": [list(grid) for grid in patch_grids],
    }
    return {"pixel_values": pixel_values, "tgt_sizes": tgt_sizes}, input_report


def prepare_synthetic_inputs(
    *,
    batch_size: int,
    patch_tokens: int,
    patch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[VisionInputs, dict[str, object]]:
    torch.manual_seed(seed)
    height_patches, width_patches = factor_grid(patch_tokens)
    pixel_values = [
        torch.randn(
            3,
            patch_size,
            patch_size * patch_tokens,
            device=device,
            dtype=dtype,
        )
        for _ in range(batch_size)
    ]
    tgt_sizes = torch.tensor(
        [[height_patches, width_patches]] * batch_size,
        dtype=torch.int32,
        device=device,
    )
    input_report: dict[str, object] = {
        "source": "synthetic",
        "batch_size": batch_size,
        "patch_tokens_per_item": [patch_tokens],
        "patch_grids": [[height_patches, width_patches]],
    }
    return {"pixel_values": pixel_values, "tgt_sizes": tgt_sizes}, input_report


def load_submodule_state(model_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    vpm_state: dict[str, torch.Tensor] = {}
    resampler_state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith("vpm."):
                    vpm_state[key.removeprefix("vpm.")] = checkpoint.get_tensor(key)
                elif key.startswith("resampler."):
                    resampler_state[key.removeprefix("resampler.")] = checkpoint.get_tensor(key)
    if not vpm_state or not resampler_state:
        raise RuntimeError(f"vision weights not found under {model_path}")
    return vpm_state, resampler_state


def run_encoder(
    model: VisionModelHarness,
    data: VisionInputs,
    vision_batch_size: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    model.config.vision_batch_size = vision_batch_size
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    baseline_allocated = torch.accelerator.memory_allocated()
    baseline_reserved = torch.accelerator.memory_reserved()
    with torch.inference_mode():
        output = MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(model, data)
    torch.accelerator.synchronize()
    peak_allocated = torch.accelerator.max_memory_allocated()
    peak_reserved = torch.accelerator.max_memory_reserved()
    return output, {
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_delta_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_delta_bytes": peak_reserved - baseline_reserved,
    }


def benchmark_encoder_latency(
    model: VisionModelHarness,
    data: VisionInputs,
    vision_batch_size: int,
    repetitions: int,
) -> dict[str, float | int]:
    model.config.vision_batch_size = vision_batch_size
    with torch.inference_mode():
        warmup_output = MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(model, data)
    del warmup_output
    torch.accelerator.synchronize()

    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repetitions):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            output = MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(model, data)
            ended.record()
            ended.synchronize()
            elapsed_ms.append(started.elapsed_time(ended))
            del output

    return {
        "repetitions": repetitions,
        "mean_ms": statistics.fmean(elapsed_ms),
        "stdev_ms": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def main() -> None:
    args = parse_args()
    config_dict = json.loads((args.model / "config.json").read_text())
    vision_config = SiglipVisionConfig(**config_dict["vision_config"])
    vision_config._attn_implementation = "eager"

    vpm = SiglipVisionTransformer(vision_config)
    if config_dict.get("drop_vision_last_layer", False):
        vpm.encoder.layers = vpm.encoder.layers[:-1]
    resampler = Resampler(
        num_queries=config_dict["query_num"],
        embed_dim=config_dict["hidden_size"],
        num_heads=config_dict["hidden_size"] // 128,
        kv_dim=vision_config.hidden_size,
        adaptive=True,
    )
    vpm_state, resampler_state = load_submodule_state(args.model)
    vpm.load_state_dict(vpm_state, strict=True)
    resampler.load_state_dict(resampler_state, strict=True)
    del vpm_state, resampler_state

    device = torch.device("cuda")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    vpm.to(device=device, dtype=dtype).eval()
    resampler.to(device=device, dtype=dtype).eval()

    patch_size = vision_config.patch_size
    if args.image is not None:
        data, input_report = prepare_image_inputs(
            args.model,
            args.image,
            device,
            dtype,
        )
    elif args.video is not None:
        data, input_report = prepare_video_inputs(args.model, args.video, device, dtype)
    else:
        data, input_report = prepare_synthetic_inputs(
            batch_size=args.batch_size,
            patch_tokens=args.patch_tokens,
            patch_size=patch_size,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )

    batch_size = len(data["pixel_values"])
    model = VisionModelHarness(
        vision_batch_size=batch_size,
        vpm=vpm,
        resampler=resampler,
    )

    reference, reference_memory = run_encoder(model, data, batch_size)
    reference_cpu = reference.float().cpu()
    del reference
    chunked, chunked_memory = run_encoder(model, data, args.vision_batch_size)
    output_dtype = str(chunked.dtype)
    chunked_cpu = chunked.float().cpu()
    del chunked

    reference_latency = benchmark_encoder_latency(
        model,
        data,
        batch_size,
        args.latency_repetitions,
    )
    chunked_latency = benchmark_encoder_latency(
        model,
        data,
        args.vision_batch_size,
        args.latency_repetitions,
    )

    reference_64 = reference_cpu.double()
    chunked_64 = chunked_cpu.double()
    difference = (reference_64 - chunked_64).abs()
    relative = difference / reference_64.abs().clamp_min(torch.finfo(torch.float64).eps)
    reference_rms = reference_64.square().mean().sqrt()
    reference_max = reference_64.abs().max()
    rmse = difference.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        reference_64.flatten(),
        chunked_64.flatten(),
        dim=0,
    )
    input_report["vision_batch_size"] = args.vision_batch_size
    input_report["dtype"] = str(dtype)
    report = {
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "input": input_report,
        "output": {
            "shape": list(reference_cpu.shape),
            "dtype": output_dtype,
            "reference_rms": reference_rms.item(),
            "reference_max_absolute_value": reference_max.item(),
            "max_absolute_error": difference.max().item(),
            "p99_absolute_error": torch.quantile(difference, 0.99).item(),
            "mean_absolute_error": difference.mean().item(),
            "rmse": rmse.item(),
            "normalized_rmse": (rmse / reference_rms).item(),
            "max_error_normalized_by_reference_max": (difference.max() / reference_max).item(),
            "max_relative_error": relative.max().item(),
            "p99_relative_error": torch.quantile(relative, 0.99).item(),
            "cosine_similarity": cosine.item(),
        },
        "reference_memory": reference_memory,
        "chunked_memory": chunked_memory,
        "reference_latency": reference_latency,
        "chunked_latency": chunked_latency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
