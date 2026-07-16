"""Benchmark the MiniCPM-o 4.5 SigLIP vision-transformer forward path.

The benchmark instantiates the production vision architecture without loading
checkpoint values. Weight values do not affect inference latency or allocation
shape, while avoiding a second copy of the full model just to measure this
isolated encoder. A batch represents images or same-sized video frames.

Example:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_siglip_vision_encoder.py \\
        --model-config /path/to/MiniCPM-o-4_5/config.json \\
        --grid-shape 40x25 --frame-counts 1,4,8,16
"""

from __future__ import annotations

import argparse
import json

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _parse_grid_shape(value: str) -> tuple[int, int]:
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)


def _synchronize() -> None:
    torch.accelerator.synchronize()


@torch.inference_mode()
def _forward(
    model: SiglipVisionTransformer,
    pixel_values: torch.Tensor,
    patch_attention_mask: torch.BoolTensor,
    target_sizes: torch.IntTensor,
) -> None:
    model(
        pixel_values,
        patch_attention_mask=patch_attention_mask,
        tgt_sizes=target_sizes,
        return_dict=False,
    )


def _measure_latency_ms(
    model: SiglipVisionTransformer,
    pixel_values: torch.Tensor,
    patch_attention_mask: torch.BoolTensor,
    target_sizes: torch.IntTensor,
    warmups: int,
    iterations: int,
) -> float:
    for _ in range(warmups):
        _forward(model, pixel_values, patch_attention_mask, target_sizes)
    _synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _forward(model, pixel_values, patch_attention_mask, target_sizes)
    end.record()
    _synchronize()
    return start.elapsed_time(end) / iterations


def _measure_peak_activation_mib(
    model: SiglipVisionTransformer,
    pixel_values: torch.Tensor,
    patch_attention_mask: torch.BoolTensor,
    target_sizes: torch.IntTensor,
) -> float:
    baseline = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    _forward(model, pixel_values, patch_attention_mask, target_sizes)
    _synchronize()
    return (torch.accelerator.max_memory_allocated() - baseline) / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--grid-shape", type=_parse_grid_shape, default=(40, 25))
    parser.add_argument("--frame-counts", type=_parse_int_list, default=_parse_int_list("1,4,8,16"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")

    with open(args.model_config) as config_file:
        vision_config = json.load(config_file)["vision_config"]
    config = SiglipVisionConfig(**vision_config)
    dtype = getattr(torch, args.dtype)
    model = SiglipVisionTransformer(config).to(device=device, dtype=dtype).eval()

    grid_height, grid_width = args.grid_shape
    pixel_height = grid_height * config.patch_size
    pixel_width = grid_width * config.patch_size
    print(
        f"device={device} dtype={args.dtype} attention={config._attn_implementation} "
        f"grid={grid_height}x{grid_width} pixels={pixel_height}x{pixel_width}"
    )
    print(f"{'frames':>7} {'encoder_ms':>12} {'peak_activation_MiB':>22}")

    for frame_count in args.frame_counts:
        pixel_values = torch.randn(
            frame_count,
            config.num_channels,
            pixel_height,
            pixel_width,
            dtype=dtype,
            device=device,
        )
        patch_attention_mask = torch.ones(
            frame_count,
            grid_height,
            grid_width,
            dtype=torch.bool,
            device=device,
        )
        target_sizes = torch.tensor(
            [[grid_height, grid_width]] * frame_count,
            dtype=torch.int32,
            device=device,
        )
        peak_activation_mib = _measure_peak_activation_mib(model, pixel_values, patch_attention_mask, target_sizes)
        latency_ms = _measure_latency_ms(
            model,
            pixel_values,
            patch_attention_mask,
            target_sizes,
            args.warmups,
            args.iterations,
        )
        print(f"{frame_count:7d} {latency_ms:12.2f} {peak_activation_mib:22.2f}")


if __name__ == "__main__":
    main()
