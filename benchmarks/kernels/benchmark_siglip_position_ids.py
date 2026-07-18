"""Microbenchmark MiniCPM-o 4.5 SigLIP position-ID construction.

This isolates the position-ID hot path from patch embedding and the vision
transformer. It compares the legacy per-item CPU path, an independent copy of
the grouped host path used by ``SiglipVisionEmbeddings``, and the rejected
device-side alternative.

Example:
    python benchmarks/kernels/benchmark_siglip_position_ids.py \
        --device cuda --batch-sizes 1,4,16,64 --shapes 32x32,28x37,40x25
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial

import torch


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        height, width = item.lower().split("x", maxsplit=1)
        shapes.append((int(height), int(width)))
    return shapes


def _make_inputs(
    batch_size: int,
    shapes: list[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.BoolTensor, torch.IntTensor]:
    target_sizes = [shapes[index % len(shapes)] for index in range(batch_size)]
    max_num_patches = max(height * width for height, width in target_sizes)
    patch_attention_mask = torch.zeros((batch_size, 1, max_num_patches), dtype=torch.bool, device=device)
    for batch_idx, (height, width) in enumerate(target_sizes):
        patch_attention_mask[batch_idx, 0, : height * width] = True
    return patch_attention_mask, torch.tensor(target_sizes, dtype=torch.int32, device=device)


def _legacy_host_position_ids(
    num_patches_per_side: int,
    patch_attention_mask: torch.BoolTensor,
    target_sizes: torch.IntTensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size, _, max_num_patches = patch_attention_mask.shape
    boundaries = torch.arange(
        1 / num_patches_per_side,
        1.0,
        1 / num_patches_per_side,
    )
    position_ids = torch.zeros((batch_size, max_num_patches), dtype=torch.long)

    for batch_idx, patch_mask in enumerate(patch_attention_mask):
        height, width = (int(value) for value in target_sizes[batch_idx].cpu())
        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / height)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / width)
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        grid_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx, patch_mask.flatten().cpu()] = grid_ids

    return position_ids.to(device=device)


def _create_grid_position_ids(
    num_patches_per_side: int,
    patch_grid_height: int,
    patch_grid_width: int,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    fractional_coords_h = torch.arange(
        0,
        1 - 1e-6,
        1 / patch_grid_height,
        device=boundaries.device,
    )
    fractional_coords_w = torch.arange(
        0,
        1 - 1e-6,
        1 / patch_grid_width,
        device=boundaries.device,
    )
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    return (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()


def _grouped_host_position_ids(
    num_patches_per_side: int,
    patch_attention_mask: torch.BoolTensor,
    target_sizes: torch.IntTensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = patch_attention_mask.size(0)
    flat_patch_attention_mask = patch_attention_mask.reshape(batch_size, -1).to(device="cpu")
    target_sizes_list = target_sizes.detach().to(device="cpu").tolist()
    position_ids = torch.zeros(flat_patch_attention_mask.shape, dtype=torch.long)
    boundaries = torch.arange(
        1 / num_patches_per_side,
        1.0,
        1 / num_patches_per_side,
    )
    grid_position_ids: dict[tuple[int, int], torch.Tensor] = {}

    for batch_idx, (patch_grid_height, patch_grid_width) in enumerate(target_sizes_list):
        grid_shape = (int(patch_grid_height), int(patch_grid_width))
        grid_ids = grid_position_ids.get(grid_shape)
        if grid_ids is None:
            grid_ids = _create_grid_position_ids(
                num_patches_per_side,
                *grid_shape,
                boundaries,
            )
            grid_position_ids[grid_shape] = grid_ids

        position_ids[batch_idx, flat_patch_attention_mask[batch_idx]] = grid_ids

    return position_ids.to(device=device)


def _device_position_ids(
    num_patches_per_side: int,
    patch_attention_mask: torch.BoolTensor,
    target_sizes: torch.IntTensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = patch_attention_mask.size(0)
    flat_patch_attention_mask = patch_attention_mask.reshape(batch_size, -1).to(device=device)
    target_sizes_list = target_sizes.cpu().tolist()
    position_ids = torch.zeros(flat_patch_attention_mask.shape, dtype=torch.long, device=device)
    boundaries = torch.arange(
        1 / num_patches_per_side,
        1.0,
        1 / num_patches_per_side,
        device=device,
    )
    grid_position_ids: dict[tuple[int, int], torch.Tensor] = {}

    for batch_idx, (height, width) in enumerate(target_sizes_list):
        grid_shape = (int(height), int(width))
        grid_ids = grid_position_ids.get(grid_shape)
        if grid_ids is None:
            grid_ids = _create_grid_position_ids(
                num_patches_per_side,
                *grid_shape,
                boundaries,
            )
            grid_position_ids[grid_shape] = grid_ids
        position_ids[batch_idx, flat_patch_attention_mask[batch_idx]] = grid_ids

    return position_ids


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize()


def _measure_latency_us(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmups: int,
    iterations: int,
) -> float:
    for _ in range(warmups):
        function()
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(iterations):
        function()
    _synchronize(device)
    return (time.perf_counter() - start) * 1e6 / iterations


def _measure_peak_memory_kib(function: Callable[[], torch.Tensor], device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.accelerator.reset_peak_memory_stats()
    function()
    _synchronize(device)
    return torch.accelerator.max_memory_allocated() / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid-size", type=int, default=70)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=_parse_int_list("1,4,16,64"))
    parser.add_argument("--shapes", type=_parse_shapes, default=_parse_shapes("32x32,28x37,24x43,40x25"))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device} grid_size={args.grid_size} shapes={args.shapes}")
    print(
        f"{'batch':>7} {'legacy_us':>12} {'grouped_us':>12} "
        f"{'device_us':>12} {'speedup':>9} {'device_exact':>12} {'peak_KiB':>10}"
    )

    for batch_size in args.batch_sizes:
        patch_attention_mask, target_sizes = _make_inputs(batch_size, args.shapes, device)
        legacy = partial(
            _legacy_host_position_ids,
            args.grid_size,
            patch_attention_mask,
            target_sizes,
            device,
        )
        grouped = partial(
            _grouped_host_position_ids,
            args.grid_size,
            patch_attention_mask,
            target_sizes,
            device=device,
        )
        device_side = partial(
            _device_position_ids,
            args.grid_size,
            patch_attention_mask,
            target_sizes,
            device,
        )

        expected = legacy()
        torch.testing.assert_close(grouped(), expected, rtol=0, atol=0)
        device_exact = torch.equal(device_side(), expected)

        legacy_us = _measure_latency_us(legacy, device, args.warmups, args.iterations)
        grouped_us = _measure_latency_us(grouped, device, args.warmups, args.iterations)
        device_us = _measure_latency_us(device_side, device, args.warmups, args.iterations)
        peak_memory_kib = _measure_peak_memory_kib(grouped, device)
        speedup = legacy_us / grouped_us
        print(
            f"{batch_size:7d} {legacy_us:12.2f} {grouped_us:12.2f} "
            f"{device_us:12.2f} {speedup:8.2f}x "
            f"{str(device_exact):>12} {peak_memory_kib:10.1f}"
        )


if __name__ == "__main__":
    main()
