# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Plot Dense vs paged Ring CUDA allocated-memory timelines.

Dense and paged modes run in separate worker processes so the paged KV cache
does not contaminate the dense baseline. PyTorch allocator history provides
timestamped ``alloc`` and ``free_requested`` events without a polling thread.
Each iteration is resampled onto one elapsed-time grid, reduced to the maximum
rank, and then reduced to the pointwise median across iterations.

Example:

    python benchmarks/diffusion/bench_paged_ring_memory_timeline.py \
        --ulysses-degree 1 --ring-degree 4 --seq-len 32768 \
        --output paged_ring_memory_timeline_r4.png
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from queue import SimpleQueue

import torch
import torch.distributed as dist
from bench_paged_ring_attention import (
    _DTYPE,
    _HEAD_SIZE,
    _NUM_HEADS,
    _build_attention,
    _build_real_adapter,
    _SmokeAttention,
)
from PIL import Image, ImageDraw, ImageFont
from vllm.utils.network_utils import get_file_store_init_method

from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import DiffusionPagedAttentionRow
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform


def _step_resample(
    samples: list[tuple[float, int]],
    grid_ms: list[float],
) -> list[float]:
    values: list[float] = []
    sample_index = 0
    current = float(samples[0][1])
    for grid_value in grid_ms:
        while sample_index + 1 < len(samples) and samples[sample_index + 1][0] <= grid_value:
            sample_index += 1
            current = float(samples[sample_index][1])
        values.append(current)
    return values


def _capture_trace(
    fn,
    device: torch.device,
    tail_ms: float,
) -> tuple[list[tuple[float, int]], float]:
    dist.barrier(device_ids=[device.index])
    current_omni_platform.synchronize()
    memory = torch.cuda.memory
    memory._record_memory_history(
        enabled="all",
        context=None,
        stacks="python",
        max_entries=200_000,
        clear_history=True,
    )
    try:
        baseline_bytes = torch.accelerator.memory_allocated(device)
        start_us = time.time_ns() / 1_000.0
        output = fn()
        current_omni_platform.synchronize()
        forward_end_us = time.time_ns() / 1_000.0
        del output
        final_bytes = torch.accelerator.memory_allocated(device)
        snapshot = memory._snapshot()
    finally:
        memory._record_memory_history(enabled=None)

    current_bytes = baseline_bytes
    samples: list[tuple[float, int]] = [(0.0, current_bytes)]
    for event in snapshot["device_traces"][device.index]:
        event_us = float(event.get("time_us", 0.0))
        if event_us < start_us:
            continue
        action = event["action"]
        if action == "alloc":
            current_bytes += int(event["size"])
        elif action == "free_requested":
            current_bytes -= int(event["size"])
        else:
            continue
        samples.append(((event_us - start_us) / 1_000.0, current_bytes))

    forward_end_ms = (forward_end_us - start_us) / 1_000.0
    samples.append((forward_end_ms + tail_ms, final_bytes))
    return samples, forward_end_ms


def _worker(
    local_rank: int,
    world_size: int,
    init_method: str,
    mode: str,
    seq_len: int,
    warmup: int,
    iters: int,
    ulysses_degree: int,
    ring_degree: int,
    tail_ms: float,
    grid_points: int,
    result_queue: SimpleQueue,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="nccl",
        distributed_init_method=init_method,
    )
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    try:
        sp_group = get_sp_group()
        ring_runner = RingParallelAttention(sp_group)
        strategy = (
            UlyssesParallelAttention(sp_group, scatter_idx=2, gather_idx=1, use_sync=False)
            if ulysses_degree > 1
            else ring_runner
        )

        adapter = None
        batch = None
        if mode == "paged":
            adapter, diffusion_layer, _ = _build_real_adapter(
                device,
                seq_len,
                ulysses_degree=ulysses_degree,
            )
        else:
            diffusion_layer = _SmokeAttention().to(device)

        attention = _build_attention(
            diffusion_layer,
            strategy,
            paged=mode == "paged",
            ring_runner=ring_runner,
        )
        if adapter is not None:
            batch = adapter.prepare_batch(
                [
                    DiffusionPagedAttentionRow(
                        request_id="memory-timeline",
                        sequence_id=0,
                        query_len=seq_len,
                        seq_len=seq_len,
                    )
                ]
            )

        torch.manual_seed(20261029 + seq_len)
        qkv = torch.randn(
            3,
            1,
            seq_len,
            _NUM_HEADS,
            _HEAD_SIZE,
            dtype=_DTYPE,
            device=device,
        )
        global_query, global_key, global_value = qkv.unbind(0)
        local_tokens = seq_len // world_size
        local_slice = slice(local_rank * local_tokens, (local_rank + 1) * local_tokens)
        query = global_query[:, local_slice].contiguous()
        key = global_key[:, local_slice].contiguous()
        value = global_value[:, local_slice].contiguous()
        run_forward = partial(
            attention._forward_impl,
            query,
            key,
            value,
            attn_metadata=None,
        )

        traces: list[list[tuple[float, int]]] = []
        forward_ends: list[float] = []
        activation = adapter.activate(batch) if adapter is not None else nullcontext()
        with set_forward_context(), activation:
            for _ in range(warmup):
                output = run_forward()
                current_omni_platform.synchronize()
                del output

            for _ in range(iters):
                samples, forward_end_ms = _capture_trace(
                    run_forward,
                    device,
                    tail_ms,
                )
                traces.append(samples)
                forward_ends.append(forward_end_ms)

        worst_forward_ends: list[float] = []
        for forward_end_ms in forward_ends:
            value_tensor = torch.tensor([forward_end_ms], dtype=torch.float64, device=device)
            dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
            worst_forward_ends.append(float(value_tensor.item()))
        median_forward_end_ms = statistics.median(worst_forward_ends)
        grid_end_ms = median_forward_end_ms + tail_ms
        grid_ms = [grid_end_ms * index / (grid_points - 1) for index in range(grid_points)]

        worst_rank_curves: list[torch.Tensor] = []
        for trace in traces:
            curve = torch.tensor(_step_resample(trace, grid_ms), dtype=torch.float64, device=device)
            dist.all_reduce(curve, op=dist.ReduceOp.MAX)
            worst_rank_curves.append(curve.cpu())
        median_curve = torch.stack(worst_rank_curves).median(dim=0).values.tolist()

        baseline_bytes = statistics.median([curve[0].item() for curve in worst_rank_curves])
        peak_bytes = max(median_curve)
        final_bytes = median_curve[-1]
        result = {
            "mode": mode,
            "seq_len": seq_len,
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "iters": iters,
            "median_forward_end_ms": median_forward_end_ms,
            "baseline_bytes": baseline_bytes,
            "peak_bytes": peak_bytes,
            "final_bytes": final_bytes,
            "grid_ms": grid_ms,
            "median_allocated_bytes": median_curve,
        }
        if local_rank == 0:
            result_queue.put(result)
    finally:
        destroy_distributed_env()


def _run_mode(args: argparse.Namespace, mode: str) -> dict:
    world_size = args.ulysses_degree * args.ring_degree
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.SimpleQueue()
    torch.multiprocessing.spawn(
        _worker,
        args=(
            world_size,
            get_file_store_init_method(),
            mode,
            args.seq_len,
            args.warmup,
            args.iters,
            args.ulysses_degree,
            args.ring_degree,
            args.tail_ms,
            args.grid_points,
            result_queue,
        ),
        nprocs=world_size,
        join=True,
    )
    return result_queue.get()


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_plot(dense: dict, paged: dict, output_path: str) -> None:
    width, height = 1400, 900
    left, right, top, bottom = 155, 75, 100, 130
    plot_width = width - left - right
    plot_height = height - top - bottom
    title_font = _font(30)
    label_font = _font(22)
    tick_font = _font(18)

    dense_x = dense["grid_ms"]
    paged_x = paged["grid_ms"]
    dense_y = [value / (1024**2) for value in dense["median_allocated_bytes"]]
    paged_y = [value / (1024**2) for value in paged["median_allocated_bytes"]]
    x_max = max(dense_x[-1], paged_x[-1])
    y_max = max(dense_y + paged_y) * 1.10

    def x_pixel(value: float) -> int:
        return left + round(value / x_max * plot_width)

    def y_pixel(value: float) -> int:
        return top + plot_height - round(value / y_max * plot_height)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    grid_color = (220, 225, 230)
    axis_color = (35, 40, 45)
    dense_color = (35, 105, 190)
    paged_color = (210, 65, 55)

    for index in range(6):
        fraction = index / 5
        x_value = x_max * fraction
        x = x_pixel(x_value)
        draw.line((x, top, x, top + plot_height), fill=grid_color, width=1)
        x_label = f"{x_value:.2f}"
        bbox = draw.textbbox((0, 0), x_label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, top + plot_height + 12), x_label, fill=axis_color, font=tick_font)

        y_value = y_max * fraction
        y = y_pixel(y_value)
        draw.line((left, y, left + plot_width, y), fill=grid_color, width=1)
        y_label = f"{y_value:.1f}"
        bbox = draw.textbbox((0, 0), y_label, font=tick_font)
        draw.text((left - (bbox[2] - bbox[0]) - 14, y - 10), y_label, fill=axis_color, font=tick_font)

    draw.line((left, top, left, top + plot_height), fill=axis_color, width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill=axis_color, width=2)

    dense_points = [(x_pixel(x), y_pixel(y)) for x, y in zip(dense_x, dense_y, strict=True)]
    paged_points = [(x_pixel(x), y_pixel(y)) for x, y in zip(paged_x, paged_y, strict=True)]
    dense_points.append((x_pixel(x_max), y_pixel(dense_y[-1])))
    paged_points.append((x_pixel(x_max), y_pixel(paged_y[-1])))
    draw.line(dense_points, fill=dense_color, width=5)
    draw.line(paged_points, fill=paged_color, width=5)

    strategy = (
        f"Ulysses({dense['ulysses_degree']}) x Ring({dense['ring_degree']})"
        if dense["ulysses_degree"] > 1
        else f"Ring({dense['ring_degree']})"
    )
    title = f"Median CUDA allocated-memory timeline: {strategy}, S={dense['seq_len']}"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 28), title, fill=axis_color, font=title_font)

    x_label = "Elapsed time from forward start (ms)"
    bbox = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, height - 55), x_label, fill=axis_color, font=label_font)

    y_layer = Image.new("RGBA", (620, 50), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_layer)
    y_draw.text((0, 8), "Median max-rank CUDA memory allocated (MiB)", fill=axis_color, font=label_font)
    y_layer = y_layer.rotate(90, expand=True)
    image.paste(y_layer, (20, (height - y_layer.height) // 2), y_layer)

    legend_y = top + 18
    draw.line((left + 20, legend_y, left + 80, legend_y), fill=dense_color, width=5)
    draw.text((left + 90, legend_y - 12), "Dense Ring", fill=dense_color, font=label_font)
    draw.line((left + 270, legend_y, left + 330, legend_y), fill=paged_color, width=5)
    draw.text((left + 340, legend_y - 12), "Paged Ring", fill=paged_color, font=label_font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=11)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=4)
    parser.add_argument("--tail-ms", type=float, default=1.0)
    parser.add_argument("--grid-points", type=int, default=240)
    parser.add_argument("--output", default="paged_ring_memory_timeline.png")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    world_size = args.ulysses_degree * args.ring_degree
    if args.seq_len <= 0 or args.seq_len % world_size:
        raise SystemExit("--seq-len must be positive and divisible by the SP world size")
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup must be non-negative")
    if args.tail_ms <= 0 or args.grid_points < 2:
        raise SystemExit("tail and grid size must be positive")
    if current_omni_platform.get_device_count() < world_size:
        raise SystemExit(f"Need {world_size} visible GPUs")

    dense = _run_mode(args, "dense")
    paged = _run_mode(args, "paged")
    _draw_plot(dense, paged, args.output)
    payload = {"dense": dense, "paged": paged}
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "MEMORY_TIMELINE_SUMMARY="
        + json.dumps(
            {
                mode: {
                    "baseline_mib": result["baseline_bytes"] / (1024**2),
                    "peak_mib": result["peak_bytes"] / (1024**2),
                    "final_mib": result["final_bytes"] / (1024**2),
                    "median_forward_end_ms": result["median_forward_end_ms"],
                }
                for mode, result in payload.items()
            },
            sort_keys=True,
        )
    )
    print(f"Plot written to {args.output}")


if __name__ == "__main__":
    main()
