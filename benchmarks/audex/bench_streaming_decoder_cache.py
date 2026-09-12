#!/usr/bin/env python3
"""Microbenchmark the Audex streaming decoder cache path without model weights.

The legacy runner reproduces the previous per-chunk ``torch.cat`` cache,
per-layer attention-mask construction, and RoPE position readback. The optimized
runner calls the production ``CausalVocosBackbone.forward`` path.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable

import torch
from torch import Tensor

from vllm_omni.model_executor.models.audex.speech_decoder.modeling_audex_causal_speech_decoder import (
    CausalCodecDecoderCache,
    CausalVocosBackbone,
)


class _LegacyConcatCache:
    def __init__(self) -> None:
        self.key_values: dict[int, tuple[Tensor, Tensor]] = {}
        self.position = 0

    def input_positions(self, length: int, device: torch.device) -> Tensor:
        return torch.arange(self.position, self.position + length, device=device).unsqueeze(0)

    def update(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if layer_idx in self.key_values:
            prev_key, prev_value = self.key_values[layer_idx]
            key = torch.cat([prev_key, key], dim=2)
            value = torch.cat([prev_value, value], dim=2)
        self.key_values[layer_idx] = (key, value)
        return key, value

    def advance(self, length: int) -> None:
        self.position += length


def _legacy_forward(model: CausalVocosBackbone, x: Tensor, cache: _LegacyConcatCache) -> Tensor:
    input_pos = cache.input_positions(x.size(1), x.device).expand(x.size(0), -1)
    for block in model.transformers:
        # The old RoPE forward did this once for Q and once for K in every layer.
        for _ in range(2):
            needed_seq_len = int(input_pos.max().item()) + 1
            model.rotary_embed.prepare(needed_seq_len, x.device)
        # No shared mask reproduces the old per-layer mask construction.
        x = block(x, cache=cache, input_pos=input_pos)
    cache.advance(x.size(1))
    return model.final_layer_norm(x)


def _decode_stream(
    model: CausalVocosBackbone,
    hidden_states: Tensor,
    chunk_frames: int,
    *,
    legacy: bool,
) -> Tensor:
    cache = _LegacyConcatCache() if legacy else CausalCodecDecoderCache()
    outputs = []
    for start in range(0, hidden_states.size(1), chunk_frames):
        chunk = hidden_states[:, start : start + chunk_frames]
        if legacy:
            outputs.append(_legacy_forward(model, chunk, cache))
        else:
            outputs.append(model(chunk, cache=cache))
    return torch.cat(outputs, dim=1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


def _measure_pair(
    legacy_fn: Callable[[], Tensor],
    optimized_fn: Callable[[], Tensor],
    device: torch.device,
    warmups: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmups):
        legacy_fn()
        optimized_fn()
    _synchronize(device)

    timings: dict[str, list[float]] = {"legacy": [], "optimized": []}
    runners = {"legacy": legacy_fn, "optimized": optimized_fn}
    for iteration in range(iterations):
        order = ("legacy", "optimized") if iteration % 2 == 0 else ("optimized", "legacy")
        for name in order:
            start = time.perf_counter()
            runners[name]()
            _synchronize(device)
            timings[name].append((time.perf_counter() - start) * 1000)
    return timings["legacy"], timings["optimized"]


def _peak_extra_memory(fn: Callable[[], Tensor], device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(device)
    before = torch.accelerator.memory_allocated(device)
    fn()
    _synchronize(device)
    return torch.accelerator.max_memory_allocated(device) - before


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--total-frames", type=int, default=512)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 SDPA is not supported on CPU; use float32 or bfloat16")
    if args.hidden_dim % args.heads != 0:
        raise ValueError("hidden-dim must be divisible by heads")
    head_dim = args.hidden_dim // args.heads

    torch.manual_seed(args.seed)
    model = CausalVocosBackbone(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        pos_meb_dim=head_dim,
    ).to(device=device, dtype=dtype)
    model.eval()
    hidden_states = torch.randn(
        args.batch_size,
        args.total_frames,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    )

    def legacy_fn() -> Tensor:
        return _decode_stream(model, hidden_states, args.chunk_frames, legacy=True)

    def optimized_fn() -> Tensor:
        return _decode_stream(model, hidden_states, args.chunk_frames, legacy=False)

    with torch.inference_mode():
        legacy_output = legacy_fn()
        optimized_output = optimized_fn()
        tolerance = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(optimized_output, legacy_output, rtol=tolerance, atol=tolerance)

        legacy_ms, optimized_ms = _measure_pair(
            legacy_fn,
            optimized_fn,
            device,
            args.warmups,
            args.iterations,
        )
        legacy_peak = _peak_extra_memory(legacy_fn, device)
        optimized_peak = _peak_extra_memory(optimized_fn, device)

    legacy_median = statistics.median(legacy_ms)
    optimized_median = statistics.median(optimized_ms)
    legacy_stddev = statistics.pstdev(legacy_ms)
    optimized_stddev = statistics.pstdev(optimized_ms)
    chunks = (args.total_frames + args.chunk_frames - 1) // args.chunk_frames
    max_diff = (optimized_output.float() - legacy_output.float()).abs().max().item()

    print(
        f"device={device} dtype={dtype} batch={args.batch_size} hidden={args.hidden_dim} "
        f"depth={args.depth} heads={args.heads} frames={args.total_frames} chunk={args.chunk_frames}"
    )
    print(
        f"legacy median:    {legacy_median:.3f} ms/request ({legacy_median / chunks:.3f} ms/chunk), "
        f"stddev={legacy_stddev:.3f}, range={min(legacy_ms):.3f}-{max(legacy_ms):.3f}"
    )
    print(
        f"optimized median: {optimized_median:.3f} ms/request ({optimized_median / chunks:.3f} ms/chunk), "
        f"stddev={optimized_stddev:.3f}, range={min(optimized_ms):.3f}-{max(optimized_ms):.3f}"
    )
    print(f"speedup: {legacy_median / optimized_median:.3f}x")
    if legacy_peak is not None and optimized_peak is not None:
        mib = 1024**2
        print(f"extra peak allocated: legacy={legacy_peak / mib:.1f} MiB optimized={optimized_peak / mib:.1f} MiB")
    print(f"max absolute difference: {max_diff:.6g}")


if __name__ == "__main__":
    main()
