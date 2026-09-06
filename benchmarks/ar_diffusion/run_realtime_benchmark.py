# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI for the realtime AR-Diffusion multi-session benchmark.

Model-neutral by construction: the chunk shape comes from the pipeline's
declared ``ARDiffusionKVCacheSpec``, so pointing ``--model`` at another
AR-Diffusion pipeline needs no code change.

One value cannot come from the capability today. ``frames_per_block`` and every
other frame count in the spec are in *latent* frames, and nothing in the spec
converts them to delivered frames, so the playout grid is not derivable from
the capability alone. Until the spec declares it, the factor arrives as
``--vae-temporal-factor``.

Every reported number must carry its conditions. ``--note`` is repeatable and
its values are copied verbatim into the summary; the run refuses to start
without at least one, because a latency without model, checkpoint, resolution,
denoising steps and hardware is not a result.

Examples::

    # Ceiling: how fast can this device go at all?
    python -m benchmarks.ar_diffusion.run_realtime_benchmark \
        --model <checkpoint> --prompt "..." --num-sessions 1 \
        --mode saturating --chunks 32 \
        --note "checkpoint=<...>" --note "hw=1xRTX-PRO-6000" --note "res=480x832"

    # Baseline: state concurrency 2, execution concurrency 1.
    python -m benchmarks.ar_diffusion.run_realtime_benchmark \
        --model <checkpoint> --prompt "..." --num-sessions 2 \
        --mode saturating --chunks 32 --note ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.ar_diffusion.realtime_harness import (
    BenchmarkConfig,
    burst_arrivals,
    poisson_arrivals,
    run_benchmark,
)
from benchmarks.ar_diffusion.realtime_metrics import LoadMode, WorkloadProfile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark realtime AR-Diffusion sessions.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--prompt", required=True, help="Initial scene prompt for every session.")
    parser.add_argument("--image", default=None, help="Optional initial RGB image, if the pipeline needs one.")

    parser.add_argument("--num-sessions", type=int, default=1, help="Sessions to run concurrently.")
    parser.add_argument("--chunks", type=int, default=16, help="Chunks each session generates.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in LoadMode],
        default=LoadMode.SATURATING.value,
        help="saturating measures the ceiling; paced measures continuity against deadlines.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=16.0,
        help="Declared playout rate. With the chunk shape it fixes the release period.",
    )
    parser.add_argument(
        "--release-period",
        type=float,
        default=None,
        help="Override the release period in seconds. Defaults to frames_per_chunk / target_fps.",
    )
    parser.add_argument("--buffer-chunks", type=int, default=1, help="Playout buffer depth, in chunks.")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Poisson session arrivals per second. Omit for a burst (all sessions start together).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the arrival process.")
    parser.add_argument(
        "--non-causal-decoder",
        action="store_true",
        help="The decoder expands every latent frame identically, so the opening chunk is full length.",
    )
    parser.add_argument(
        "--vae-temporal-factor",
        type=int,
        default=1,
        help=(
            "Raw frames per latent frame. Supplied here because ARDiffusionKVCacheSpec "
            "declares frames_per_block in latent frames and carries no factor that "
            "converts it to delivered frames; see the module docstring."
        ),
    )

    parser.add_argument("--num-gpus", type=int, default=1, help="Devices in use, for frames per GPU-second.")
    parser.add_argument("--events-dir", type=Path, default=None, help="Directory for per-session events JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Write the run summary JSON here.")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Run conditions recorded verbatim in the summary. Repeatable, and at least one is required.",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-pending-events", type=int, default=8)
    return parser.parse_args(argv)


def build_config(
    args: argparse.Namespace,
    *,
    frames_per_chunk: int,
    frames_per_first_chunk: int | None = None,
    resident_bytes: int | None = None,
) -> BenchmarkConfig:
    """Turn parsed arguments plus the pipeline's chunk shape into a config.

    ``frames_per_chunk`` is supplied by the caller after reading the pipeline
    capability, which is what keeps this module free of model knowledge.
    """
    if not args.note:
        raise ValueError(
            "At least one --note is required: a latency without model, checkpoint, "
            "resolution, denoising steps and hardware is not a result."
        )
    target_fps = args.target_fps
    if args.release_period is not None:
        if args.release_period <= 0:
            raise ValueError("--release-period must be positive.")
        # A release period override is expressed as the fps that produces it,
        # so the profile keeps a single source of truth for the playout grid.
        target_fps = frames_per_chunk / args.release_period

    profile = WorkloadProfile(
        frames_per_chunk=frames_per_chunk,
        frames_per_first_chunk=frames_per_first_chunk,
        target_fps=target_fps,
        buffer_chunks=args.buffer_chunks,
        resident_bytes_per_session=resident_bytes,
    )
    arrivals = (
        poisson_arrivals(args.num_sessions, rate_per_s=args.arrival_rate, seed=args.seed)
        if args.arrival_rate is not None
        else burst_arrivals(args.num_sessions)
    )
    return BenchmarkConfig(
        profile=profile,
        mode=LoadMode(args.mode),
        chunks_per_session=args.chunks,
        arrivals=arrivals,
        num_gpus=args.num_gpus,
        events_dir=args.events_dir,
    )


def frames_per_chunk_from_spec(spec: Any, *, vae_temporal_factor: int = 1, causal: bool = True) -> tuple[int, int]:
    """Raw frames a chunk delivers, as ``(steady_state, first_chunk)``.

    ``frames_per_block`` is read from the pipeline capability, which is what
    keeps the chunk shape out of this module. ``vae_temporal_factor`` has to be
    supplied by the caller: the spec expresses every frame count in latent
    frames and declares nothing that converts them to delivered frames, so no
    runtime component can compute the playout grid from the capability alone.

    The conversion is not a plain multiplication. A causal video decoder maps
    ``n`` latent frames to ``(n - 1) * factor + 1`` raw frames, so a session's
    opening chunk delivers fewer frames than every chunk after it -- which is
    also why a single declared integer could not express this mapping.
    """
    frames_per_block = getattr(spec, "frames_per_block", None)
    if isinstance(frames_per_block, bool) or not isinstance(frames_per_block, int) or frames_per_block <= 0:
        raise ValueError("The pipeline must declare a positive integer frames_per_block.")
    if isinstance(vae_temporal_factor, bool) or not isinstance(vae_temporal_factor, int) or vae_temporal_factor <= 0:
        raise ValueError("vae_temporal_factor must be a positive integer.")
    steady = frames_per_block * vae_temporal_factor
    first = (frames_per_block - 1) * vae_temporal_factor + 1 if causal else steady
    return steady, first


async def _main(args: argparse.Namespace) -> int:
    # Imported lazily so --help and the config path stay importable without a
    # CUDA build, which is also what lets the harness be tested on CPU.
    from benchmarks.ar_diffusion.engine_binding import build_realtime_backend
    from vllm_omni.experimental.ar_diffusion import ARDiffusionSessionManager

    backend = await build_realtime_backend(args)
    frames_per_chunk, frames_per_first_chunk = frames_per_chunk_from_spec(
        backend.spec,
        vae_temporal_factor=args.vae_temporal_factor,
        causal=not args.non_causal_decoder,
    )
    config = build_config(
        args,
        frames_per_chunk=frames_per_chunk,
        frames_per_first_chunk=frames_per_first_chunk,
        # Only the model-owned term is declared; runner-owned KV bytes are not
        # part of the spec, so this under-reports total residency.
        resident_bytes=getattr(backend.spec, "model_owned_state_bytes_per_session", None),
    )

    manager: ARDiffusionSessionManager = backend.manager

    async def factory(session_id: str):
        return await manager.create_session(session_id)

    summary = await run_benchmark(config, factory, notes=tuple(args.note))
    payload = json.dumps(summary.to_dict(), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_main(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
