# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one or more LingBot-World 2.0 realtime AR-Diffusion sessions.

This example exercises the internal session API. It is not a public HTTP or
WebSocket protocol. Each JSONL event advances the world by one three-latent
frame AR block and writes the latent plus its identity and replica-route
metadata. With ``--deploy-config`` and ``--num-sessions 2`` this is also the
hardware driver for session-affine Replica-DP validation. The current
scoped implementation supports at most ten ticks per generation epoch because
its image condition is bounded to 117 pixel frames; reset or create a session
to start a new world.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
_CAMERA_ACTION_SCHEMA = "lingbot.camera_actions.v1"
_FRAMES_PER_BLOCK = 3
# ((117 pixel frames - 1) / VAE temporal factor 4 + 1) / 3 latent frames.
_MAX_REALTIME_TICKS = 10


def _camera_event_data(frames: list[list[str]]) -> dict[str, Any]:
    """Build the event-side script consumed by LingBotCameraControlReducer."""
    return {"mode": "script", "frames": frames}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run in-process realtime LingBot-World 2.0 sessions.")
    parser.add_argument("--model", default=_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--image", required=True, help="Initial RGB image.")
    parser.add_argument("--prompt", required=True, help="Initial scene prompt.")
    parser.add_argument(
        "--events",
        required=True,
        help="JSONL file with one prompt/action event per AR block; at most 10 events.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for chunk latents and metadata.")
    parser.add_argument(
        "--session-id",
        default="lingbot-world",
        help="Session id, or the prefix when --num-sessions is greater than one.",
    )
    parser.add_argument("--num-sessions", type=int, default=1, help="Number of sessions driven concurrently.")
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Optional deploy YAML. Required for a replicated stage; direct mode keeps the original single engine.",
    )
    parser.add_argument(
        "--require-distinct-replicas",
        action="store_true",
        help="Fail unless every session is placed on a different replica after its first tick.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.1)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size in direct mode; deploy YAML owns TP when --deploy-config is set.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args(argv)


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"events line {line_number} is not valid JSON: {exc.msg}.") from None
        if not isinstance(value, dict):
            raise ValueError(f"events line {line_number} must be a JSON object.")
        event_id = value.get("event_id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError(f"events line {line_number} requires a non-negative integer event_id.")
        prompt = value.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError(f"events line {line_number} prompt must be a non-empty string.")
        frames = value.get("frames")
        if frames is not None:
            if not isinstance(frames, list) or len(frames) != _FRAMES_PER_BLOCK:
                raise ValueError(f"events line {line_number} frames must contain exactly three lists.")
            for frame in frames:
                if not isinstance(frame, list) or any(not isinstance(action, str) for action in frame):
                    raise ValueError(f"events line {line_number} frames must contain only action strings.")
        if prompt is None and frames is None:
            raise ValueError(f"events line {line_number} must update prompt and/or frames.")
        events.append({"event_id": event_id, "prompt": prompt, "frames": frames})
    if not events:
        raise ValueError("events file must contain at least one event.")
    if len(events) > _MAX_REALTIME_TICKS:
        raise ValueError(
            "events file must contain at most "
            f"{_MAX_REALTIME_TICKS} events because the current LingBot realtime "
            "image-condition horizon is 117 pixel frames."
        )
    event_ids = [event["event_id"] for event in events]
    if event_ids != sorted(set(event_ids)):
        raise ValueError("event_id values must be unique and strictly increasing.")
    return events


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    image = Path(args.image).expanduser().resolve()
    events = Path(args.events).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    deploy_config = Path(args.deploy_config).expanduser().resolve() if args.deploy_config else None
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")
    if not events.is_file():
        raise ValueError("--events must point to an existing JSONL file.")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    if args.num_sessions <= 0:
        raise ValueError("--num-sessions must be positive.")
    if args.require_distinct_replicas and args.num_sessions < 2:
        raise ValueError("--require-distinct-replicas needs at least two sessions.")
    if deploy_config is not None and not deploy_config.is_file():
        raise ValueError("--deploy-config must point to an existing YAML file.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1].")
    return image, events, output_dir, deploy_config


async def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    image, events_path, output_dir, deploy_config = _validate_args(args)
    events = _load_events(events_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports stay below pure input validation so --help and helper tests do not
    # require a CUDA-enabled vLLM installation.
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.lingbot_world.actions import LingBotCameraControlReducer
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import (
        ARDiffusionSessionEvent,
        ARDiffusionSessionManager,
        ARDiffusionWorkerLifecycle,
    )
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine_kwargs: dict[str, Any] = {"model": args.model}
    if deploy_config is None:
        engine_kwargs.update(
            {
                "engine_backend": "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
                "enforce_eager": args.enforce_eager,
                "parallel_config": DiffusionParallelConfig(tensor_parallel_size=args.tensor_parallel_size),
                "max_num_seqs": 1,
                "model_config": {
                    "ar_diffusion_height": args.height,
                    "ar_diffusion_width": args.width,
                    "ar_diffusion_kv_config": {
                        "gpu_memory_fraction": args.gpu_memory_fraction,
                        "warmup_cudagraph": True,
                    },
                },
            },
        )
    else:
        # The deploy config owns engine selection, memory, compile mode,
        # replica count, device partitioning, and per-replica TP.
        engine_kwargs["deploy_config"] = str(deploy_config)
    engine = AsyncOmni(**engine_kwargs)
    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=9,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=args.seed,
        output_type="latent",
        extra_args={"flow_shift": 5.0},
    )
    consumer = ARDiffusionOmniTickConsumer(
        engine,
        prompt_provider=lambda tick: {
            "prompt": tick.prompt,
            "multi_modal_data": {"image": str(image)},
        },
        sampling_params_list=[sampling],
        diffusion_stage_id=0,
    )
    manager = ARDiffusionSessionManager(
        tick_consumer=consumer,
        lifecycle=ARDiffusionWorkerLifecycle(
            engine,
            stage_ids=[0],
            timeout=180.0,
            # Ticks are pinned to the replica owning the session's AR state;
            # reset/close hand the id back to the load balancer.
            route_release=consumer.release_session_route,
        ),
        max_pending_events=32,
        control_reducer_factory=LingBotCameraControlReducer,
    )
    session_ids = (
        [args.session_id]
        if args.num_sessions == 1
        else [f"{args.session_id}-{index}" for index in range(args.num_sessions)]
    )
    sessions = {session_id: await manager.create_session(session_id) for session_id in session_ids}
    measurements: dict[str, list[dict[str, Any]]] = {session_id: [] for session_id in session_ids}
    rounds: list[dict[str, float | int]] = []
    route_owners: dict[str, int | None] = {session_id: None for session_id in session_ids}
    stage_pool = engine.engine.stage_pools[0]

    async def run_one_chunk(session_id: str, chunk_index: int) -> None:
        started = time.perf_counter()
        output = await sessions[session_id].next_chunk()
        elapsed = time.perf_counter() - started
        if len(output.images) != 1 or not isinstance(output.images[0], torch.Tensor):
            raise RuntimeError("Expected one latent tensor from each realtime LingBot chunk.")
        latent = output.images[0].detach().float().cpu()
        metadata = consumer.chunk_metadata(output).to_dict()
        is_finite = bool(torch.isfinite(latent).all())
        if not is_finite:
            raise RuntimeError(f"Session {session_id!r} chunk {chunk_index} contains non-finite values.")
        replica_id = stage_pool.get_session_replica_id(session_id)
        if replica_id is None:
            raise RuntimeError(f"Session {session_id!r} completed a tick without a replica route.")
        previous_owner = route_owners[session_id]
        if previous_owner is not None and replica_id != previous_owner:
            raise RuntimeError(f"Session {session_id!r} moved from replica {previous_owner} to {replica_id}.")
        route_owners[session_id] = replica_id

        session_dir = output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        torch.save(latent, session_dir / f"chunk_{chunk_index:03d}.pt")
        chunk_record = {
            "chunk_index": chunk_index,
            "latency_seconds": elapsed,
            "shape": list(latent.shape),
            "finite": is_finite,
            "replica_id": replica_id,
            "metadata": metadata,
        }
        (session_dir / f"chunk_{chunk_index:03d}.json").write_text(
            json.dumps(chunk_record, indent=2, sort_keys=True) + "\n"
        )
        measurements[session_id].append(chunk_record)
        print(json.dumps({"session_id": session_id, **chunk_record}, sort_keys=True), flush=True)

    ticks_started = time.perf_counter()
    try:
        for chunk_index, event in enumerate(events):
            controls = ()
            if event["frames"] is not None:
                controls = (
                    ARDiffusionControlInput(
                        track="camera",
                        schema=_CAMERA_ACTION_SCHEMA,
                        data=_camera_event_data(event["frames"]),
                    ),
                )
            await asyncio.gather(
                *(
                    sessions[session_id].accept_event(
                        ARDiffusionSessionEvent(
                            event_id=event["event_id"],
                            prompt=event["prompt"]
                            if event["prompt"] is not None
                            else (args.prompt if chunk_index == 0 else None),
                            controls=controls,
                        )
                    )
                    for session_id in session_ids
                )
            )
            # The first gather exercises concurrent route binding. Following
            # rounds assert that each session remains on its original owner.
            round_started = time.perf_counter()
            await asyncio.gather(*(run_one_chunk(session_id, chunk_index) for session_id in session_ids))
            rounds.append(
                {
                    "chunk_index": chunk_index,
                    "wall_seconds": time.perf_counter() - round_started,
                }
            )
            if chunk_index == 0 and args.require_distinct_replicas:
                owners = [route_owners[session_id] for session_id in session_ids]
                if len(set(owners)) != len(owners):
                    raise RuntimeError(
                        "Expected one distinct replica per session, but first-tick owners were "
                        f"{dict(zip(session_ids, owners, strict=True))}."
                    )
        tick_wall_seconds = time.perf_counter() - ticks_started
    finally:
        try:
            await asyncio.gather(*(manager.close_session(session_id) for session_id in session_ids))
            leaked_routes = {
                session_id: stage_pool.get_session_replica_id(session_id)
                for session_id in session_ids
                if stage_pool.get_session_replica_id(session_id) is not None
            }
            if leaked_routes:
                raise RuntimeError(f"Session routes were not released on close: {leaked_routes}.")
        finally:
            engine.shutdown()

    completed_chunks = sum(len(chunks) for chunks in measurements.values())
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "deploy_config": str(deploy_config) if deploy_config is not None else None,
                "resolution": [args.height, args.width],
                "num_sessions": args.num_sessions,
                "route_owners": route_owners,
                "tick_wall_seconds": tick_wall_seconds,
                "aggregate_chunks_per_second": completed_chunks / tick_wall_seconds,
                "aggregate_latent_frames_per_second": (completed_chunks * _FRAMES_PER_BLOCK / tick_wall_seconds),
                "rounds": rounds,
                "sessions": measurements,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary_path


def main(argv: Sequence[str] | None = None) -> Path:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    main()
