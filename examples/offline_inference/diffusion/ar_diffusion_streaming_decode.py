# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream LingBot-World 2.0 pixels from one stepwise request.

One ``AsyncOmni.generate()`` drives the whole rollout: each AR block leaves
``post_decode()`` as one video chunk, decoded through that session's own
temporal VAE cache rather than as an isolated clip. That is what this script
shows end to end -- chunk *N + 1* continues chunk *N*, so the streamed timeline
carries the same frame count as an offline whole-clip decode of the same
latents (9 frames for the opening chunk, then 12 per chunk) instead of losing
each block's opening frame to a restarted decoder.

The decode lives inside the pipeline, keyed by ``request_id``, so nothing here
owns a VAE: this file is a client of the served path, and the same chunks reach
a WebSocket client of ``/v1/realtime/video`` unchanged.

Examples::

    python examples/offline_inference/diffusion/ar_diffusion_streaming_decode.py \
        --image scene.png --prompt "a lit hallway" \
        --action-script actions.json --output-dir /tmp/lingbot-stream

``--action-script`` is a JSON list with one three-frame camera action list per
AR block, for example ``[[["w"], ["w"], ["w"]], [["a"], [], []]]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only for type hints -- the real imports stay deferred inside run() so
    # --help and the offline validation helpers don't require a CUDA-enabled
    # vLLM install.
    import numpy as np

_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
_FRAMES_PER_BLOCK = 3
_TEMPORAL_COMPRESSION = 4
# ((117 pixel frames - 1) / VAE temporal factor 4 + 1) / 3 latent frames.
_MAX_CHUNKS = 10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream decoded LingBot-World 2.0 video chunks from a single stepwise request.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default=_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--image", required=True, help="Initial RGB image.")
    parser.add_argument("--prompt", required=True, help="Scene prompt for the rollout.")
    parser.add_argument(
        "--action-script",
        required=True,
        help="JSON file holding one three-frame camera action list per AR block; at most 10 blocks.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for decoded frames and metadata.")
    parser.add_argument("--request-id", default="lingbot-world-stream", help="Request id; also the session id.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args(argv)


def _load_action_script(path: Path) -> list[list[list[str]]]:
    """Parse and validate the per-block camera action script.

    The shape is the one ``sampling_params.extra_args["camera_action_script"]``
    takes: one entry per AR block, each entry three per-frame action lists.
    """
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"--action-script is not valid JSON: {exc.msg}.") from None
    if not isinstance(value, list) or not value:
        raise ValueError("--action-script must be a non-empty JSON list of per-block action lists.")
    if len(value) > _MAX_CHUNKS:
        raise ValueError(
            f"--action-script must hold at most {_MAX_CHUNKS} blocks because the current LingBot "
            "image-condition horizon is 117 pixel frames."
        )
    script: list[list[list[str]]] = []
    for block_index, block in enumerate(value):
        if not isinstance(block, list) or len(block) != _FRAMES_PER_BLOCK:
            raise ValueError(f"--action-script block {block_index} must contain exactly three frame action lists.")
        frames: list[list[str]] = []
        for frame in block:
            if not isinstance(frame, list) or any(not isinstance(action, str) for action in frame):
                raise ValueError(f"--action-script block {block_index} frames must contain only action strings.")
            frames.append(list(frame))
        script.append(frames)
    return script


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    image = Path(args.image).expanduser().resolve()
    action_script = Path(args.action_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")
    if not action_script.is_file():
        raise ValueError("--action-script must point to an existing JSON file.")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1].")
    if not args.request_id.strip():
        raise ValueError("--request-id must contain non-whitespace text.")
    return image, action_script, output_dir


def _as_frames(value: Any) -> np.ndarray:
    """Normalize one streamed chunk to ``[T, H, W, 3]`` uint8.

    ``output_type="np"`` reaches the client as a float array in ``[0, 1]``,
    with or without the leading batch axis depending on how the formatter
    unwrapped it; a nested list is the ``"pil"`` shape. Anything else means
    the chunk is not pixels -- most likely the request still asked for
    latents -- and is worth failing on rather than saving.
    """
    import numpy as np
    import PIL.Image
    import torch

    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if value and all(isinstance(frame, PIL.Image.Image) for frame in value):
            return np.stack([np.asarray(frame.convert("RGB")) for frame in value])
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    if not isinstance(value, np.ndarray):
        raise RuntimeError(f"Expected decoded pixels from each AR block, got {type(value).__name__}.")
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[-1] != 3:
        raise RuntimeError(f"Expected one [T, H, W, 3] pixel chunk per AR block, got shape {tuple(value.shape)}.")
    if value.dtype != np.uint8:
        value = (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return value


def _chunk_metadata(output: Any) -> dict[str, Any]:
    multimodal = getattr(output, "multimodal_output", None) or {}
    metadata = multimodal.get("metadata") if isinstance(multimodal, dict) else None
    ar_diffusion = metadata.get("ar_diffusion") if isinstance(metadata, dict) else None
    if not isinstance(ar_diffusion, dict):
        raise RuntimeError("Streamed chunk is missing its ar_diffusion metadata envelope.")
    return dict(ar_diffusion)


async def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    image, action_script_path, output_dir = _validate_args(args)
    camera_action_script = _load_action_script(action_script_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports stay below pure input validation so --help and helper tests do not
    # require a CUDA-enabled vLLM installation.
    import numpy as np

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    num_chunks = len(camera_action_script)
    num_frames = (num_chunks * _FRAMES_PER_BLOCK - 1) * _TEMPORAL_COMPRESSION + 1

    engine = AsyncOmni(
        model=args.model,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        # One request, many chunks: the stepwise contract from #6844. Streaming
        # output is what turns each post_decode() into a delivered chunk.
        step_execution=True,
        diffusion_streaming_output=True,
        model_config={
            "ar_diffusion_height": args.height,
            "ar_diffusion_width": args.width,
            "ar_diffusion_kv_config": {
                "gpu_memory_fraction": args.gpu_memory_fraction,
                "warmup_cudagraph": True,
            },
        },
    )
    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=args.seed,
        # Pixels, not latents: this is what routes the chunk through the
        # session's streaming VAE decode instead of handing back a tensor.
        output_type="np",
        extra_args={"flow_shift": 5.0, "camera_action_script": camera_action_script},
    )
    prompt = {"prompt": args.prompt, "multi_modal_data": {"image": str(image)}}

    measurements: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    first_frame_at: float | None = None
    chunk_started = t_start
    try:
        async for output in engine.generate(prompt, sampling, request_id=args.request_id):
            images = getattr(output, "images", None)
            if not images:
                continue
            if len(images) != 1:
                raise RuntimeError("Expected exactly one decoded chunk per AR block.")
            frames = _as_frames(images[0])
            now = time.perf_counter()
            if first_frame_at is None:
                first_frame_at = now - t_start
            chunk_index = len(measurements)
            metadata = _chunk_metadata(output)

            np.save(output_dir / f"chunk_{chunk_index:03d}.npy", frames)
            (output_dir / f"chunk_{chunk_index:03d}.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            measurements.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_latency_seconds": now - chunk_started,
                    "frames": int(frames.shape[0]),
                    "frame_shape": list(frames.shape),
                    "metadata": metadata,
                }
            )
            chunk_started = now
            print(json.dumps(measurements[-1], sort_keys=True), flush=True)
    finally:
        engine.shutdown()

    delivered_frames = sum(int(chunk["frames"]) for chunk in measurements)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "request_id": args.request_id,
                "chunks_delivered": len(measurements),
                "chunks_expected": num_chunks,
                "frames_delivered": delivered_frames,
                "frames_expected": num_frames,
                # The point of session-owned decode: a streamed rollout carries
                # the same frame timeline as an offline decode of the same
                # latents. Per-chunk decode would deliver fewer, because every
                # block would re-expand its own opening frame.
                "matches_offline_frame_timeline": delivered_frames == num_frames,
                "time_to_first_frame_seconds": first_frame_at,
                "total_seconds": time.perf_counter() - t_start,
                "chunks": measurements,
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
