# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded Matrix-Game qualification with real geometry frames and optional oracle.

Run the reference and Omni modes in separate processes on the same GPU/software
stack. See recipes/Skywork/Matrix-Game-4.0.md for the pinned reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.platforms import current_omni_platform


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument(
        "--input", required=True, type=Path, help="uint8 THWC .npy geometry frames at the selected size"
    )
    parser.add_argument("--captions", type=Path, help="Frame-aligned Racing caption JSONL; enables dynamic prompts")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--output", required=True, type=Path, help="output prefix for .npy frames and .json metrics")
    parser.add_argument("--mode", choices=("omni", "reference"), default="omni")
    parser.add_argument("--reference-runtime", choices=("native", "rtx5090"), default="native")
    parser.add_argument("--scene", choices=("castle", "ancient-egypt", "racing"), default="castle")
    parser.add_argument("--frames", type=int, default=241)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-graph", action="store_true", help="Use native Omni steady-state graph replay")
    parser.add_argument("--fp8", action="store_true", help="Set the shared quantization_config='fp8' for online FP8")
    parser.add_argument(
        "--attention-backend",
        choices=("TORCH_SDPA", "CUDNN_ATTN", "FLASH_ATTN"),
        default="TORCH_SDPA",
        help="Omni attention backend; the platform selector checks device and installed-kernel support",
    )
    parser.add_argument("--warmup-chunks", type=int, default=25)
    parser.add_argument("--compare", type=Path, help="reference output .npy to compare against")
    parser.add_argument("--require-exact", action="store_true", help="Fail if compared RGB frames are not identical")
    args = parser.parse_args()
    if args.frames < 1 or (args.frames - 1) % 4 or args.warmup_chunks < 0:
        parser.error("--frames must be 1+4k and --warmup-chunks must be nonnegative")
    if args.require_exact and args.compare is None:
        parser.error("--require-exact needs --compare")
    if args.mode == "reference" and args.attention_backend != "TORCH_SDPA":
        parser.error("--attention-backend selects the Omni backend, not the reference runtime")
    if (args.height, args.width) not in {(480, 864), (736, 1280)}:
        parser.error("Use 864x480 or experimental 1280x736")
    if args.mode == "reference" and (args.height, args.width) != (480, 864):
        parser.error("The pinned reference runtime only supports 864x480")
    if args.captions is not None and args.scene != "racing":
        parser.error("--captions requires --scene racing")
    return args


def load_captions(path: Path | None, frames: int) -> dict[int, str]:
    """Require one native caption per opening/four-frame chunk, without gaps."""
    if path is None:
        return {}
    captions, expected = {}, 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        size = 1 if expected == 0 else 4
        prompt = row.get("prompt")
        if (
            row.get("frame_start") != expected
            or row.get("frame_end") != expected + size - 1
            or not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > 8192
        ):
            raise ValueError(f"Caption must cover frames {expected} through {expected + size - 1}")
        captions[expected] = prompt
        expected += size
        if expected >= frames:
            return captions
    raise ValueError("Caption file does not cover the requested frames")


def peak_memory() -> dict[str, float]:
    return {
        "peak_allocated_gib": torch.accelerator.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.accelerator.max_memory_reserved() / 1024**3,
    }


async def render_omni(args: argparse.Namespace, video: np.ndarray) -> tuple[np.ndarray, list[float], dict[str, float]]:
    from vllm_omni.diffusion.data import OmniDiffusionConfig, VideoOutputTransportConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.matrix_game4.pipeline_matrix_game4 import END_KEY, SESSION_KEY, START_KEY
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    captions = load_captions(args.captions, args.frames)
    config = OmniDiffusionConfig(
        model=str(args.resources),
        model_class_name="MatrixGame4Pipeline",
        enforce_eager=True,
        num_gpus=1,
        max_num_seqs=1,
        model_config={
            "scene": args.scene,
            "prompt_mode": "dynamic" if captions else "fixed",
            "cuda_graph": args.cuda_graph,
            "height": args.height,
            "width": args.width,
        },
        enable_prompt_embed_cache=bool(captions),
        prompt_embed_cache_size=8,
        quantization_config="fp8" if args.fp8 else None,
        distributed_executor_backend="uni",
        diffusion_attention_config={"default": {"backend": args.attention_backend}},
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=True),
    )
    engine = DiffusionEngine(config)
    memory = peak_memory()
    outputs, timings = [], []
    try:
        for start in [0, *range(1, args.frames, 4)]:
            size = 1 if start == 0 else 4
            # The input owns writable host storage, independent of the mmap.
            pixels = torch.from_numpy(np.array(video[start : start + size]))
            request = OmniDiffusionRequest(
                prompt={"prompt": captions.get(start, ""), "multi_modal_data": {"video": pixels}},
                request_id=f"matrix-{start}",
                sampling_params=OmniDiffusionSamplingParams(
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    output_type="np",
                    extra_args={SESSION_KEY: "qualification", START_KEY: start, END_KEY: False},
                ),
            )
            current_omni_platform.synchronize()
            begin = time.perf_counter()
            chunk = None
            async for results in engine.step_streaming(request):
                for result in results:
                    if result.images:
                        if len(result.images) != 1:
                            raise RuntimeError("Expected one batched video payload")
                        chunk = result.images[0]
            current_omni_platform.synchronize()
            timings.append(time.perf_counter() - begin)
            # The runner resets peaks per request; retain both startup and all chunks.
            memory = {name: max(memory[name], value) for name, value in peak_memory().items()}
            if chunk is None or chunk.shape != (1, size, args.height, args.width, 3) or chunk.dtype != np.uint8:
                raise RuntimeError(
                    f"Invalid Omni output at frame {start}: "
                    f"type={type(chunk)}, shape={getattr(chunk, 'shape', None)}, "
                    f"dtype={getattr(chunk, 'dtype', None)}"
                )
            outputs.append(chunk[0])
        close_request = OmniDiffusionRequest(
            prompt={"prompt": ""},
            request_id="matrix-close",
            sampling_params=OmniDiffusionSamplingParams(
                seed=args.seed,
                num_inference_steps=args.steps,
                output_type="np",
                extra_args={SESSION_KEY: "qualification", START_KEY: args.frames, END_KEY: True},
            ),
        )
        async for _ in engine.step_streaming(close_request):
            pass
    finally:
        engine.close()
    return np.concatenate(outputs), timings, memory


def render_reference(args: argparse.Namespace, video: np.ndarray) -> tuple[np.ndarray, list[float]]:
    # Reference-only imports: install the pinned oracle separately for this mode.
    from matrix_game.inference.loader import ModelComponents
    from matrix_game.scenes import add_model_arguments, resolve_model_arguments

    captions = load_captions(args.captions, args.frames)
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    options = parser.parse_args(
        [
            "--scene",
            args.scene,
            "--resources",
            str(args.resources),
            "--preset",
            "realtime",
            "--dit-runtime",
            args.reference_runtime,
            "--prompt-mode",
            "dynamic" if captions else "fixed",
            "--steps",
            "1000" if args.steps == 1 else "1000,500",
        ]
    )
    resolve_model_arguments(options)
    components = ModelComponents(options)
    components.load()
    stream = components.stream(seed=args.seed)
    current_prompt = options.prompt
    outputs, timings = [], []
    try:
        for start in [0, *range(1, args.frames, 4)]:
            size = 1 if start == 0 else 4
            # Match Omni's timed boundary: owned uint8 input through RGB output.
            raw = torch.from_numpy(np.array(video[start : start + size]))
            current_omni_platform.synchronize()
            begin = time.perf_counter()
            prompt = captions.get(start, current_prompt)
            if prompt != current_prompt:
                stream.set_prompt(components.caption_encoder(prompt))
                current_prompt = prompt
            pixels = raw.permute(0, 3, 1, 2).float()
            pixels = pixels * (2.0 / 255.0) - 1.0
            stream.push(pixels, frame_start=start, on_rgb=outputs.append)
            current_omni_platform.synchronize()
            timings.append(time.perf_counter() - begin)
    finally:
        stream.close()
    return torch.cat(outputs).numpy(), timings


def main() -> None:
    import vllm

    args = arguments()
    video = np.load(args.input, mmap_mode="r")
    if (
        video.dtype != np.uint8
        or video.ndim != 4
        or video.shape[1:] != (args.height, args.width, 3)
        or len(video) < args.frames
    ):
        raise ValueError(f"Input must contain enough uint8 RGB geometry frames at {args.width}x{args.height}")
    current_omni_platform.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    if args.mode == "omni":
        frames, timings, memory = asyncio.run(render_omni(args, video))
    else:
        frames, timings = render_reference(args, video)
        memory = peak_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), frames)
    measured = timings[args.warmup_chunks :]
    report = {
        "mode": args.mode,
        "scene": args.scene,
        "reference_runtime": args.reference_runtime if args.mode == "reference" else None,
        "frames": len(frames),
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "cuda_graph": args.cuda_graph if args.mode == "omni" else None,
        "quantization_config": ("fp8" if args.fp8 else None) if args.mode == "omni" else None,
        "attention_backend": args.attention_backend if args.mode == "omni" else None,
        "input": str(args.input),
        "captions": str(args.captions) if args.captions else None,
        "prompt_mode": "dynamic" if args.captions else "fixed",
        "resources": str(args.resources),
        "warmup_chunks": args.warmup_chunks,
        "chunk_seconds": timings,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "device": current_omni_platform.get_device_name(),
        "cuda_runtime": torch.version.cuda,
        **memory,
        "memory_scope": "allocator peaks observed after initialization and requests; internal resets may hide startup",
        "codec": "taew2_2",
        "precision": "fp8" if (args.fp8 if args.mode == "omni" else args.reference_runtime == "rtx5090") else "bf16",
        "timing_scope": "uint8 RGB input through RGB output; includes normalization/codec/clean KV and Omni engine; "
        "excludes loading, session close and the reported warmup chunks",
    }
    if measured:
        report.update(
            steady_fps=sum(1 if i == 0 else 4 for i in range(args.warmup_chunks, len(timings))) / sum(measured),
            chunk_p50_ms=float(np.percentile(measured, 50) * 1000),
            chunk_p95_ms=float(np.percentile(measured, 95) * 1000),
        )
    if args.compare:
        reference = np.load(args.compare, mmap_mode="r")
        if reference.shape != frames.shape or reference.dtype != frames.dtype:
            raise ValueError("Reference output geometry/dtype differs")
        # Accumulate per frame so long comparisons do not materialize several
        # full FP32 copies of an already large RGB video on the host.
        error_sum, error_max, exact = 0.0, 0.0, True
        for actual, expected in zip(frames, reference):
            absolute = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
            error_sum += absolute.sum(dtype=np.float64)
            error_max = max(error_max, float(absolute.max()))
            exact = exact and np.array_equal(actual, expected)
        report.update(
            pixel_mae=float(error_sum / frames.size),
            pixel_max_error=error_max,
            exact_match=exact,
        )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "chunk_seconds"}, indent=2))
    if args.require_exact and not report["exact_match"]:
        raise SystemExit("Quality gate failed: RGB output differs from the pinned reference")


if __name__ == "__main__":
    main()
