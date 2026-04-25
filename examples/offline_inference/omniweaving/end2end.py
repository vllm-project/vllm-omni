from __future__ import annotations

import argparse
import time

try:
    from omniweaving_common import (
        build_prompt_payload,
        ensure_cuda_available,
        extract_stage_metrics,
        save_video,
        video_payload_to_uint8,
    )
except ModuleNotFoundError:
    from examples.offline_inference.omniweaving.omniweaving_common import (
        build_prompt_payload,
        ensure_cuda_available,
        extract_stage_metrics,
        save_video,
        video_payload_to_uint8,
    )


# T2V 480p: landscape grid (common diffusers 832-wide convention).
# I2V 480p (official / bench): `aspect_ratio=26:15` → 480×848 (see OMNIWEAVING_I2V_MI2V_PERF.md).
RESOLUTION_PRESETS = {
    "480p": {"height": 480, "width": 832},
    "720p": {"height": 720, "width": 1280},
}
RESOLUTION_PRESETS_I2V_480P = {"height": 480, "width": 848}

FLOW_SHIFT_PRESETS = {
    "t2v": {"480p": 5.0, "720p": 9.0},
    # Official OmniWeaving I2V 480p aligned case uses flow_shift=7.0 (same table as perf doc).
    "i2v": {"480p": 7.0, "720p": 7.0},
    "mi2v": {"480p": 7.0, "720p": 7.0},
}

# Required by DiffusionWorker.re_init_pipeline whenever custom_pipeline_args is non-empty.
_OMNIWEAVING_PIPELINE_CLASS = "vllm_omni.diffusion.models.omniweaving.pipeline_omniweaving.OmniWeavingPipeline"

VIDEO_BITRATE_PRESETS = {
    "480p": "6M",
    "720p": "12M",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline inference example for OmniWeaving.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path or HuggingFace repo ID of the OmniWeaving model."
    )
    parser.add_argument(
        "--qwen-path",
        type=str,
        default=None,
        help="Optional local path for Qwen2.5-VL when running in offline environments.",
    )
    parser.add_argument(
        "--siglip-path",
        type=str,
        default=None,
        help="Optional local path for SigLIP when running in offline environments.",
    )
    parser.add_argument(
        "--t2v-path",
        type=str,
        default=None,
        help="Optional local path for HunyuanVideo-1.5 Diffusers assets (VAE/scheduler).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cute bear wearing a Christmas hat moving naturally.",
        help="Text prompt.",
    )
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt for CFG.")
    parser.add_argument(
        "--image-path", type=str, default=None, help="Optional image path for image-to-video generation."
    )
    parser.add_argument(
        "--image-paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional multiple image paths for multi-image conditioning.",
    )
    parser.add_argument("--output", type=str, default="omniweaving_output.mp4", help="Output MP4 path.")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="CFG guidance scale.")
    parser.add_argument(
        "--resolution",
        type=str,
        choices=sorted(RESOLUTION_PRESETS),
        default="480p",
        help="Resolution preset. For I2V/MI2V with 480p and no explicit --height/--width, uses official-aligned "
        "480×848 (26:15). T2V 480p remains 480×832. --height/--width override.",
    )
    parser.add_argument("--height", type=int, default=None, help="Video height.")
    parser.add_argument("--width", type=int, default=None, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=33, help="Number of frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS.")
    parser.add_argument(
        "--video-bitrate",
        type=str,
        default=None,
        help="Optional ffmpeg video bitrate override, e.g. 12M. Defaults to a resolution-aware quality preset.",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        help="ffmpeg codec used when writing the output MP4.",
    )
    parser.add_argument(
        "--vae-use-slicing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="VAE spatial slicing during decode (default: off; lowers peak VRAM on some VAEs).",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VAE spatial tiling during decode (default: on; use --no-vae-use-tiling to disable).",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Optional VAE patch-parallel size for decode.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "gguf"],
        help="Optional transformer quantization method.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Scheduler flow shift. Defaults to a mode-aware value based on the selected resolution.",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to expose stage durations in the output.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading the model.",
    )
    return parser


def _infer_mode(args: argparse.Namespace) -> str:
    if args.image_paths:
        return "mi2v" if len(args.image_paths) > 1 else "i2v"
    if args.image_path:
        return "i2v"
    return "t2v"


def _infer_resolution(height: int, width: int, fallback: str) -> str:
    if height == RESOLUTION_PRESETS_I2V_480P["height"] and width == RESOLUTION_PRESETS_I2V_480P["width"]:
        return "480p"
    for name, preset in RESOLUTION_PRESETS.items():
        if preset["height"] == height and preset["width"] == width:
            return name
    return fallback


def main(args: argparse.Namespace) -> None:
    ensure_cuda_available()
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    mode = _infer_mode(args)
    resolution_preset = dict(RESOLUTION_PRESETS[args.resolution])
    # Match official 480p I2V / MI2V output size (26:15) when user does not override dimensions.
    if args.resolution == "480p" and mode in ("i2v", "mi2v") and not args.height and not args.width:
        resolution_preset.update(RESOLUTION_PRESETS_I2V_480P)
    args.height = args.height or resolution_preset["height"]
    args.width = args.width or resolution_preset["width"]
    resolved_resolution = _infer_resolution(args.height, args.width, args.resolution)
    flow_shift = args.flow_shift
    if flow_shift is None:
        flow_shift = FLOW_SHIFT_PRESETS[mode][resolved_resolution]
    video_bitrate = args.video_bitrate or VIDEO_BITRATE_PRESETS.get(resolved_resolution)

    print(f"Loading OmniWeaving model from {args.model} (TP={args.tensor_parallel_size})...")

    custom_pipeline_args = {}
    if args.qwen_path:
        custom_pipeline_args["qwen_path"] = args.qwen_path
    if args.siglip_path:
        custom_pipeline_args["siglip_path"] = args.siglip_path
    if args.t2v_path:
        custom_pipeline_args["t2v_path"] = args.t2v_path
    if custom_pipeline_args:
        custom_pipeline_args["pipeline_class"] = _OMNIWEAVING_PIPELINE_CLASS

    omni = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        flow_shift=flow_shift,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        custom_pipeline_args=custom_pipeline_args or None,
    )
    try:
        prompt_payload = build_prompt_payload(
            args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            image_paths=args.image_paths,
        )
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            fps=args.fps,
        )

        print(
            f"Dispatching request: {args.width}x{args.height}, "
            f"{args.num_frames} frames, steps={args.num_inference_steps}, "
            f"flow_shift={flow_shift}, video_bitrate={video_bitrate or 'auto'}."
        )
        started_at = time.perf_counter()
        outputs = omni.generate(
            prompts=prompt_payload,
            sampling_params_list=sampling_params,
        )
        latency_s = time.perf_counter() - started_at
        if not outputs:
            raise RuntimeError("Omni.generate() returned an empty output list.")

        video_uint8 = video_payload_to_uint8(outputs[0])
        save_video(
            video_uint8,
            args.output,
            fps=args.fps,
            codec=args.video_codec,
            bitrate=video_bitrate,
        )
        stage_durations, peak_memory_gib = extract_stage_metrics(outputs[0])

        print(f"Saved video to {args.output}")
        print(f"Latency: {latency_s:.2f}s")
        if peak_memory_gib is not None:
            print(f"Peak VRAM: {peak_memory_gib:.2f} GiB")
        if stage_durations:
            print(f"Stage durations: {stage_durations}")
    finally:
        omni.close()


if __name__ == "__main__":
    main(build_parser().parse_args())
