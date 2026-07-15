# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance-3B offline inference (single-stage).

Lance is BAGEL-lineage; this example covers the single-stage paths wired in
this PR:

    text2img  — image generation from text
    img2text  — image understanding (caption / VQA)

Examples:

    python examples/offline_inference/lance/end2end.py \
        --model bytedance-research/Lance \
        --prompts "a corgi astronaut on the moon, cinematic" \
        --output ./out

    python examples/offline_inference/lance/end2end.py \
        --model bytedance-research/Lance --modality img2text \
        --image-path /path/to/photo.jpg \
        --prompts "Describe this image in detail."
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bytedance-research/Lance", help="HF repo or local path.")
    p.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    p.add_argument("--txt-prompts", type=str, default=None, help="File with one prompt per line.")
    p.add_argument(
        "--modality",
        default="text2img",
        choices=["text2img", "img2text", "text2video", "video2text", "img2img", "video2video", "image2video"],
        help="Lance single-stage modality.",
    )
    p.add_argument("--image-path", type=str, default=None, help="Input image for img2text.")
    p.add_argument("--video-path", type=str, default=None, help="Input video file for video2text.")
    p.add_argument("--num-frames", type=int, default=25, help="Number of RGB frames for text2video (max 121).")
    p.add_argument("--video-height", type=int, default=480, help="Video frame height.")
    p.add_argument("--video-width", type=int, default=768, help="Video frame width.")
    p.add_argument("--height", type=int, default=None, help="Image height (t2i). Default = max_hw (1024).")
    p.add_argument("--width", type=int, default=None, help="Image width (t2i). Default = max_hw (1024).")
    p.add_argument(
        "--fps", type=int, default=12, help="Output video FPS when saving MP4 (matches upstream Lance's save_fps=12)."
    )
    p.add_argument("--output", type=str, default=".", help="Output directory.")
    p.add_argument("--steps", type=int, default=30, help="Denoising steps (Lance default 30).")
    p.add_argument("--cfg-text-scale", type=float, default=4.0, help="Text CFG scale (Lance default 4.0).")
    p.add_argument("--timestep-shift", type=float, default=3.5, help="Flow-match timestep shift (Lance default 3.5).")
    p.add_argument("--negative-prompt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-text-tokens", type=int, default=512, help="Max tokens for img2text generation.")
    p.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Sample (vs greedy) for x2t generation; greedy frequently emits an immediate EOS for Lance.",
    )
    p.add_argument("--no-sample", dest="do_sample", action="store_false", help="Disable sampling (greedy decoding).")
    p.add_argument(
        "--text-temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for x2t generation. Lance frequently emits an immediate EOS below ~0.7; 0.8 is a good default.",
    )
    p.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=(
            "Override the task system prompt (used by x2t_image / x2t_video for "
            'per-example QA instructions like "Look at the image carefully and '
            'answer the question."; without it x2t falls back to a '
            "caption-style default and the model describes instead of answering)."
        ),
    )
    # Lance is single-stage diffusion — no deploy YAML needed.  Required
    # engine knobs (``pipeline``, ``enforce_eager``, ``trust_remote_code``,
    # ``max_num_seqs=1`` …) are passed as flat kwargs to ``Omni`` below
    # and ``create_default_diffusion`` materializes the stage config.

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(p)
    return p.parse_args()


from vllm_omni.diffusion.models.lance.prompts import (  # noqa: E402
    VIDEO_PAD,
    VISION_END,
    VISION_START,
    render_lance_prompt,
)

# Visual block matching upstream Lance: <|vision_start|><|video_pad|><|vision_end|>
# (Lance's renderer uses ``<|video_pad|>`` even for image inputs by default.)
_VISION_BLOCK = f"{VISION_START}{VIDEO_PAD}{VISION_END}"


def _format_text2img_prompts(prompts):
    return [{"prompt": render_lance_prompt("t2i", p), "modalities": ["image"]} for p in prompts]


def _format_text2video_prompts(prompts, num_frames, video_h, video_w):
    extra = {"num_frames": num_frames, "video_height": video_h, "video_width": video_w}
    return [{"prompt": render_lance_prompt("t2v", p), "modalities": ["video"], "extra_args": extra} for p in prompts]


def _format_image2video_prompts(prompts, image_path, num_frames, video_h, video_w):
    """Image-to-Video (no first-frame pin): image + text → long video.

    Passes the input image via ``multi_modal_data.first_frame``.  The pipeline
    treats it as a 1-frame reference (VAE+ViT prefill) and generates a fresh
    multi-frame video at the requested ``num_frames`` × ``video_h`` ×
    ``video_w`` shape.
    """
    import os as _os

    if not image_path or not _os.path.exists(image_path):
        raise ValueError(f"image2video requires --image-path pointing to an existing file, got: {image_path}")
    from PIL import Image as _Image

    img = _Image.open(image_path).convert("RGB")
    extra = {"num_frames": num_frames, "video_height": video_h, "video_width": video_w}
    return [
        {
            "prompt": render_lance_prompt("i2v", p, vision_token=_VISION_BLOCK),
            "multi_modal_data": {"first_frame": img},
            "modalities": ["video"],
            "extra_args": extra,
        }
        for p in prompts
    ]


def _format_img2img_prompts(prompts, image_path):
    import os as _os

    if not image_path or not _os.path.exists(image_path):
        raise ValueError(f"img2img requires --image-path pointing to an existing file, got: {image_path}")
    from PIL import Image as _Image

    img = _Image.open(image_path).convert("RGB")
    return [
        {
            "prompt": render_lance_prompt("image_edit", p, vision_token=_VISION_BLOCK),
            "multi_modal_data": {"img2img": img},
            "modalities": ["image"],
        }
        for p in prompts
    ]


def _format_video2video_prompts(prompts, video_path, num_frames, video_h, video_w):
    import os as _os

    if not video_path or not _os.path.exists(video_path):
        raise ValueError(f"video2video requires --video-path pointing to an existing file, got: {video_path}")

    extra = {"num_frames": num_frames, "video_height": video_h, "video_width": video_w}
    return [
        {
            "prompt": render_lance_prompt("video_edit", p, vision_token=_VISION_BLOCK),
            "multi_modal_data": {"video": video_path},
            "modalities": ["video"],
            "extra_args": extra,
        }
        for p in prompts
    ]


def _format_video2text_prompts(prompts, video_path, system_prompt=None):
    import os as _os

    if not video_path or not _os.path.exists(video_path):
        raise ValueError(f"video2text requires --video-path pointing to an existing file, got: {video_path}")
    import imageio.v3 as iio

    video = iio.imread(video_path)
    return [
        {
            "prompt": render_lance_prompt("x2t_video", p, vision_token=_VISION_BLOCK, system_prompt=system_prompt),
            "multi_modal_data": {"video": video},
            "modalities": ["text"],
        }
        for p in prompts
    ]


def _format_img2text_prompts(prompts, image_path, system_prompt=None):
    from PIL import Image

    if not image_path or not os.path.exists(image_path):
        raise ValueError(f"img2text requires --image-path pointing to an existing file, got: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return [
        {
            "prompt": render_lance_prompt("x2t_image", p, vision_token=_VISION_BLOCK, system_prompt=system_prompt),
            "multi_modal_data": {"image": img},
            "modalities": ["text"],
        }
        for p in prompts
    ]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Lance aligns more naturally with upstream when using FLASH_ATTN
    # (upstream uses flash_attn_varlen_func; SDPA accumulates ~5x more
    # numerical drift through the 36-layer Qwen2 stack on B300).  Default
    # to flash_attn when the user hasn't explicitly set a backend.
    os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")

    if args.txt_prompts:
        with open(args.txt_prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    elif args.modality == "img2text":
        prompts = args.prompts or ["Describe this image in detail."]
    else:
        prompts = args.prompts or ["A cute cat, studio lighting, 4k"]

    from vllm_omni.entrypoints.omni import Omni

    # Video modalities require the Lance_3B_Video checkpoint (3D
    # latent_pos_embed shaped (126976, 2048)).  Without it, vllm-omni silently
    # loads Lance_3B (4096, 2048) — the image-only table — and any t_lat >= 1
    # immediately indexes out of bounds.  Rewrite the model path so
    # LancePipeline._select_video_variant picks the video subfolder.
    resolved_model = args.model
    if args.modality in ("text2video", "video2video", "video2text", "image2video"):
        if not str(resolved_model).rstrip("/").endswith("Lance_3B_Video"):
            candidate = os.path.join(resolved_model, "Lance_3B_Video")
            if os.path.isdir(candidate):
                resolved_model = candidate

    omni_kwargs = vars(args).copy()
    omni_kwargs["model"] = resolved_model
    # Lance single-stage defaults (formerly in vllm_omni/deploy/lance.yaml):
    omni_kwargs.setdefault("pipeline", "lance")
    omni_kwargs.setdefault("max_num_batched_tokens", 32768)
    omni_kwargs.setdefault("max_num_seqs", 1)
    omni_kwargs.setdefault("enforce_eager", True)
    omni_kwargs.setdefault("trust_remote_code", True)
    omni_kwargs.setdefault("enable_prefix_caching", False)
    omni_kwargs.setdefault("async_chunk", False)
    omni = Omni(**omni_kwargs)

    if args.modality == "img2text":
        formatted = _format_img2text_prompts(prompts, args.image_path, system_prompt=args.system_prompt)
    elif args.modality == "text2video":
        formatted = _format_text2video_prompts(prompts, args.num_frames, args.video_height, args.video_width)
    elif args.modality == "video2text":
        formatted = _format_video2text_prompts(prompts, args.video_path, system_prompt=args.system_prompt)
    elif args.modality == "img2img":
        formatted = _format_img2img_prompts(prompts, args.image_path)
    elif args.modality == "video2video":
        formatted = _format_video2video_prompts(
            prompts, args.video_path, args.num_frames, args.video_height, args.video_width
        )
    elif args.modality == "image2video":
        formatted = _format_image2video_prompts(
            prompts, args.image_path, args.num_frames, args.video_height, args.video_width
        )
    else:
        formatted = _format_text2img_prompts(prompts)

    params_list = omni.default_sampling_params_list
    diffusion_params = params_list[0]  # single-stage: one param set
    diffusion_params.num_inference_steps = args.steps  # type: ignore
    if args.seed is not None:
        diffusion_params.seed = args.seed  # type: ignore
    if args.height is not None:
        diffusion_params.height = args.height  # type: ignore
    if args.width is not None:
        diffusion_params.width = args.width  # type: ignore
    extra = getattr(diffusion_params, "extra_args", {}) or {}
    extra["cfg_text_scale"] = args.cfg_text_scale
    extra["timestep_shift"] = args.timestep_shift
    if args.modality in ("img2text", "video2text"):
        extra["max_think_tokens"] = args.max_text_tokens
        extra["do_sample"] = args.do_sample
        extra["text_temperature"] = args.text_temperature
    if args.negative_prompt is not None:
        extra["negative_prompt"] = args.negative_prompt
    diffusion_params.extra_args = extra  # type: ignore

    outputs = list(omni.generate(prompts=formatted, sampling_params_list=params_list))

    if args.modality in ("img2text", "video2text"):
        for i, req_output in enumerate(outputs):
            multimodal_output = getattr(req_output, "multimodal_output", {}) or {}
            metadata = multimodal_output.get("metadata", {}) if isinstance(multimodal_output, dict) else {}
            text_metadata = metadata.get("text", {}) if isinstance(metadata, dict) else {}
            text = (multimodal_output.get("text") if isinstance(multimodal_output, dict) else None) or (
                text_metadata.get("text_output") if isinstance(text_metadata, dict) else None
            )
            text = text or getattr(req_output, "output", None) or getattr(req_output, "text", None)
            print(f"[Output {i}] {text}")
        return

    if args.modality in ("text2video", "video2video", "image2video"):
        for i, req_output in enumerate(outputs):
            frames = getattr(req_output, "images", None)
            if not frames:
                print(f"[Output {i}] no video frames returned")
                continue
            mp4_path = os.path.join(args.output, f"lance_{i}.mp4")
            try:
                import imageio.v3 as iio
                import numpy as np

                arr = np.stack([np.asarray(f) for f in frames], axis=0)
                iio.imwrite(mp4_path, arr, fps=args.fps, codec="libx264", quality=8)
                print(f"[Output {i}] Saved MP4 ({len(frames)} frames, {arr.shape[1]}x{arr.shape[2]}) to {mp4_path}")
            except Exception as exc:
                print(f"[Output {i}] MP4 encode failed ({exc!r}); writing per-frame ONGs instead")
                frame_dir = os.path.join(args.output, f"lance_{i}_frames")
                os.makedirs(frame_dir, exist_ok=True)
                for j, frame in enumerate(frames):
                    frame.save(os.path.join(frame_dir, f"{j:04d}.png"))
                print(f"  → {frame_dir} ({len(frames)} frames)")
        return

    idx = 0
    for req_output in outputs:
        images = getattr(req_output, "images", None)
        if not images:
            continue
        for j, img in enumerate(images):
            path = os.path.join(args.output, f"lance_{idx}_{j}.png")
            img.save(path)
            print(f"[Output] Saved image to {path}")
        idx += 1


if __name__ == "__main__":
    main()
