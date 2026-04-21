import argparse

import imageio
import numpy as np
import torch
from PIL import Image

# Use official top-level API
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main(args):
    print(f"🚀 Loading OmniWeaving model from {args.model} (TP={args.tensor_parallel_size})...")

    # 1. Initialize engine
    omni = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    # 2. Prepare multimodal inputs
    multi_modal_data = {}
    if args.image_path:
        multi_modal_data["image"] = Image.open(args.image_path).convert("RGB")
        print(f"🖼️ Loaded input image from {args.image_path}")

    # 3. Set sampling parameters
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
    )

    # 4. Execute inference
    # [FIXED] Proper vLLM multimodal dictionary structure
    if multi_modal_data:
        inputs = [{"prompt": args.prompt, "multi_modal_data": multi_modal_data}]
    else:
        inputs = [args.prompt]

    print(f"📡 Dispatching request with {args.num_frames} frames to orchestrator...")

    # [FIXED] Use the exact parameter name found in the source code: 'sampling_params_list'
    outputs = omni.generate(
        prompts=inputs,
        sampling_params_list=sampling_params,
    )

    # 5. Post-process and save
    if not outputs:
        raise RuntimeError("❌ Omni.generate() returned an empty list!")

    req_output = outputs[0]

    # Safely unbox the vLLM object
    if hasattr(req_output, "outputs") and isinstance(req_output.outputs, list) and len(req_output.outputs) > 0:
        raw_data = req_output.outputs[0]
    else:
        raw_data = req_output

    # Find the actual tensor payload
    if hasattr(raw_data, "data"):
        tensor_obj = raw_data.data
    elif hasattr(raw_data, "video"):
        tensor_obj = raw_data.video
    elif isinstance(raw_data, dict) and "video" in raw_data:
        tensor_obj = raw_data["video"]
    else:
        tensor_obj = raw_data  # Fallback: the object itself is the tensor

    # Convert to standard Numpy array safely
    if hasattr(tensor_obj, "cpu"):
        video_tensor = tensor_obj.detach().cpu().float().numpy()
    else:
        video_tensor = np.array(tensor_obj)

    print(f"🔍 Extracted Raw Tensor Shape: {video_tensor.shape}, Dimensions: {video_tensor.ndim}D")

    # Handle NaNs or Infs
    if not np.isfinite(video_tensor).all():
        video_tensor = np.nan_to_num(video_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # Remove the Batch dimension if it exists (e.g., [1, C, F, H, W] -> [C, F, H, W])
    if video_tensor.ndim == 5:
        video_tensor = np.squeeze(video_tensor, axis=0)

    if video_tensor.ndim != 4:
        raise ValueError(f"❌ Expected 4D tensor after squeeze, got {video_tensor.ndim}D: {video_tensor.shape}")

    # Determine channel position and transpose for ImageIO ([F, H, W, C])
    if video_tensor.shape[0] == 3:
        video_np = np.transpose(video_tensor, (1, 2, 3, 0))  # [C, F, H, W] -> [F, H, W, C]
    elif video_tensor.shape[1] == 3:
        video_np = np.transpose(video_tensor, (0, 2, 3, 1))  # [F, C, H, W] -> [F, H, W, C]
    else:
        video_np = np.transpose(video_tensor, (1, 2, 3, 0))  # Blind fallback

    # Normalize to [0, 1] then scale to [0, 255]
    v_min, v_max = video_np.min(), video_np.max()
    if v_max > v_min:
        video_np = (video_np - v_min) / (v_max - v_min)
    else:
        video_np = np.zeros_like(video_np)

    video_uint8 = (video_np * 255).clip(0, 255).astype(np.uint8)

    # Save to MP4 file
    output_filename = args.output
    imageio.mimsave(output_filename, video_uint8, fps=8)
    print(f"🎉 SUCCESS: Video successfully saved to {output_filename}!")

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"📊 Peak VRAM usage: {peak_memory_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline inference example for OmniWeaving.")
    parser.add_argument("--model", type=str, required=True, help="Path to the OmniWeaving model.")
    parser.add_argument(
        "--prompt", type=str, default="A cute bear wearing a Christmas hat moving naturally.", help="Text prompt."
    )
    parser.add_argument("--image-path", type=str, default=None, help="Path to the input reference image (optional).")
    parser.add_argument("--output", type=str, default="omniweaving_output.mp4", help="Output file name.")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="CFG guidance scale.")
    parser.add_argument("--height", type=int, default=256, help="Video height.")
    parser.add_argument("--width", type=int, default=256, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=9, help="Number of frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    main(args)
