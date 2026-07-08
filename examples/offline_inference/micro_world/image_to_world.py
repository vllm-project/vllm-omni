# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AMD Micro-World Image-to-World (I2W) example.

Generates an action-controlled video from an input image with keyboard/mouse
inputs. Uses the 14B AdaLN variant; cross-attention has an image branch
(``add_k_proj``/``add_v_proj``) fed by a CLIP-encoded image embedding.

Setup
─────
    uv pip install -e .

Usage
─────
    # Reproduce the AMD reference example (street_night.jpg + walking-down).
    # By default, --model resolves AMD/Micro-World and composes the local load
    # directory with the Wan2.1 I2V base.
    python image_to_world.py \
        --image street_night.jpg \
        --output micro_world_i2w_output.mp4

    # Custom prompt + action list:
    python image_to_world.py \
        --image my_scene.jpg \
        --prompt "First-person walk through a forest path." \
        --action-list '[[40, "1 0 0 0 0 0 0 0 0"], [80, "0 0 1 0 0 0 0 0 0"], "40"]'


    # Overlay the WASD/cursor HUD (post-processing, same as T2W):
    python image_to_world.py --image my_scene.jpg --draw-hud
"""

import argparse
import json

# Reuse the action presets and parse_reference_action_list from text_to_world.
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

sys.path.insert(0, str(Path(__file__).parent))
from _model_resolver import MICRO_WORLD_REPO, resolve_micro_world_model  # noqa: E402
from text_to_world import build_actions, parse_reference_action_list  # noqa: E402

_DEFAULT_PROMPT = (
    "First-person perspective walking down a lively city street at night. "
    "Neon signs and bright billboards glow on both sides, cars drive past with "
    "headlights and taillights streaking slightly. Camera motion directly aligned "
    "with user actions, immersive urban night scene."
)
# Reference's action_list for I2W: (back+strafe-right) → (forward) → (left).
_DEFAULT_ACTION_LIST = '[[20, "0 1 0 1 0 0 0 0 0"], [40, "1 0 0 0 0 0 0 0 0"], [80, "0 0 1 0 0 0 0 0 0"], "40"]'


def parse_args():
    parser = argparse.ArgumentParser(description="AMD Micro-World I2W generation.")
    parser.add_argument(
        "--model",
        default=MICRO_WORLD_REPO,
        help="Consolidated Micro-World repo id. Defaults to AMD/Micro-World.",
    )
    parser.add_argument("--image", required=True, help="Input image path (jpg/png).")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--actions", default="walk_forward", help="Action preset (used if no --action-list/file).")
    parser.add_argument("--action-file", default=None, help="JSON file with {mouse_actions, keyboard_actions}.")
    parser.add_argument(
        "--action-list",
        default=_DEFAULT_ACTION_LIST,
        help="Micro-World reference action_list as JSON string.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=str, default="micro_world_i2w_output.mp4")
    parser.add_argument("--draw-hud", action="store_true", help="Overlay WASD + mouse cursor HUD on each frame.")
    parser.add_argument("--stage-init-timeout", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.action_file:
        with open(args.action_file) as f:
            action_data = json.load(f)
        keyboard_actions = action_data["keyboard_actions"]
        mouse_actions = action_data["mouse_actions"]
    elif args.action_list:
        keyboard_actions, mouse_actions = parse_reference_action_list(json.loads(args.action_list))
    else:
        keyboard_actions, mouse_actions = build_actions(args.actions, args.num_frames)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    stage_config = str(
        Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_i2w.yaml"
    )
    omni = Omni(
        model=resolve_micro_world_model(args.model, "I2W"),
        stage_configs_path=stage_config,
        flow_shift=args.flow_shift,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.stage_init_timeout,
    )

    image = Image.open(args.image).convert("RGB")
    prompt_dict = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": image},
    }
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    print(
        f"Generating {args.num_frames} frames at {args.width}x{args.height} "
        f"from image={args.image} with actions={args.actions}"
    )
    t0 = time.perf_counter()
    outputs = omni.generate(
        prompt_dict,
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            generator=generator,
            extra_args={"mouse_actions": mouse_actions, "keyboard_actions": keyboard_actions},
        ),
    )
    print(f"Generation time: {time.perf_counter() - t0:.2f}s")

    first = outputs[0] if isinstance(outputs, list) else outputs
    if hasattr(first, "is_pipeline_output") and first.is_pipeline_output:
        first = first.request_output
    frames = first.images[0] if hasattr(first, "images") and first.images else None
    if frames is None:
        raise ValueError("No video frames in output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0].permute(1, 2, 3, 0) if video.shape[1] in (3, 4) else video[0]
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        video = video.float().numpy()
    elif isinstance(frames, np.ndarray):
        video = frames[0] if frames.ndim == 5 else frames
    else:
        video = frames

    if args.draw_hud:
        from _hud import draw_hud

        video = draw_hud(np.asarray(video), mouse_actions, keyboard_actions)

    export_to_video(video, str(output_path), fps=args.fps)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
