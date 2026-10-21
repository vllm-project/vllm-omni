# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prepare a finite LingBot session; the browser panel owns its live connection."""

from .lingbot_geometry import DEFAULT_GEOMETRY
from .lingbot_request import (
    DEFAULT_MODEL,
    check_start_payload,
    encode_first_frame,
    validate_endpoint,
    validate_size,
)

MODEL = DEFAULT_MODEL


class VLLMOmniLingBotWorld:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "server_url": ("STRING", {"default": "ws://127.0.0.1:8000/v1/realtime/video"}),
                "model": ("STRING", {"default": MODEL}),
                "prompt": ("STRING", {"default": "A quiet forest path in daylight.", "multiline": True}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 129, "min": 9, "max": 4089, "step": 12}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1, "control_after_generate": False}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "prepare"
    CATEGORY = "vLLM-Omni"
    DESCRIPTION = "Run to prepare the scene, then open the interaction panel and click Start."

    def prepare(self, image, server_url, model, prompt, width, height, num_frames, fps, seed):
        url = validate_endpoint(server_url)
        if not prompt.strip() or not model.strip():
            raise ValueError("Model and prompt are required.")
        validate_size(width, height)
        if fps <= 0:
            raise ValueError("LingBot requires positive FPS.")
        DEFAULT_GEOMETRY.chunks_for(num_frames)
        payload = {
            "type": "session.start",
            "model": model.strip(),
            "prompt": prompt.strip(),
            "image_reference": {"image_url": encode_first_frame(image, width, height)},
            "format": "m4s",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "num_inference_steps": 4,
            "extra_params": {"flow_shift": 5.0},
        }
        check_start_payload(payload)
        return {
            "ui": {"omni_lingbot": [{"url": url, "payload": payload, "keyboard": True}]},
            "result": (),
        }
