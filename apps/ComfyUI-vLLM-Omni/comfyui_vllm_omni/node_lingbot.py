# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prepare a finite LingBot session; the browser panel owns its live connection."""

import json
from urllib.parse import urlsplit

import torch

from .utils.api_client import VLLMOmniClient
from .utils.format import image_tensor_to_base64

MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
MODES = ["keyboard (camera API required)", "stationary", "forward"]


class VLLMOmniLingBotWorld:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "server_url": ("STRING", {"default": "ws://127.0.0.1:8000/v1/realtime/video"}),
                "model": ("STRING", {"default": MODEL}),
                "prompt": ("STRING", {"default": "A quiet forest path in daylight.", "multiline": True}),
                "mode": (MODES, {"default": MODES[0]}),
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

    def prepare(self, image, server_url, model, prompt, mode, width, height, num_frames, fps, seed, camera_json=None):
        url = urlsplit(server_url.strip())
        if url.scheme not in ("ws", "wss") or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError("Use a ws:// or wss:// video endpoint without embedded credentials or a fragment.")
        if not prompt.strip() or not model.strip():
            raise ValueError("Model and prompt are required.")
        if mode not in MODES:
            raise ValueError("Unknown camera mode.")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("Width and height must be positive multiples of 16 matching the Omni deploy config.")
        if num_frames < 9 or (num_frames - 9) % 12 or fps <= 0:
            raise ValueError("LingBot requires num_frames = 9 + 12k and positive FPS.")
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError("Connect exactly one RGB image; select one image from a batch first.")
        image = image.detach().cpu()
        if not torch.isfinite(image).all():
            raise ValueError("The input image contains non-finite values.")
        image = image.clamp(0, 1)
        payload = {
            "type": "session.start",
            "model": model.strip(),
            "prompt": prompt.strip(),
            "image_reference": {"image_url": image_tensor_to_base64(image, size=(width, height))},
            "format": "m4s",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "num_inference_steps": 4,
            "extra_params": {"flow_shift": 5.0},
        }
        if camera_json is not None:
            payload["extra_params"]["camera_action_script"] = _parse_camera_json(camera_json, num_frames)
        elif mode != MODES[0]:
            actions = ["w"] if mode == "forward" else []
            payload["extra_params"]["camera_action_script"] = [
                [list(actions) for _ in range(3)] for _ in range((num_frames - 9) // 12 + 1)
            ]
        if len(json.dumps(payload)) > 4 * 1024 * 1024:
            raise ValueError("The start message exceeds Omni's 4 MiB limit; reduce the image dimensions or prompt.")
        return {
            "ui": {
                "omni_lingbot": [
                    {
                        "url": server_url.strip(),
                        "payload": payload,
                        "keyboard": mode == MODES[0],
                    }
                ]
            },
            "result": (),
        }


def _parse_camera_json(camera_json: str, num_frames: int) -> list:
    """Validate the server's per-chunk action format without importing model code."""
    if not isinstance(camera_json, str):
        raise ValueError("Camera JSON must be a string containing a JSON array.")
    if len(camera_json.encode("utf-8")) > 64 * 1024:
        raise ValueError("Camera JSON must not exceed 64 KiB.")
    try:
        script = json.loads(camera_json)
    except (ValueError, TypeError) as exc:
        raise ValueError("Camera JSON must be a JSON array of chunks.") from exc
    chunks = (num_frames - 9) // 12 + 1
    if not isinstance(script, list) or len(script) != chunks:
        raise ValueError(f"Camera JSON requires exactly {chunks} chunks for {num_frames} frames.")
    for chunk in script:
        if not isinstance(chunk, list) or len(chunk) != 3:
            raise ValueError("Each camera chunk requires exactly three action lists.")
        for actions in chunk:
            if not isinstance(actions, list) or any(
                not isinstance(key, str) or key.lower() not in ("w", "a", "s", "d", "i", "j", "k", "l")
                for key in actions
            ):
                raise ValueError("Each action list supports only W/A/S/D/I/J/K/L keys.")
    return script


class VLLMOmniLingBotRealtimeJSON(VLLMOmniLingBotWorld):
    DESCRIPTION = "Prepare a fixed JSON camera trajectory, then open the panel and click Start."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].pop("mode")
        inputs["required"]["num_frames"][1]["default"] = 33
        inputs["required"]["camera_json"] = (
            "STRING",
            {"default": '[[["w"],["w"],["w"]],[["a"],[],[]],[[],[],[]]]', "multiline": True},
        )
        return inputs

    def prepare(self, image, server_url, model, prompt, width, height, num_frames, fps, seed, camera_json):
        return super().prepare(
            image,
            server_url,
            model,
            prompt,
            "stationary",
            width,
            height,
            num_frames,
            fps,
            seed,
            camera_json=camera_json,
        )


class VLLMOmniLingBotOfflineJSON(VLLMOmniLingBotRealtimeJSON):
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = False
    FUNCTION = "generate"
    DESCRIPTION = "Run a JSON camera trajectory on Omni and return the complete video for Save Video."

    async def generate(self, image, server_url, model, prompt, width, height, num_frames, fps, seed, camera_json):
        config = self.prepare(image, server_url, model, prompt, width, height, num_frames, fps, seed, camera_json)[
            "ui"
        ]["omni_lingbot"][0]
        video = await VLLMOmniClient(config["url"]).generate_video_stream(config["payload"])
        return (video,)
