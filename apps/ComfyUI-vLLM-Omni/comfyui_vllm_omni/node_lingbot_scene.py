# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LingBot World shots, camera moves and offline rendering.

A shot is a prompt held for a whole number of chunks. A sequence chains shots
on one port, and the renderer turns the chain into a single session: the first
shot's prompt starts it, every later shot's prompt goes out as an interaction
just before its chunk, and the camera moves compile into the per-chunk
``camera_action_script`` the server reads. Nothing here talks to a GPU to build
a payload, so the whole surface is testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lingbot_geometry import DEFAULT_GEOMETRY
from .lingbot_moves import LingBotSequence, Shot, compile_sequence, parse_moves, sequence_payload
from .lingbot_request import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    check_start_payload,
    encode_first_frame,
    validate_endpoint,
    validate_size,
)
from .utils.api_client import VLLMOmniClient

DEFAULT_MOVES = """\
# One move per line: lane value start end. Frames count from the shot's start;
# 'end' means the shot's last frame. Lanes: move (forward/back), strafe
# (left/right), look_v (up/down), look_h (left/right).
move    forward  0    end
look_h  right    24   end
"""


@dataclass(frozen=True)
class LingBotMoves:
    """A moves text on the ``LINGBOT_MOVES`` port.

    The text is parsed by the shot that receives it, because a move's ``end``
    and its bounds are relative to that shot's length.
    """

    text: str


class VLLMOmniLingBotMoves:
    """Author camera moves once and connect them to one or more shots."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moves": ("STRING", {"default": DEFAULT_MOVES, "multiline": True}),
            }
        }

    RETURN_TYPES = ("LINGBOT_MOVES",)
    RETURN_NAMES = ("moves",)
    FUNCTION = "build"
    CATEGORY = "vLLM-Omni"
    DESCRIPTION = "Camera moves for one shot, as one line per move."

    def build(self, moves):
        text = moves.strip()
        if not text:
            raise ValueError("Add at least one move, or leave the moves input unconnected.")
        return (LingBotMoves(text),)


class VLLMOmniLingBotShot:
    """One prompt held for a number of chunks, with optional camera moves."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A quiet forest path in daylight.", "multiline": True}),
                "chunks": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "One chunk is 12 frames after the first; 0.75 s at 16 FPS.",
                    },
                ),
            },
            "optional": {
                "sequence": ("LINGBOT_SEQUENCE",),
                "moves": ("LINGBOT_MOVES",),
            },
        }

    RETURN_TYPES = ("LINGBOT_SEQUENCE",)
    RETURN_NAMES = ("sequence",)
    FUNCTION = "add"
    CATEGORY = "vLLM-Omni"
    DESCRIPTION = "Append a shot to a LingBot sequence. Feed the last shot to Render and to the Visualizer."

    def add(self, prompt, chunks, sequence=None, moves=None):
        text = prompt.strip()
        if not text:
            raise ValueError("A shot needs a prompt.")
        shots = sequence.shots if sequence is not None else ()
        index = len(shots)
        frames = DEFAULT_GEOMETRY.shot_frames(index, int(chunks))
        move_list = parse_moves(moves.text, shot_frames=frames) if moves is not None else ()
        return (LingBotSequence((*shots, Shot(prompt=text, frames=frames, moves=move_list))),)


class VLLMOmniLingBotRender:
    """Render a whole sequence into one video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sequence": ("LINGBOT_SEQUENCE",),
                "server_url": ("STRING", {"default": DEFAULT_ENDPOINT}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1, "control_after_generate": False}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "vLLM-Omni"
    DESCRIPTION = "Run every shot in one LingBot session and return the complete video."

    async def generate(self, image, sequence, server_url, model, width, height, fps, seed):
        url = validate_endpoint(server_url)
        validate_size(width, height)
        if not model.strip():
            raise ValueError("Model is required.")
        if fps <= 0:
            raise ValueError("FPS must be positive.")
        compiled = compile_sequence(sequence, DEFAULT_GEOMETRY)
        payload = {
            "type": "session.start",
            "model": model.strip(),
            "prompt": sequence.shots[0].prompt,
            "image_reference": {"image_url": encode_first_frame(image, width, height)},
            "format": "m4s",
            "width": width,
            "height": height,
            "num_frames": compiled.total_frames,
            "fps": fps,
            "seed": seed,
            "num_inference_steps": 4,
            "extra_params": {"flow_shift": 5.0, "camera_action_script": compiled.camera_script()},
        }
        check_start_payload(payload)
        video = await VLLMOmniClient(url).generate_video_stream(payload, dict(compiled.prompt_switches))
        return (video,)


class VLLMOmniLingBotVisualizer:
    """Draw a sequence's shots and camera moves on a frame ruler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("LINGBOT_SEQUENCE",),
                "fps": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 60,
                        "tooltip": "Only sets the ruler's second marks; keep it equal to the Render node's FPS.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show"
    CATEGORY = "vLLM-Omni"
    DESCRIPTION = ("Show the shots, moves and chunk grid. It renders nothing; its FPS only labels the ruler.",)

    def show(self, sequence, fps):
        if fps <= 0:
            raise ValueError("FPS must be positive.")
        compiled = compile_sequence(sequence, DEFAULT_GEOMETRY)
        payload = sequence_payload(sequence, compiled, DEFAULT_GEOMETRY, fps=fps)
        return {"ui": {"omni_lingbot_sequence": [payload]}, "result": ()}
