# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MODEL_ID = "tencent/Hy-Nomas"
DIT_FILE = "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
DECODER_FILE = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
REQUIRED_FILES = (DIT_FILE, DECODER_FILE)
TEXT_ENCODER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass(frozen=True)
class HyNomasARConfig:
    height: int = 704
    width: int = 1280
    chunk_latent_frames: int = 16
    max_latent_frames: int = 120
    max_segments: int = 16
    session_capacity: int = 1

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> HyNomasARConfig:
        values = {name: args.get(f"ar_{name}", info.default) for name, info in cls.__dataclass_fields__.items()}
        for name, value in values.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"Hy-Nomas ar_{name} must be a positive integer.")
        if values["height"] % 32 or values["width"] % 32:
            raise ValueError("Hy-Nomas ar_height and ar_width must be multiples of 32.")
        if values["chunk_latent_frames"] > values["max_latent_frames"]:
            raise ValueError("Hy-Nomas ar_chunk_latent_frames must not exceed ar_max_latent_frames.")
        return cls(**values)
