# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pika H3 Flex release contract and request-owned inference noise.

Release: https://huggingface.co/Pika-Labs/Pika-H3-Flex
Sampling/noise convention: Pika-Labs/Pika-H3-Flex at
06bd434ce0b6e287e6f6c4058420460f47ba031b (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import torch

from .packed_tokens import minimax_h3_patchify_video_latent

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PIKA_BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PIKA_KNOTS = (1000, 750, 500, 250, 100, 40, 0)
PIKA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


@dataclass(frozen=True)
class PikaSpec:
    task: str
    sha256: str

    name: ClassVar[str] = "Pika H3 Flex"
    denoise_steps: ClassVar[int] = 6

    @property
    def conditioning_dtype(self) -> torch.dtype:
        return torch.float32 if self.task == "ref2va" else torch.bfloat16

    @property
    def base_schedule(self) -> tuple[float, ...]:
        return tuple(knot / 1000 for knot in PIKA_KNOTS)

    def check_request(self, sampling: OmniDiffusionSamplingParams, task: str, *, step_execution: bool = False) -> None:
        supported = {"ref2va"} if self.task == "ref2va" else {"t2va", "fl2va"}
        if task not in supported:
            raise ValueError(f"Pika {self.task} adapter cannot serve task={task}")
        if float(sampling.lora_scale) != 1.0:
            raise ValueError("Pika H3 Flex requires lora_scale=1 (or 0 to disable the adapter)")
        steps = sampling.num_inference_steps
        if steps is not None and steps != 6:
            raise ValueError("Pika H3 Flex requires num_inference_steps=6, counting transformer forwards")
        if step_execution and steps is None:
            raise ValueError("Pika H3 Flex step execution requires explicit num_inference_steps=6")
        extra = sampling.extra_args or {}
        for key, value in (("flow_shift", 12.0), ("audio_flow_shift", 3.0)):
            if float(extra.get(key, value)) != value:
                raise ValueError(f"Pika H3 Flex requires {key}={value:g}")
        if extra.get("long_video_mode", "full") != "full" or extra.get("continuation_prompts") is not None:
            raise ValueError("Pika H3 Flex supports full clips; continuation is not supported")
        if extra.get("audio_mode", "native") != "native":
            raise ValueError("Pika H3 Flex requires audio_mode=native")
        if extra.get("sampler", "euler") != "euler":
            raise ValueError("Pika H3 Flex selects its own re-noising sampler; remove the sampler override")
        if extra.get("continue_from") is not None:
            raise ValueError("Pika H3 Flex does not support continue_from")


def read_pika_spec(path: str | Path) -> tuple[Path, PikaSpec] | None:
    """Identify by a sibling manifest, never by a filename containing 'pika'."""
    path = Path(path)
    root = path if path.is_dir() else path.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if path.is_dir() and any((path / task / "manifest.json").is_file() for task in ("fl2va", "ref2va")):
            raise ValueError("Select the Pika fl2va/ or ref2va/ adapter directory, not the repository root")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != "pika-flex":
        return None
    task = manifest.get("task")
    if task not in {"fl2va", "ref2va"}:
        raise ValueError(f"Unsupported Pika task: {task!r}")
    expected = {
        "format_version": 1,
        "release_id": f"pika-flex-{task}-v1",
        "rank": 128,
        "alpha": 128,
        "trunk": "transformer_ref" if task == "ref2va" else "transformer",
        "numerics_policy": "fp32_conditioning_v1" if task == "ref2va" else "bf16_conditioning_v1",
        "base": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": PIKA_BASE_REVISION},
        "sampling": {"knots": list(PIKA_KNOTS), "video_shift": 12.0, "audio_shift": 3.0, "method": "renoise"},
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unsupported Pika manifest {key}: expected {value!r}")
    if manifest.get("target_modules") != list(PIKA_TARGET_MODULES):
        raise ValueError("Unsupported Pika target_modules")
    digest = manifest.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Pika manifest must include a SHA-256 digest")
    weights = root / "adapter.safetensors"
    if not path.is_dir() and path != weights:
        raise ValueError("Pika manifest applies only to its sibling adapter.safetensors")
    return weights, PikaSpec(task=task, sha256=digest)


def verify_pika_weights(path: Path, spec: PikaSpec) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ValueError(f"Pika adapter checksum mismatch: {path}")


class PikaNoise:
    """Independent initial, re-noising, and augmentation streams per request.

    Draw in Pika's external [B,T,C,H,W] video and [B,T,64] stereo audio
    layouts before packing. Drawing directly in token order changes the seed.
    Every sequence-parallel rank uses the same seed and full latent geometry.
    """

    def __init__(
        self, *, seed: int, device: torch.device, latent_t: int, latent_h: int, latent_w: int, audio_t: int
    ) -> None:
        self.device = device
        self.video_shape = (1, latent_t, 24, latent_h, latent_w)
        self.audio_shape = (1, audio_t, 64)
        self.initial = torch.Generator(device=device).manual_seed(seed)
        self.renoise = torch.Generator(device=device).manual_seed(seed + 1)
        self.augmentation = torch.Generator(device=device).manual_seed(seed + 7919)

    def video(self, generator: torch.Generator, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        noise = torch.randn(shape or self.video_shape, generator=generator, device=self.device, dtype=torch.float32)
        return minimax_h3_patchify_video_latent(noise.permute(0, 2, 1, 3, 4), patch_size=(1, 2, 2))

    def initial_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video(self.initial), self.audio(self.initial)

    def audio(self, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(self.audio_shape, generator=generator, device=self.device, dtype=torch.float32)
        return noise.reshape(self.audio_shape[1], 2, 32).permute(1, 0, 2).reshape(-1, 32).contiguous()

    def visual_anchor(self, rows: torch.Tensor, shapes: list[tuple[int, int, int]], dtype: torch.dtype) -> torch.Tensor:
        rows = rows.to(device=self.device, dtype=dtype).float()
        output = []
        offset = 0
        for t, h, w in shapes:
            noise = self.video(self.augmentation, (1, t, 24, h, w))
            count = noise.shape[0]
            # Match the reference's Python scalar coefficients, including 1-t.
            output.append(0.999 * rows[offset : offset + count] + (1.0 - 0.999) * noise)
            offset += count
        if offset != rows.shape[0]:
            raise ValueError("Pika visual condition rows do not match their latent shapes")
        return torch.cat(output)

    def step(self, denoised: torch.Tensor, sigma_next: float, *, modality: str) -> torch.Tensor:
        if not math.isfinite(sigma_next) or not 0 <= sigma_next <= 1:
            raise ValueError("Pika sigma must be finite and in [0, 1]")
        if sigma_next == 0:
            return denoised  # No generator evaluation or RNG draw at terminal zero.
        if modality == "video":
            noise = self.video(self.renoise)
        elif modality == "audio":
            noise = self.audio(self.renoise)
        else:
            raise ValueError(f"Unknown Pika modality: {modality}")
        if noise.shape != denoised.shape:
            raise ValueError("Pika re-noising shape does not match the request")
        sigma = denoised.new_tensor(sigma_next, dtype=torch.float32)
        return ((1 - sigma) * denoised.float() + sigma * noise).to(denoised.dtype)


def pika_sigmas(shift: float) -> list[float]:
    base = torch.tensor(PIKA_KNOTS, dtype=torch.float64) / 1000
    return (shift * base / (1 + (shift - 1) * base)).float().tolist()
