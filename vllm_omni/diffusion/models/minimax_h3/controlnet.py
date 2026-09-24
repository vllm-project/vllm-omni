# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original VideoX-Fun H3 Union control branch, adapted to Omni TP layers.

Reference: aigc-apps/VideoX-Fun at 968f0e2192ba4c7a12868bf36d73260d135424ca,
minimax_h3_transformer3d_control.py and pipeline_minimax_h3_control.py.
The released full-width AdaLN weights are required; ComfyUI repacks are not supported.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .minimax_h3_blocks import MiniMaxH3DiTArchConfig, MiniMaxH3DiTBlock
from .packed_tokens import minimax_h3_patchify_video_latent

CONTROL_TYPES = frozenset({"canny", "depth", "hed", "mlsd", "pose", "inpaint"})
CONTROL_BLOCKS = (0, 10, 20, 30, 40)


def parse_control(extra: dict[str, Any]) -> dict[str, Any] | None:
    modes = CONTROL_TYPES.intersection(extra)
    if not modes:
        return None
    if len(modes) != 1:
        raise ValueError("MiniMax H3 accepts exactly one control type per request")
    mode = next(iter(modes))
    value = extra[mode]
    if not isinstance(value, dict):
        raise ValueError(f"MiniMax H3 {mode} control settings must be an object")
    unknown = set(value) - {"control_path", "source_path", "mask_path", "control_context_scale"}
    if unknown:
        raise ValueError(f"Unsupported MiniMax H3 control settings: {sorted(unknown)}")
    value = dict(value)
    scale = value.get("control_context_scale", 1.0)
    try:
        valid_scale = (
            not isinstance(scale, bool) and isinstance(scale, (float, int)) and math.isfinite(scale) and scale >= 0
        )
    except OverflowError:
        valid_scale = False
    if not valid_scale:
        raise ValueError("control_context_scale must be finite and nonnegative")
    value["control_context_scale"] = float(scale)
    for role in ("control_path", "source_path", "mask_path"):
        if value.get(role) is not None and (not isinstance(value[role], str) or not value[role]):
            raise ValueError(f"{role} must be a nonempty local path")
    if value.get("source_path") and not value.get("mask_path"):
        raise ValueError("source_path requires mask_path")
    if not value.get("control_path") and not value.get("mask_path"):
        raise ValueError("MiniMax H3 control requires control_path or mask_path")
    if mode == "inpaint" and not value.get("mask_path"):
        raise ValueError("inpaint control requires mask_path")
    if mode != "inpaint" and not value.get("control_path"):
        raise ValueError(f"{mode} requires control_path")
    return value


def load_control_pixels(path: str, *, height: int, width: int, num_frames: int, mask: bool = False) -> torch.Tensor:
    """Sample at 24 FPS, retaining only resized CPU frames for the requested window."""
    import av
    import numpy as np
    from PIL import Image

    if min(height, width, num_frames) < 1:
        raise ValueError("Control canvas dimensions and frame count must be positive")

    def fit_frame(pixels: np.ndarray) -> torch.Tensor:
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        value = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
        if mask:
            value.gt_(0.5)
        if value.shape[-2:] != (height, width):
            value = F.interpolate(value[None], (height, width), mode="bilinear", align_corners=False)[0]
        if mask:
            value.gt_(0.5)
        return value

    if mask:
        try:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("L")).copy()
        except (OSError, ValueError):
            pass  # A temporal mask uses the same timestamp policy as the hint.
        else:
            return fit_frame(pixels)[None, :, None]
    frames = None
    frame_count = 0
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("Control media has no video stream")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 24)
        first_time = None
        previous = None
        previous_time = -1.0
        source_shape = None
        for index, frame in enumerate(container.decode(stream)):
            if index >= 3600:
                raise ValueError("Control media exceeds the bounded decode budget for the requested window")
            shape = (frame.height, frame.width)
            if source_shape is not None and shape != source_shape:
                raise ValueError("Control video frame dimensions must be consistent")
            source_shape = shape
            timestamp = float(frame.time) if frame.time is not None else index / rate
            if first_time is None:
                first_time = timestamp
            timestamp -= first_time
            if not math.isfinite(timestamp) or timestamp <= previous_time:
                timestamp = previous_time + 1 / rate
            previous_time = timestamp
            pixels = fit_frame(frame.to_ndarray(format="gray" if mask else "rgb24"))
            if previous is None:
                previous = pixels
            if frames is None:
                frames = torch.empty((1, pixels.shape[0], num_frames, height, width), dtype=torch.float32, device="cpu")
            # Container PTS is quantized: Matroska commonly represents 1/24 s
            # as 42 ms. Treat times within half a tick as the same boundary,
            # otherwise a 24 FPS source repeats frame 0 and drops frame 1.
            time_base = frame.time_base or stream.time_base
            tolerance = float(time_base) / 2 + 1e-9 if time_base is not None else 1e-9
            while frame_count < num_frames and frame_count / 24 < timestamp - tolerance:
                frames[0, :, frame_count].copy_(previous)
                frame_count += 1
            previous = pixels
            if frame_count >= num_frames:
                break
        if frames is None or previous is None:
            raise ValueError("Control media contains no decoded frames")
        while frame_count < num_frames:
            frames[0, :, frame_count].copy_(previous)
            frame_count += 1
    return frames


def fit_control_canvas(pixels: torch.Tensor, height: int, width: int, num_frames: int) -> torch.Tensor:
    """Reference geometry: truncate/hold final frame, then bilinear canvas resize."""
    if pixels.ndim != 5 or pixels.shape[0] != 1 or min(pixels.shape[2:]) < 1:
        raise ValueError("Control media must have shape [1, C, T, H, W] with nonempty dimensions")
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("Control pixels must be finite and in [0, 1]")
    pixels = pixels.float()
    if pixels.shape[2] < num_frames:
        pixels = torch.cat((pixels, pixels[:, :, -1:].expand(-1, -1, num_frames - pixels.shape[2], -1, -1)), dim=2)
    else:
        pixels = pixels[:, :, :num_frames]
    if pixels.shape[-2:] != (height, width):
        frames = F.interpolate(pixels[0].permute(1, 0, 2, 3), (height, width), mode="bilinear", align_corners=False)
        pixels = frames.permute(1, 0, 2, 3)[None]
    return pixels


def build_control_rows(
    control: torch.Tensor | None,
    source: torch.Tensor | None,
    mask: torch.Tensor | None,
    *,
    height: int,
    width: int,
    num_frames: int,
    latent_shape: tuple[int, int, int],
    encode: Any,
    device: torch.device,
) -> torch.Tensor:
    """Patchify each channel group separately, exactly as the training pipeline.

    encode returns normalized posterior-mode latents [1,24,T,H,W]. No mask
    means ZERO extra columns, rather than an encoded black video.
    """

    def fit(value: torch.Tensor, channels: int) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != channels:
            raise ValueError(f"Expected control media with {channels} channels")
        return fit_control_canvas(value, height, width, num_frames).to(device)

    def patch(latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape[2:]) != latent_shape:
            raise ValueError(f"Control latent grid {tuple(latent.shape[2:])} != output grid {latent_shape}")
        return minimax_h3_patchify_video_latent(latent, patch_size=(1, 2, 2)).float()

    if source is not None and mask is None:
        raise ValueError("Inpainting source requires mask")
    if control is None and mask is None:
        raise ValueError("Control video or mask is required")
    control_rows = (
        patch(encode(fit(control, 3)))
        if control is not None
        else patch(torch.zeros((1, 24, *latent_shape), device=device, dtype=torch.float32))
    )
    if mask is None:
        return F.pad(control_rows, (0, 25 * 4))
    mask = (fit((mask > 0.5).float(), 1) > 0.5).float()
    visible = 1 - mask
    pixels = fit(source, 3) * visible if source is not None else torch.zeros_like(visible.expand(-1, 3, -1, -1, -1))
    source_latents = encode(pixels)
    visibility = F.interpolate(visible, size=latent_shape, mode="trilinear", align_corners=False)
    return torch.cat((control_rows, patch(visibility), patch(source_latents)), dim=-1)


class MiniMaxH3ControlBlock(MiniMaxH3DiTBlock):
    def __init__(self, arch: MiniMaxH3DiTArchConfig, index: int):
        prefix = f"controlnet.control_blocks.{index}"
        super().__init__(arch, None, prefix=prefix)
        if index == 0:
            self.before_proj = ColumnParallelLinear(
                arch.hidden_size,
                arch.hidden_size,
                bias=True,
                gather_output=True,
                params_dtype=torch.bfloat16,
                prefix=f"{prefix}.before_proj",
            )
        self.after_proj = ColumnParallelLinear(
            arch.hidden_size,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.after_proj",
        )


class MiniMaxH3ControlNet(nn.Module):
    """Five reference blocks sharing the base packed attention and timestep plan."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig, places: tuple[int, ...] = CONTROL_BLOCKS):
        super().__init__()
        if not places or places[0] != 0 or places[-1] >= arch.num_layers:
            raise ValueError("Invalid H3 control injection blocks")
        self.arch = arch
        self.places = places
        self.control_proj_in = ColumnParallelLinear(
            49 * math.prod(arch.patch_size),
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=torch.float32,
            prefix="controlnet.control_proj_in",
        )
        self.control_blocks = nn.ModuleList([MiniMaxH3ControlBlock(arch, i) for i in range(len(places))])

    def forward(
        self,
        hidden: torch.Tensor,
        control_rows: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        **block_kwargs: Any,
    ) -> dict[int, torch.Tensor]:
        expected = (video_indices.numel(), 49 * math.prod(self.arch.patch_size))
        if tuple(control_rows.shape) != expected:
            raise ValueError(f"control_rows must be {expected}, got {tuple(control_rows.shape)}")
        embeds, _ = self.control_proj_in(control_rows.float())
        stream = hidden.index_copy(0, video_indices, embeds.to(hidden.dtype))
        stream, _ = self.control_blocks[0].before_proj(stream)
        stream = stream + hidden
        hints = {}
        for place, block in zip(self.places, self.control_blocks, strict=True):
            stream = block(stream, **block_kwargs)
            skip, _ = block.after_proj(stream)
            # Official control_apply_audio=False retains text skips too.
            hints[place] = skip.index_fill(0, audio_indices, 0)
        return hints

    def checkpoint_shapes(self) -> dict[str, tuple[int, ...]]:
        """Unsharded original shapes; validate before TP loaders slice tensors."""
        arch = self.arch
        hidden = arch.hidden_size
        inner = arch.num_attention_heads * arch.attention_head_dim
        ffn = arch.ffn_hidden_size
        shapes = {
            "control_proj_in.weight": (hidden, 49 * math.prod(arch.patch_size)),
            "control_proj_in.bias": (hidden,),
        }
        for i in range(len(self.places)):
            block_shapes = {
                "attn.to_q.weight": (inner, hidden),
                "attn.to_k.weight": (inner, hidden),
                "attn.to_v.weight": (inner, hidden),
                "attn.to_out.0.weight": (hidden, inner),
                "attn.norm_q.weight": (arch.attention_head_dim,),
                "attn.norm_k.weight": (arch.attention_head_dim,),
                "norm1.weight": (hidden,),
                "norm2.weight": (hidden,),
                "ff.net.0.proj.weight": (2 * ffn, hidden),
                "ff.net.2.weight": (hidden, ffn),
                "adaln_proj.linear.weight": (18 * hidden, arch.time_embed_dim),
                "adaln_proj.linear.bias": (18 * hidden,),
                "after_proj.weight": (hidden, hidden),
                "after_proj.bias": (hidden,),
            }
            if i == 0:
                block_shapes.update({"before_proj.weight": (hidden, hidden), "before_proj.bias": (hidden,)})
            shapes.update({f"control_blocks.{i}.{name}": shape for name, shape in block_shapes.items()})
        return shapes

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        seen: set[str] = set()
        shapes = self.checkpoint_shapes()
        expected = set(shapes)
        for original, tensor in weights:
            if original not in expected or original in seen:
                raise ValueError(f"Unexpected or duplicate original H3 control weight: {original}")
            if tuple(tensor.shape) != shapes[original]:
                raise ValueError(
                    f"Original H3 control weight {original} has shape {tuple(tensor.shape)}, "
                    f"expected {shapes[original]}"
                )
            seen.add(original)
            name = original.replace(".attn.norm_q.", ".attn.q_norm.").replace(".attn.norm_k.", ".attn.k_norm.")
            name = name.replace(".attn.to_out.0.", ".attn.out_proj.").replace(".ff.net.0.proj.", ".mlp.fc1.")
            name = name.replace(".ff.net.2.", ".mlp.fc2.")
            shard = None
            for part in "qkv":
                if f".attn.to_{part}." in name:
                    name = name.replace(f".attn.to_{part}.", ".attn.qkv_proj.")
                    shard = part
                    break
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is not None:
                loader(param, tensor, shard)
            elif ".mlp.fc1." in name:
                # Diffusers SwiGLU stores [up, gate]; Omni stores [gate, up].
                up, gate = tensor.chunk(2, dim=0)
                loader(param, gate, 0)
                loader(param, up, 1)
            else:
                loader(param, tensor)
            loaded.add(name)
        missing = expected - seen
        if missing:
            raise ValueError(f"Missing original H3 control weights: {sorted(missing)}")
        return loaded
