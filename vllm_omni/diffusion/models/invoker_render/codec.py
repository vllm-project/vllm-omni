# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

from .assets import PACKET_FRAMES, VIDEO_SIZE


def encode_rgb(vae: MiniMaxH3VideoVAE, pixels: np.ndarray, *, proxy_resize: str | None, eos: bool) -> torch.Tensor:
    model = vae.model
    device = next(model.parameters()).device
    if proxy_resize == "lanczos" and pixels.shape[1:3] != VIDEO_SIZE[::-1]:
        pixels = np.stack(
            [np.asarray(Image.fromarray(frame).resize(VIDEO_SIZE, Image.Resampling.LANCZOS)) for frame in pixels]
        )
    values = torch.as_tensor(pixels, device=device).permute(0, 3, 1, 2).float().div_(255)
    if proxy_resize == "nearest" and values.shape[-2:] != VIDEO_SIZE[::-1]:
        values = F.interpolate(values, size=VIDEO_SIZE[::-1], mode="nearest")
    values = model.processor.transform_tensor(values).transpose(0, 1)[None].to(torch.bfloat16)
    if eos:
        values = torch.cat((values, values[:, :, -1:].expand(-1, -1, PACKET_FRAMES - values.shape[2], -1, -1)), dim=2)
    with torch.autocast(device_type=device.type, enabled=False):
        mean = model._adaptive_encode(values).chunk(2, dim=1)[0][0].float()
    if mean.shape != (24, 1 if proxy_resize is None else 5, VIDEO_SIZE[1] // 16, VIDEO_SIZE[0] // 16):
        raise RuntimeError(f"Unexpected Invoker-Render encoder output: {tuple(mean.shape)}")
    center = mean.new_tensor(vae.config_dict["latents_mean"]).view(-1, 1, 1, 1)
    scale = mean.new_tensor(vae.config_dict["latents_std"]).view(-1, 1, 1, 1)
    return ((mean - center) / scale)[:, : 2 if eos else mean.shape[1]].to(torch.bfloat16).contiguous()


@dataclass
class DecodeState:
    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]
    position: int = 0
    pending: torch.Tensor | None = None
    finished: bool = False


class CausalVideoDecoder(nn.Module):
    # Reuse the window and checkpoint layout of CodeWorldModel's Apache-2.0 causal ViT decoder.
    def __init__(self, config_file: Path, weights_file: Path, vae: MiniMaxH3VideoVAE, device: torch.device):
        super().__init__()
        with config_file.open() as source:
            cfg = json.load(source)
        options = cfg["decoder_kwargs"]
        if (
            options.get("patch_size", 16),
            options.get("patch_size_t", 4),
            options["in_channels"],
            options["out_channels"],
            cfg["embed_dim"],
        ) != (16, 4, 24, 3, 24):
            raise ValueError("Invoker-Render decoder requires 24 latent channels and 16x16x4 RGB patches")
        package = type(vae.model).__module__.rpartition(".")[0]
        vit = import_module(f"{package}.vae_vit")
        with torch.device("meta"):
            decoder = vit.ViT3DDecoder(**options)
            self.post_quant_conv = nn.Conv3d(24, 24, 1)
        self.x_embedder = decoder.x_embedder
        self.register_tokens = decoder.register_tokens
        self.blocks = decoder.transformer_blocks
        self.norm_out = decoder.norm_out
        self.proj_out = decoder.proj_out
        self.heads, self.head_dim = options["heads"], options["dim_head"]
        self.rope = vit.RotaryEmbeddingND(
            int(self.head_dim * options["rope_dim_ratio"]), options["rope_theta"], n_dim=3, use_angle=True
        )
        self.apply_rotary = import_module(f"{package}.func")._apply_rotary_pos_emb_impl
        processor = import_module(f"{package}.vae_processor")
        self.processor = processor.VAEProcessor(
            vae_ratio=16,
            vae_ratio_t=4,
            clip_length=17,
            frame_overlap=5,
            token_overlap=2,
            tokens_chunk_size=5,
            isolated_last_frame=False,
            latent_patch_size=1,
            crop_mode="top_left",
            pixel_norm_type=cfg["pixel_norm_type"],
            use_3d_conv=True,
        )
        for name in ("latents_mean", "latents_std"):
            values = torch.tensor(cfg[name], dtype=torch.float32, device=device)
            if (
                values.shape != (24,)
                or not bool(torch.isfinite(values).all())
                or (name == "latents_std" and bool((values <= 0).any()))
            ):
                raise ValueError(f"Invalid Invoker-Render decoder {name}")
            self.register_buffer(name, values, persistent=False)
        self.load_state_dict(load_file(str(weights_file), device="cpu"), strict=True, assign=True)
        self.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    def start(self) -> DecodeState:
        return DecodeState([None] * len(self.blocks), [None] * len(self.blocks))

    def decode(self, state: DecodeState, latents: torch.Tensor, *, eos: bool) -> torch.Tensor:
        if state.finished:
            raise RuntimeError("Invoker-Render decoder already received EOS")
        values = latents.transpose(0, 1)[None].float()
        if state.pending is not None:
            values = torch.cat((state.pending, values), dim=1)
        stop = values.shape[1] // 2 * 2
        with torch.autocast(device_type=values.device.type, dtype=torch.bfloat16, enabled=values.device.type != "cpu"):
            pixels = [self._decode_pair(values[:, index : index + 2], state) for index in range(0, stop, 2)]
            tail = values[:, stop:]
            state.pending = tail.clone() if tail.shape[1] else None
            if eos and state.pending is not None:
                count = 1 if state.position % 5 == 0 else 4
                pixels.append(self._decode_pair(tail.repeat(1, 2, 1, 1, 1), state)[:, :, :count])
                state.pending = None
        state.finished = eos
        if not pixels:
            return torch.empty((0, values.shape[-2] * 16, values.shape[-1] * 16, 3), dtype=torch.uint8)
        rgb = self.processor.revert_tensor(torch.cat(pixels, dim=2).float())
        return rgb[0].mul(255).to(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu()

    @staticmethod
    def _windows(values: torch.Tensor, shift: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        batch, time, height, width = values.shape[:4]
        padded_height = (height + shift + 15) // 16 * 16
        padded_width = (width + shift + 15) // 16 * 16
        rows, cols = padded_height // 16, padded_width // 16
        rest = values.shape[4:]
        padded = values.new_zeros(batch, time, padded_height, padded_width, *rest)
        padded[:, :, shift : shift + height, shift : shift + width] = values
        mask = torch.zeros(padded_height, padded_width, dtype=torch.bool, device=values.device)
        mask[shift : shift + height, shift : shift + width] = True
        mask = mask.view(rows, 16, cols, 16).permute(0, 2, 1, 3).reshape(rows * cols, 1, 256)
        windowed = padded.view(batch, time, rows, 16, cols, 16, *rest).permute(
            0, 2, 4, 1, 3, 5, *range(6, 6 + len(rest))
        )
        return (
            windowed.reshape(batch * rows * cols, time * 256, *rest),
            mask.expand(-1, time, -1).reshape(rows * cols, time * 256),
            (batch, time, height, width, rows, cols),
        )

    @staticmethod
    def _unwindow(values: torch.Tensor, shape: tuple[int, ...], shift: int) -> torch.Tensor:
        batch, time, height, width, rows, cols = shape
        values = values.reshape(batch, rows, cols, time, 16, 16, -1).permute(0, 3, 1, 4, 2, 5, 6)
        return values.reshape(batch, time, rows * 16, cols * 16, -1)[
            :, :, shift : shift + height, shift : shift + width
        ]

    def _decode_pair(self, latents: torch.Tensor, state: DecodeState) -> torch.Tensor:
        batch, time, _, height, width = latents.shape
        device = latents.device
        current = state.position
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else latents.dtype
        raw = latents * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2))
        with torch.autocast(device_type=device.type, enabled=False):
            hidden = self.x_embedder(raw.permute(0, 2, 3, 4, 1).float()).to(dtype)
        suffix = torch.cat(
            (self.register_tokens.expand(batch, -1, -1), hidden.new_zeros(batch, 1, hidden.shape[-1])), dim=1
        ).to(dtype)
        suffix_count = suffix.shape[1]
        times = (torch.arange(current, current + time, device=device, dtype=torch.float32) + 0.5) / 7 * 2 - 1
        suffix_pos = torch.zeros(1, suffix_count, 3, dtype=torch.float32, device=device)
        suffix_pos[..., 0] = times[0]
        suffix_rope = self.rope(suffix_pos)

        def qkv(block, tensor):
            values = block.attn.to_qkv(block.norm1(tensor.float()).to(dtype))
            query, key, value = values.reshape(*tensor.shape[:-1], self.heads, 3 * self.head_dim).chunk(3, dim=-1)
            if block.attn.norm_q is not None:
                query = block.attn.norm_q(query.float()).to(dtype)
                key = block.attn.norm_k(key.float()).to(dtype)
            return query, key, value

        def attention(query, keys, values, mask=None):
            return F.scaled_dot_product_attention(
                query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
            ).transpose(1, 2)

        for i, block in enumerate(self.blocks):
            shift = 8 if i % 2 else 0
            grid = [((torch.arange(n, device=device) + shift) % 16 + 0.5) / 8 - 1 for n in (height, width)]
            rotary = self.rope(torch.stack(torch.meshgrid(times, *grid, indexing="ij"), dim=-1).reshape(1, -1, 3))
            q, k, v = qkv(block, hidden)
            qs, ks, vs = qkv(block, suffix)
            q = self.apply_rotary(q.flatten(1, 3), rotary).reshape_as(q)
            k = self.apply_rotary(k.flatten(1, 3), rotary).reshape_as(k)
            qs, ks = self.apply_rotary(qs, suffix_rope), self.apply_rotary(ks, suffix_rope)
            if state.keys[i] is not None:
                k = torch.cat((state.keys[i], k), dim=1)
                v = torch.cat((state.values[i], v), dim=1)
            qw, _, shape = self._windows(q, shift)
            kw, mask, _ = self._windows(k, shift)
            vw, _, _ = self._windows(v, shift)
            windows = qw.shape[0] // batch
            mask = torch.cat((mask, mask.new_ones(windows, suffix_count)), dim=1).repeat(batch, 1)[:, None, None, :]
            keys = torch.cat((kw, ks.repeat_interleave(windows, dim=0)), dim=1)
            values = torch.cat((vw, vs.repeat_interleave(windows, dim=0)), dim=1)
            out = self._unwindow(attention(qw, keys, values, mask), shape, shift)
            suffix_out = attention(qs, torch.cat((k.flatten(1, 3), ks), dim=1), torch.cat((v.flatten(1, 3), vs), dim=1))
            hidden = hidden + block.attn.to_out(out) * block.scale1
            suffix = suffix + block.attn.to_out(suffix_out.flatten(-2)) * block.scale1
            hidden = hidden + block.ff(block.norm2(hidden.float()).to(dtype)) * block.scale2
            suffix = suffix + block.ff(block.norm2(suffix.float()).to(dtype)) * block.scale2
            state.keys[i], state.values[i] = k[:, -4:].clone(), v[:, -4:].clone()
        hidden = self.norm_out(hidden)
        with torch.autocast(device_type=device.type, enabled=False):
            output = self.proj_out(hidden.float())
        output = output.view(batch, time, height, width, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7)
        output = output.reshape(batch, 3, time * 4, height * 16, width * 16)
        keep = torch.ones(time * 4, dtype=torch.bool, device=device)
        for i in range(time):
            if (current + i) % 5 == 0:
                keep[i * 4 : i * 4 + 3] = False
        state.position += time
        return output[:, :, keep]
