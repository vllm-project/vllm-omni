#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Source: https://github.com/SolarX-Team/Realization-Streaming-VAE-Decoder
# Revision: a7c61a6e02bef8539178a17b6e150a775b3a0ebb; see LICENSE.
"""Latent-by-latent streaming decode for the MiniMax Realization video VAE.

The official decoder is a bidirectional ViT over windows of 7 latents with stride 5, so
the frames of a latent are ready only after the 2 latents after it exist. StreamingDecoder
keeps the weights and the window geometry and decodes every latent when it arrives:

window    slots 0-6 of a window that starts on a multiple of 5; every slot gets the RoPE
          position it has in a full 7-latent window.
step      the new latent runs alone at its slot and attends to the cached K/V of the slots
          before it. Its frames are emitted right away and never changed: 1 frame for
          latents 0, 5, 10, ..., 4 frames for the others.
shift     when the window moves, the 2 latents it shares with the previous window run
          again at slots 0 and 1, in the pass of the window's third latent. Their frames
          were emitted from the previous window; this pass only refills the cache.
first_refresh  slot 0 runs again together with latent 1, once per stream (V3).
compensate     a + b * x is taken off every emitted pixel, frame 0 also c * x * x, x being
          the pixel's own value. The constants are per channel and per frame for latents
          0-5 (tables/first_window.pt) and per slot, channel and frame position after that
          (tables/steady.pt), fitted offline on 64 x 64 cell means.

Keys are cached after RoPE, one [tiles, suffix + 7 * hw, heads, dim_head] buffer per layer
with the register and cls tokens first; attention reads the first suffix + (slot + 1) * hw
rows in place. compile=True runs the norms, RoPE, residuals and the FFN gate of each block
through torch.compile, which fuses them into a few kernels; the matmuls and attention stay
as they are. Inductor rounds to bfloat16 where eager does (emulate_precision_casts), the
remaining difference is the order of reductions.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F

TABLES = Path(__file__).resolve().parent / "tables"
_COMPILED: dict = {}


def load_table(path: str | Path) -> dict:
    """{"a", "b", "pool"} and an optional "c", as float32 on the CPU."""
    table = torch.load(path, map_location="cpu")
    return {key: value.float() if torch.is_tensor(value) else value for key, value in table.items()}


def _token_ids(dims: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """video_vae.func.create_token_ids with id_type="length_normalized"."""
    axes = []
    for size in dims:
        coords = torch.arange(0.5, size, dtype=dtype, device=device)
        coords = coords / size
        axes.append(2.0 * coords - 1.0)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, len(dims) - 1).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """video_vae.func._apply_rotary_pos_emb_impl."""
    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)
    rot = cos.shape[-1]
    if rot < t.shape[-1]:
        head, rest = t[..., :rot], t[..., rot:]
        return torch.cat(((head * cos) + (_rotate_half(head) * sin), rest), dim=-1)
    return (t * cos) + (_rotate_half(t) * sin)


def _unpack(x: torch.Tensor, patch: int, patch_t: int, frames: int, height: int, width: int) -> torch.Tensor:
    """video_vae.vae_vit._unpack_tensors_3d."""
    batch, _, channels = x.shape
    channels //= patch_t * patch * patch
    x = x.view(batch, frames // patch_t, height // patch, width // patch, channels, patch_t, patch, patch)
    return x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(batch, channels, frames, height, width)


def _rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """nn.RMSNorm on a float32 copy of x, as the decoder runs its norms."""
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x if weight is None else x * weight


def _attn_in(tokens, norm, eps, qkv_weight, qkv_bias, q_norm, q_eps, k_norm, k_eps, heads, dim_head, cos, sin):
    normed = _rms(tokens, norm, eps).to(tokens.dtype)
    qkv = F.linear(normed, qkv_weight, qkv_bias).view(tokens.shape[0], tokens.shape[1], heads, 3 * dim_head)
    query, key, value = qkv.chunk(3, dim=-1)
    query = _rope(_rms(query, q_norm, q_eps).to(tokens.dtype), cos, sin)
    key = _rope(_rms(key, k_norm, k_eps).to(tokens.dtype), cos, sin)
    return query, key, value


def _attn_out(tokens, attn, out_weight, out_bias, scale1, norm, eps, w1_weight, w1_bias, w2_weight, w2_bias, scale2):
    tokens = tokens + F.linear(attn.reshape(tokens.shape), out_weight, out_bias) * scale1
    hidden = F.linear(_rms(tokens, norm, eps).to(tokens.dtype), w1_weight, w1_bias)
    gate, hidden = hidden.chunk(2, dim=-1)
    hidden = F.linear(F.silu(gate) * hidden, w2_weight, w2_bias)
    return tokens + hidden * scale2


def _compiled(fn):
    if fn not in _COMPILED:
        torch._inductor.config.emulate_precision_casts = True
        dynamo = torch._dynamo.config
        limit = "recompile_limit" if hasattr(dynamo, "recompile_limit") else "cache_size_limit"
        setattr(dynamo, limit, max(getattr(dynamo, limit), 64))
        _COMPILED[fn] = torch.compile(fn, fullgraph=True, dynamic=False)
    return _COMPILED[fn]


def _eps(norm: nn.Module) -> float:
    return torch.finfo(torch.float32).eps if norm.eps is None else float(norm.eps)


class StreamingDecoder:
    """Decode Realization latents one at a time, without any future latent.

    vae            the loaded Realization VAE model (realization_vae.load_vae)
    first_refresh  run slot 0 again together with latent 1; off gives V2
    compensate     subtract the a + b * x tables; off, with first_refresh off, gives V1
    compile        fuse the element-wise work of every block with torch.compile
    token_budget   tokens per forward; the spatial tiles of a step are batched up to it

    One decoder holds one stream and its K/V cache: 2 x 36 x [tiles, 5 + 7 * 256, 32, 64]
    in bfloat16, 14.8 GB at 1344x768 (28 tiles). reset() starts the next stream and keeps
    the buffers.
    """

    def __init__(
        self,
        vae,
        first_refresh: bool = True,
        compensate: bool = True,
        compile: bool = True,
        token_budget: int = 26000,
        tables: str | Path = TABLES,
    ) -> None:
        self.vae = vae
        self.decoder = vae.decoder
        self.chunk = int(vae.tokens_chunk_size)
        self.window = self.chunk + int(vae.token_overlap)
        self.ratio = int(vae.vae_ratio_t)
        self.key_frames = self.ratio - int(vae.frame_pre_padding)
        self.first_refresh = bool(first_refresh)
        self.compile = bool(compile)
        self.token_budget = int(token_budget)
        self.first_table = self.steady_table = None
        if compensate:
            self.first_table = load_table(Path(tables) / "first_window.pt")
            self.steady_table = load_table(Path(tables) / "steady.pt")
            first_frames = self.first_frame(self.chunk) + self.frames_of(self.chunk)
            if tuple(self.first_table["a"].shape) != (3, first_frames) or tuple(self.steady_table["a"].shape) != (
                self.window,
                3,
                self.ratio,
            ):
                raise ValueError("the compensation tables do not match this VAE's window")
            c = self.first_table.get("c")
            self._first_c = None if c is None else c.abs().sum(0) > 0
        block = self.decoder.transformer_blocks[0]
        if self.compile and not (
            block.use_scale
            and block.ff.use_gated
            and isinstance(block.ff.act_fn, nn.SiLU)
            and all(isinstance(n, nn.RMSNorm) for n in (block.norm1, block.norm2, block.attn.norm_q, block.attn.norm_k))
        ):
            raise ValueError("compile=True expects RMSNorm everywhere, use_scale and a gated SiLU FFN")
        registers = self.decoder.register_tokens
        self.suffix = 1 + (0 if registers is None else int(registers.shape[1]))
        self._plan = None
        self._k = self._v = None
        self._rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._device_tables: dict[str, dict] = {}
        self.reset()

    def reset(self) -> None:
        """Start a new stream. The K/V buffers stay allocated for a stream of the same size."""
        self.received = 0
        self.slot_passes = 0
        self._start = None
        self._valid = [False] * self.window
        self._latents: dict[int, torch.Tensor] = {}

    def frames_of(self, index: int) -> int:
        """Frames emitted for latent `index`."""
        return self.key_frames if index % self.chunk == 0 else self.ratio

    def first_frame(self, index: int) -> int:
        """Index of the first frame of latent `index` in the decoded video."""
        group, rest = divmod(index, self.chunk)
        head = 0 if rest == 0 else self.key_frames + self.ratio * (rest - 1)
        return group * (self.key_frames + (self.chunk - 1) * self.ratio) + head

    @torch.inference_mode()
    def push(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the next latent, [1, C, 1, h, w] as decode_base takes it.

        Returns its frames, [1, 3, frames_of(index), H, W] in the VAE's normalized pixel
        space; realization_vae.to_uint8 turns them into RGB.
        """
        index = self.received
        z = self._tile(latent, new_stream=index == 0)
        self.received += 1
        start = self._window_start(index)
        if start != self._start:
            self._start = start
            self._valid = [False] * self.window
            self._latents = {i: kept for i, kept in self._latents.items() if i >= start}
        self._latents[index] = z
        slot = index - start
        fresh = {s for s in range(slot) if not self._valid[s]} | {slot}
        if self.first_refresh and start == 0 and slot == 1:
            fresh.add(0)
        fresh = sorted(fresh)

        z = torch.cat([self._latents[start + s] for s in fresh], dim=2)
        rope = self._rope_of(fresh)
        batch = max(1, self.token_budget // (len(fresh) * self._plan["hw"] + self.suffix))
        spans = [slice(b, min(b + batch, z.shape[0])) for b in range(0, z.shape[0], batch)]
        tiles = torch.cat([self._layers(span, z[span], fresh, slot, rope) for span in spans], dim=0)
        for s in fresh:
            self._valid[s] = True
        self.slot_passes += len(fresh)
        out = self._blend(tiles)[:, :, self.ratio - self.frames_of(index) :]
        return self._compensate(index, slot, out)

    def stream(self, latents: torch.Tensor):
        """push() every latent of [1, C, T, h, w] in order, yielding their frames."""
        for index in range(latents.shape[2]):
            yield self.push(latents[:, :, index : index + 1])

    def _window_start(self, index: int) -> int:
        lowest = max(0, index - self.window + 1)
        return ((lowest + self.chunk - 1) // self.chunk) * self.chunk

    def _tile(self, latent: torch.Tensor, new_stream: bool) -> torch.Tensor:
        weight = self.decoder.x_embedder.weight
        z = self.vae.post_quant_conv(latent.to(device=weight.device, dtype=weight.dtype))
        key = (int(z.shape[-2]), int(z.shape[-1]))
        if self._plan is None or self._plan["latent"] != key:
            if not new_stream:
                raise ValueError("the latent size changed inside a stream; call reset() first")
            self._plan = self._make_plan(z)
        pieces = [z[..., top : top + h, left : left + w] for top, h, left, w in self._owned_tiles(self._plan["tiles"])]
        return torch.cat(pieces, dim=0).to(weight.dtype)

    def _local_tile_count(self, total):
        return total

    def _owned_tiles(self, tiles):
        return tiles

    def _make_plan(self, z: torch.Tensor) -> dict:
        ratio = int(self.vae.vae_ratio)
        y_idx, y_len, y_overlap = self.vae.split_tiles(z.shape[-2] * ratio, True)
        x_idx, x_len, x_overlap = self.vae.split_tiles(z.shape[-1] * ratio, True)
        tiles = []
        for top, height in zip(y_idx, y_len):
            for left, width in zip(x_idx, x_len):
                tiles.append((top // ratio, height // ratio, left // ratio, width // ratio))
        shapes = {(h, w) for _, h, _, w in tiles}
        if len(shapes) != 1:
            raise RuntimeError(f"tiles differ in shape: {sorted(shapes)}")
        h, w = shapes.pop()
        dtype = self.decoder.x_embedder.weight.dtype
        ids = _token_ids((self.window, h, w), z.device, dtype)
        suffix_ids = torch.zeros((1, self.suffix, 3), device=z.device, dtype=dtype)
        cos, sin = self.decoder.pos_embed(torch.cat([ids, suffix_ids], dim=1))
        attn = self.decoder.transformer_blocks[0].attn
        shape = (self._local_tile_count(len(tiles)), self.suffix + self.window * h * w, attn.heads, attn.dim_head)
        self._k = self._v = None
        self._k = [torch.empty(shape, dtype=dtype, device=z.device) for _ in self.decoder.transformer_blocks]
        self._v = [torch.empty(shape, dtype=dtype, device=z.device) for _ in self.decoder.transformer_blocks]
        self._rope_cache = {}
        return {
            "latent": (int(z.shape[-2]), int(z.shape[-1])),
            "tiles": tiles,
            "rows": len(y_idx),
            "cols": len(x_idx),
            "y_overlap": y_overlap,
            "x_overlap": x_overlap,
            "h": h,
            "w": w,
            "hw": h * w,
            "cos": cos,
            "sin": sin,
        }

    def _rope_of(self, fresh: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """cos, sin for the fresh slots followed by the suffix tokens, whose positions are all zero."""
        key = tuple(fresh)
        if key not in self._rope_cache:
            hw, tail = self._plan["hw"], self.window * self._plan["hw"]
            self._rope_cache[key] = tuple(
                torch.cat([table[:, s * hw : (s + 1) * hw] for s in fresh] + [table[:, tail:]], dim=1)
                for table in (self._plan["cos"], self._plan["sin"])
            )
        return self._rope_cache[key]

    def _layers(self, span: slice, z: torch.Tensor, fresh: list[int], slot: int, rope) -> torch.Tensor:
        decoder = self.decoder
        plan = self._plan
        hw = plan["hw"]
        suffix = self.suffix
        batch = z.shape[0]
        n_fresh = len(fresh) * hw
        hidden = z.permute(0, 2, 3, 4, 1).reshape(batch, n_fresh, z.shape[1])
        with torch.autocast("cuda", enabled=False):
            hidden = decoder.x_embedder(hidden)
        parts = [hidden]
        if decoder.register_tokens is not None:
            parts.append(decoder.register_tokens.expand(batch, -1, -1).to(hidden.dtype))
        parts.append(hidden.new_zeros(batch, 1, hidden.shape[-1]))
        tokens = torch.cat(parts, dim=1)
        runs = []
        for index, s in enumerate(fresh):
            if runs and runs[-1][1] + runs[-1][2] == s:
                runs[-1][2] += 1
            else:
                runs.append([index, s, 1])
        visible = suffix + (slot + 1) * hw
        cos, sin = rope
        attn_in = _compiled(_attn_in) if self.compile else None
        attn_out = _compiled(_attn_out) if self.compile else None

        for layer, block in enumerate(decoder.transformer_blocks):
            attention = block.attn
            if self.compile:
                query, key, value = attn_in(
                    tokens,
                    block.norm1.weight,
                    _eps(block.norm1),
                    attention.to_qkv.weight,
                    attention.to_qkv.bias,
                    attention.norm_q.weight,
                    _eps(attention.norm_q),
                    attention.norm_k.weight,
                    _eps(attention.norm_k),
                    attention.heads,
                    attention.dim_head,
                    cos,
                    sin,
                )
            else:
                normed = block.norm1(tokens.float()).to(tokens.dtype)
                query, key, value = self._qkv(attention, normed)
                query = _rope(query, cos, sin)
                key = _rope(key, cos, sin)
            k_cache = self._k[layer][span]
            v_cache = self._v[layer][span]
            k_cache[:, :suffix] = key[:, n_fresh:]
            v_cache[:, :suffix] = value[:, n_fresh:]
            for index, s, count in runs:
                rows = slice(suffix + s * hw, suffix + (s + count) * hw)
                cols = slice(index * hw, (index + count) * hw)
                k_cache[:, rows] = key[:, cols]
                v_cache[:, rows] = value[:, cols]
            attn = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                k_cache[:, :visible].transpose(1, 2),
                v_cache[:, :visible].transpose(1, 2),
                dropout_p=0.0,
            ).transpose(1, 2)
            if self.compile:
                tokens = attn_out(
                    tokens,
                    attn,
                    attention.to_out.weight,
                    attention.to_out.bias,
                    block.scale1,
                    block.norm2.weight,
                    _eps(block.norm2),
                    block.ff.w1.weight,
                    block.ff.w1.bias,
                    block.ff.w2.weight,
                    block.ff.w2.bias,
                    block.scale2,
                )
            else:
                attn = attention.to_out(attn.reshape(batch, tokens.shape[1], -1))
                tokens = tokens + (attn * block.scale1 if block.use_scale else attn)
                normed = block.norm2(tokens.float()).to(tokens.dtype)
                ff = block.ff(normed)
                tokens = tokens + (ff * block.scale2 if block.use_scale else ff)

        own = fresh.index(slot) * hw
        with torch.autocast("cuda", enabled=False):
            output = decoder.proj_out(decoder.norm_out(tokens[:, own : own + hw].contiguous()))
        config = decoder.config
        return _unpack(
            output,
            config.patch_size,
            config.patch_size_t,
            config.patch_size_t,
            plan["h"] * config.patch_size,
            plan["w"] * config.patch_size,
        )

    @staticmethod
    def _qkv(attention: nn.Module, hidden: torch.Tensor):
        batch, length, _ = hidden.shape
        qkv = attention.to_qkv(hidden).view(batch, length, -1, 3 * attention.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        if attention.norm_q is not None:
            query = attention.norm_q(query.float()).to(query.dtype)
        if attention.norm_k is not None:
            key = attention.norm_k(key.float()).to(key.dtype)
        return query, key, value

    def _blend(self, tiles: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        cols = plan["cols"]
        rows = [[tiles[i * cols + j : i * cols + j + 1] for j in range(cols)] for i in range(plan["rows"])]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.vae.blend(rows[i - 1][j], tile, plan["y_overlap"][i - 1], dim=-2)
                if j > 0:
                    tile = self.vae.blend(row[j - 1], tile, plan["x_overlap"][j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -plan["y_overlap"][i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -plan["x_overlap"][j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _table(self, name: str, device: torch.device) -> dict:
        if name not in self._device_tables:
            table = self.first_table if name == "first" else self.steady_table
            self._device_tables[name] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in table.items()}
        return self._device_tables[name]

    def _compensate(self, index: int, slot: int, out: torch.Tensor) -> torch.Tensor:
        if self.first_table is None:
            return out
        count = out.shape[2]
        if index <= self.chunk:
            first = self.first_frame(index)
            table = self._table("first", out.device)

            def pick(t):
                return t[:, first : first + count].reshape(1, 3, count, 1, 1)

            use_c = self._first_c is not None and bool(self._first_c[first : first + count].any())
        else:
            table = self._table("steady", out.device)

            def pick(t):
                return t[slot, :, self.ratio - count :].reshape(1, 3, count, 1, 1)

            use_c = "c" in table
        x = out.float()
        shift = pick(table["a"]) + pick(table["b"]) * x
        if use_c:
            shift = shift + pick(table["c"]) * x * x
        return (x - shift).to(out.dtype)
