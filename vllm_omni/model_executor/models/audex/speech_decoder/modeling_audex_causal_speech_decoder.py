# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from .configuration_audex_causal_speech_decoder import AudexCausalSpeechDecoderConfig
from .streaming_utils import load_audex_causal_speech_decoder as _load_decoder_for_remote_code

REMOTE_CODE_IMPORTS = (_load_decoder_for_remote_code,)


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()
        self._rope_ready = False

    def rope_init(self, device: torch.device | None = None) -> None:
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        self.max_seq_len = max_seq_len
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        needed_seq_len = seq_len if input_pos is None else int(input_pos.max().item()) + 1
        if (
            not getattr(self, "_rope_ready", False)
            or self.theta.device != x.device
            or needed_seq_len > self.cache.size(0)
        ):
            self.rope_init(device=x.device)
            if needed_seq_len > self.cache.size(0):
                self.build_rope_cache(max(needed_seq_len, self.cache.size(0) * 2))
            self._rope_ready = True

        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        return x_out.flatten(3).type_as(x)


class CausalCodecDecoderCache:
    def __init__(self) -> None:
        self.key_values: dict[int, tuple[Tensor, Tensor]] = {}
        self.position = 0

    def input_positions(self, length: int, device: torch.device) -> Tensor:
        return torch.arange(self.position, self.position + length, device=device).unsqueeze(0)

    def update(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if layer_idx in self.key_values:
            prev_key, prev_value = self.key_values[layer_idx]
            key = torch.cat([prev_key, key], dim=2)
            value = torch.cat([prev_value, value], dim=2)
        self.key_values[layer_idx] = (key, value)
        return key, value

    def advance(self, length: int) -> None:
        self.position += length

    def reset(self) -> None:
        self.key_values.clear()
        self.position = 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.silu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_embed: RotaryPositionalEmbeddings, layer_idx: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim must be divisible by n_heads, got dim={dim}, n_heads={n_heads}")
        self.n_heads = n_heads
        self.layer_idx = layer_idx
        self.rotary_embed = rotary_embed
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: CausalCodecDecoderCache | None = None,
        input_pos: Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        head_dim = qkv.size(-1) // (3 * self.n_heads)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.rotary_embed(q.transpose(1, 2), input_pos=input_pos).transpose(1, 2)
        k = self.rotary_embed(k.transpose(1, 2), input_pos=input_pos).transpose(1, 2)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if input_pos is None:
                raise ValueError("input_pos is required when cache is set")
            k, v = cache.update(self.layer_idx, k, v)
            key_pos = torch.arange(k.size(2), device=x.device).view(1, 1, 1, -1)
            attn_mask = key_pos <= input_pos.view(input_pos.size(0), 1, -1, 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_embed: RotaryPositionalEmbeddings, layer_idx: int):
        super().__init__()
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_embed=rotary_embed, layer_idx=layer_idx)
        self.mlp = MLP(dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: CausalCodecDecoderCache | None = None,
        input_pos: Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.att(self.att_norm(x), cache=cache, input_pos=input_pos)
        return x + self.mlp(self.ffn_norm(x))


class PatchHead(nn.Module):
    def __init__(self, dim: int, hop_length: int = 320):
        super().__init__()
        self.proj = nn.Linear(dim, hop_length, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.proj(x))
        return x.reshape(x.size(0), 1, -1)


class CausalVocosBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        depth: int = 12,
        heads: int = 32,
        pos_meb_dim: int = 64,
    ):
        super().__init__()
        rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=rotary_embed, layer_idx=idx)
                for idx in range(depth)
            ]
        )
        self.final_layer_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cache: CausalCodecDecoderCache | None = None) -> torch.Tensor:
        input_pos = None
        if cache is not None:
            input_pos = cache.input_positions(x.size(1), x.device).expand(x.size(0), -1)
        for block in self.transformers:
            x = block(x, cache=cache, input_pos=input_pos)
        if cache is not None:
            cache.advance(x.size(1))
        return self.final_layer_norm(x)


class CausalCodecDecoderVocos(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        depth: int = 12,
        heads: int = 32,
        pos_meb_dim: int = 64,
        hop_length: int = 320,
        vq_dim: int = 2048,
        lookahead_steps: int = 0,
    ):
        super().__init__()
        if lookahead_steps < 0:
            raise ValueError(f"lookahead_steps must be >= 0, got {lookahead_steps}")

        self.wav_proj = nn.Linear(hop_length, hidden_dim, bias=False)
        self.fc_post_a = nn.Linear(vq_dim, hidden_dim, bias=False)
        self.lookahead_steps = lookahead_steps
        if lookahead_steps > 0:
            self.lookahead_conv = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=lookahead_steps + 1,
                padding=0,
                groups=hidden_dim,
                bias=False,
            )
            self.lookahead_act = nn.SiLU()
            self.lookahead_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
            nn.init.zeros_(self.lookahead_proj.weight)
        else:
            self.lookahead_conv = None
            self.lookahead_act = None
            self.lookahead_proj = None
        self.backbone = CausalVocosBackbone(hidden_dim, depth, heads, pos_meb_dim)
        self.head = PatchHead(hidden_dim, hop_length)

    def _project_tokens(self, vq_emb: torch.Tensor) -> torch.Tensor:
        return self.fc_post_a(vq_emb)

    def _apply_lookahead(self, x: torch.Tensor) -> torch.Tensor:
        if self.lookahead_conv is None:
            return x
        if self.lookahead_act is None or self.lookahead_proj is None:
            raise RuntimeError("lookahead modules are not initialized")
        h = F.pad(x.transpose(1, 2), (0, self.lookahead_steps))
        h = self.lookahead_proj(self.lookahead_act(self.lookahead_conv(h)))
        return x + h.transpose(1, 2)

    def _apply_lookahead_window(self, x: torch.Tensor) -> torch.Tensor:
        if self.lookahead_conv is None:
            return x
        if self.lookahead_act is None or self.lookahead_proj is None:
            raise RuntimeError("lookahead modules are not initialized")
        if x.size(1) <= self.lookahead_steps:
            raise ValueError(f"lookahead window needs more than {self.lookahead_steps} frames, got {x.size(1)}")
        h = self.lookahead_proj(self.lookahead_act(self.lookahead_conv(x.transpose(1, 2))))
        return x[:, : h.size(2)] + h.transpose(1, 2)

    def decode_cached(
        self,
        vq_emb: torch.Tensor,
        cache: CausalCodecDecoderCache,
        lookahead_vq_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._project_tokens(vq_emb)
        if self.lookahead_steps > 0:
            if lookahead_vq_emb is None:
                lookahead_vq_emb = vq_emb.new_zeros(vq_emb.size(0), self.lookahead_steps, vq_emb.size(-1))
            if lookahead_vq_emb.size(1) != self.lookahead_steps:
                raise ValueError(
                    f"lookahead_vq_emb must have {self.lookahead_steps} frames, got {lookahead_vq_emb.size(1)}"
                )
            lookahead_x = self._project_tokens(lookahead_vq_emb)
            x = self._apply_lookahead_window(torch.cat([x, lookahead_x], dim=1))
        x = self.backbone(x, cache=cache)
        return self.head(x)

    def forward(
        self,
        vq_emb: torch.Tensor,
        patched_wav: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        x = self._project_tokens(vq_emb)
        x = self._apply_lookahead(x)
        if patched_wav is not None:
            h = self.wav_proj(patched_wav)
            mask = torch.bernoulli(
                torch.full(
                    (x.size(0), x.size(1), 1),
                    min(max(alpha, 0.0), 1.0),
                    device=x.device,
                    dtype=x.dtype,
                )
            )
            x = x + h * mask
        return self.head(self.backbone(x))


class AudexSpeechTokenEmbedder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        token_embed_dim: int,
        codebook_levels: Sequence[int],
    ) -> None:
        super().__init__()
        if len(codebook_levels) != token_embed_dim:
            raise ValueError(
                f"token_embed_dim={token_embed_dim} must match codebook_levels length={len(codebook_levels)}"
            )
        self.codebook_levels = tuple(int(level) for level in codebook_levels)
        self.project_out = nn.Linear(token_embed_dim, output_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.size(-1) != 1:
            raise ValueError(f"indices last dimension must be 1, got {indices.size(-1)}")
        levels = torch.tensor(self.codebook_levels, dtype=torch.long, device=indices.device)
        basis = torch.cumprod(torch.cat([levels.new_ones(1), levels[:-1]]), dim=0)
        level_indices = (indices.long() // basis) % levels
        dtype = self.project_out.weight.dtype
        codes = level_indices.to(dtype=dtype)
        levels = levels.to(dtype=dtype)
        codes = codes * (2.0 / (levels - 1.0)) - 1.0
        return self.project_out(codes)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self(indices)


class AudexCausalSpeechDecoderModel(PreTrainedModel):
    config_class = AudexCausalSpeechDecoderConfig
    base_model_prefix = "module"
    all_tied_weights_keys: dict[str, Any] = {}
    Cache = CausalCodecDecoderCache

    def __init__(self, config: AudexCausalSpeechDecoderConfig):
        super().__init__(config)
        self.audex_speech_token_embedder = AudexSpeechTokenEmbedder(
            output_dim=config.vq_dim,
            token_embed_dim=config.token_embed_dim,
            codebook_levels=config.codebook_levels,
        )
        self.module = CausalCodecDecoderVocos(
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            heads=config.heads,
            pos_meb_dim=config.pos_meb_dim,
            hop_length=config.hop_length,
            vq_dim=config.vq_dim,
            lookahead_steps=config.lookahead_steps,
        )

    @property
    def lookahead_steps(self) -> int:
        return self.module.lookahead_steps

    def create_cache(self) -> CausalCodecDecoderCache:
        return CausalCodecDecoderCache()

    def decode_cached(
        self,
        vq_emb: torch.Tensor,
        cache: CausalCodecDecoderCache,
        lookahead_vq_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.module.decode_cached(vq_emb, cache, lookahead_vq_emb=lookahead_vq_emb)

    def create_session(
        self,
        *,
        chunk_frames: int = 1,
        sample_rate: int | None = None,
        return_numpy: bool = True,
    ) -> AudexCausalSpeechDecoderSession:
        return AudexCausalSpeechDecoderSession(
            decoder=self,
            chunk_frames=chunk_frames,
            sample_rate=sample_rate or self.config.sample_rate,
            return_numpy=return_numpy,
        )

    def forward(
        self,
        vq_emb: torch.Tensor,
        patched_wav: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        return self.module(vq_emb, patched_wav=patched_wav, alpha=alpha)


class AudexCausalSpeechDecoderSession:
    def __init__(
        self,
        decoder: AudexCausalSpeechDecoderModel,
        *,
        chunk_frames: int,
        sample_rate: int,
        return_numpy: bool,
    ):
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
        self.decoder = decoder
        self.chunk_frames = chunk_frames
        self.sample_rate = sample_rate
        self.return_numpy = return_numpy
        self.cache = decoder.create_cache()
        self.buffer: list[list[int]] = []

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def reset(self) -> None:
        self.cache = self.decoder.create_cache()
        self.buffer.clear()

    def push(self, token_frames: Sequence[Sequence[int]]) -> Iterator[tuple[int, Any]]:
        self.buffer.extend(list(frame) for frame in token_frames)
        yield from self._drain(flush=False)

    def flush(self) -> Iterator[tuple[int, Any]]:
        yield from self._drain(flush=True)

    def _drain(self, *, flush: bool) -> Iterator[tuple[int, Any]]:
        ready_frames = len(self.buffer) - self.decoder.lookahead_steps
        while self.buffer and (flush or ready_frames >= self.chunk_frames):
            emit_frames = min(self.chunk_frames, len(self.buffer)) if flush else self.chunk_frames
            wav = self._decode_buffered_frames(emit_frames, flush=flush)
            del self.buffer[:emit_frames]
            ready_frames = len(self.buffer) - self.decoder.lookahead_steps
            yield self.sample_rate, self._format_chunk(wav)

    def _embed_speech_token_frames(self, token_frames: Sequence[Sequence[int]]) -> torch.Tensor:
        indices = torch.tensor(token_frames, dtype=torch.long, device=self.device).unsqueeze(0)
        return self.decoder.audex_speech_token_embedder.get_output_from_indices(indices)

    def _decode_buffered_frames(self, emit_frames: int, *, flush: bool) -> torch.Tensor:
        with torch.inference_mode():
            vq_emb = self._embed_speech_token_frames(self.buffer[:emit_frames])
            lookahead_vq_emb = None
            if self.decoder.lookahead_steps > 0:
                future_frames = self.buffer[emit_frames : emit_frames + self.decoder.lookahead_steps]
                future_parts = []
                if future_frames:
                    future_parts.append(self._embed_speech_token_frames(future_frames))
                missing_frames = self.decoder.lookahead_steps - len(future_frames) if flush else 0
                if missing_frames > 0:
                    future_parts.append(vq_emb.new_zeros(vq_emb.size(0), missing_frames, vq_emb.size(-1)))
                lookahead_vq_emb = torch.cat(future_parts, dim=1) if future_parts else None
            return self.decoder.decode_cached(vq_emb, self.cache, lookahead_vq_emb=lookahead_vq_emb)

    def _format_chunk(self, wav: torch.Tensor) -> Any:
        chunk = wav.squeeze().float().detach().cpu()
        if not self.return_numpy:
            return chunk

        import numpy as np

        return chunk.numpy().astype(np.float32, copy=False)
