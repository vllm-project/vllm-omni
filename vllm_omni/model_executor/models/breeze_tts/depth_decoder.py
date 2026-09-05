# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze's within-frame transformer, with frame-local KV state.

Attention follows the reference BF16 matmul / FP32 softmax ordering. The
cache belongs to one generate_frame call; interleaved requests cannot share it.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1)
    scores = logits.float() / temperature
    if top_k > 0:
        threshold = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(dim=-1, descending=True)
        probabilities = values.softmax(-1)
        cumulative = probabilities.cumsum(-1)
        # Keep the first candidate crossing the nucleus boundary as in HF.
        remove = cumulative - probabilities >= top_p
        scores = scores.scatter(-1, indices, values.masked_fill(remove, -torch.inf))
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)


class BreezeRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(x.dtype)


class BreezeDepthLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.q_size = self.heads * self.dim
        self.kv_size = self.kv_heads * self.dim
        self.qkv = nn.Linear(config.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.gate_up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.input_layernorm = BreezeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BreezeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start: int,
        cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        residual = x
        q, k, v = self.qkv(self.input_layernorm(x)).split([self.q_size, self.kv_size, self.kv_size], -1)
        q = q.view(batch, length, self.heads, self.dim).transpose(1, 2)
        k = k.view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        v = v.view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        q1, q2 = q.chunk(2, -1)
        k1, k2 = k.chunk(2, -1)
        q = q * cos + torch.cat((-q2, q1), -1) * sin
        k = k * cos + torch.cat((-k2, k1), -1) * sin
        end = start + length
        cache[0][:, :, start:end] = k
        cache[1][:, :, start:end] = v
        keys = cache[0][:, :, :end].repeat_interleave(self.heads // self.kv_heads, 1)
        values = cache[1][:, :, :end].repeat_interleave(self.heads // self.kv_heads, 1)
        scores = torch.matmul(q, keys.transpose(-1, -2)) * self.dim**-0.5
        causal = torch.arange(end, device=x.device)[None, :] <= torch.arange(start, end, device=x.device)[:, None]
        scores = scores.masked_fill(~causal, torch.finfo(q.dtype).min)
        probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(probabilities, values).transpose(1, 2).reshape(batch, length, self.q_size)
        x = residual + self.o_proj(y)
        gate, up = self.gate_up(self.post_attention_layernorm(x)).chunk(2, -1)
        return x + self.down_proj(F.silu(gate) * up)


class BreezeDepthDecoder(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(self.num_codebooks * self.vocab_size, config.audio_embed_size)
        self.inputs_embeds_projector = nn.Linear(config.audio_embed_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([BreezeDepthLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = BreezeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.codebooks_head = nn.Parameter(
            torch.empty(self.num_codebooks - 1, config.hidden_size, self.vocab_size, dtype=torch.float32)
        )
        rope = config.rope_parameters
        inv = 1.0 / (rope["rope_theta"] ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        wavelength = 2 * torch.pi / inv
        low = rope["original_max_position_embeddings"] / rope["low_freq_factor"]
        high = rope["original_max_position_embeddings"] / rope["high_freq_factor"]
        smooth = (rope["original_max_position_embeddings"] / wavelength - rope["low_freq_factor"]) / (
            rope["high_freq_factor"] - rope["low_freq_factor"]
        )
        scaled = (1 - smooth) * inv / rope["factor"] + smooth * inv
        inv = torch.where(wavelength > low, inv / rope["factor"], torch.where(wavelength < high, inv, scaled))
        angles = torch.outer(torch.arange(self.num_codebooks).float(), inv)
        angles = torch.cat((angles, angles), -1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def generate_frame(
        self,
        hidden: torch.Tensor,
        first: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        shape = (batch, self.config.num_key_value_heads, self.num_codebooks, self.config.head_dim)
        caches = [(hidden.new_empty(shape), hidden.new_empty(shape)) for _ in self.layers]
        x = torch.cat((hidden.unsqueeze(1), self.embed_tokens(first.reshape(batch, 1))), 1)
        frames = [first]
        start = 0
        for codebook in range(self.num_codebooks - 1):
            x = self.inputs_embeds_projector(x)
            end = start + x.shape[1]
            cos = self.rope_cos[start:end].to(x.dtype)[None, None]
            sin = self.rope_sin[start:end].to(x.dtype)[None, None]
            for layer, cache in zip(self.layers, caches, strict=True):
                x = layer(x, start, cache, cos, sin)
            logits = F.linear(self.norm(x[:, -1]).float(), self.codebooks_head[codebook].T.float())
            token = sample_logits(logits[:, : self.vocab_size - 3], temperature, top_k, top_p, generator)
            frames.append(token)
            x = self.embed_tokens((token + (codebook + 1) * self.vocab_size).reshape(batch, 1))
            start = end
        return torch.stack(frames, -1)
