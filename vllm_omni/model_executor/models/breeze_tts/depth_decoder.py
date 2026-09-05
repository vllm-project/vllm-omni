# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze's within-frame transformer, with frame-local KV state.

Attention follows the reference BF16 matmul / FP32 softmax ordering. Each
frame overwrites its causal cache before reading it. Captured workspaces
belong to their graphs; generated frames and RNG state belong to requests.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.platforms import current_platform

from vllm_omni.platforms import current_omni_platform


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
    noise: torch.Tensor | None = None,
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
    probabilities = scores.softmax(-1)
    if noise is not None:
        return (probabilities / noise).argmax(-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def sample_graph_logits(logits: torch.Tensor, parameters: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Sample with mutable GPU parameters and a request-owned RNG draw.

    Sorting once supplies both the top-k threshold and the nucleus order.
    Filtering keeps ties at the top-k threshold, and keeps the first token
    crossing the nucleus boundary. The three reserved codec IDs remain in
    the RNG workspace, with zero probability, as in the reference sampler.
    """
    temperature, top_k, top_p = parameters.unbind()
    scaled = logits.float() / torch.where(temperature > 0, temperature, 1.0)
    values, indices = scaled.sort(dim=-1, descending=True)
    cutoff = torch.where(top_k > 0, top_k, logits.shape[-1]).long().clamp(1, logits.shape[-1]) - 1
    threshold = values.gather(-1, cutoff.expand(values.shape[0], 1))
    values = values.masked_fill(values < threshold, -torch.inf)
    cumulative = values.softmax(-1).cumsum(-1)
    remove = F.pad(cumulative[..., :-1] > top_p, (1, 0), value=False) & (top_p < 1)
    values = values.masked_fill(remove, -torch.inf)
    # Argmax of p / Exp(1) is the single-sample multinomial algorithm.
    selected = (values.softmax(-1) / noise.gather(-1, indices)).argmax(-1, keepdim=True)
    sampled = indices.gather(-1, selected).squeeze(-1)
    return torch.where(temperature > 0, sampled, logits.argmax(-1))


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
        positions: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
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
        cache[0].index_copy_(2, positions, k)
        cache[1].index_copy_(2, positions, v)
        # Fold query heads sharing one KV head into its query dimension.
        # This avoids materializing repeated keys/values for grouped-query
        # attention while preserving BF16 matmul and FP32 softmax ordering.
        grouped_q = q.reshape(batch, self.kv_heads, self.heads // self.kv_heads * length, self.dim)
        scores = (
            torch.matmul(grouped_q, cache[0].transpose(-1, -2)).reshape(batch, self.heads, length, cache[0].shape[2])
            * self.dim**-0.5
        )
        scores = scores.masked_fill(~mask, torch.finfo(q.dtype).min)
        probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        grouped_probabilities = probabilities.reshape(batch, self.kv_heads, -1, cache[0].shape[2])
        y = torch.matmul(grouped_probabilities, cache[1]).reshape(batch, self.heads, length, self.dim)
        y = y.transpose(1, 2).reshape(batch, length, self.q_size)
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
        self.register_buffer("positions", torch.arange(self.num_codebooks), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.arange(self.num_codebooks)[None, :] <= torch.arange(self.num_codebooks)[:, None],
            persistent=False,
        )
        self._graphs: dict[tuple[int, int], BreezeDepthGraph] = {}
        self._compiled_layer = None
        self._compiled_sampler = None

    def _allocate_cache(self, hidden: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        shape = (hidden.shape[0], self.config.num_key_value_heads, self.num_codebooks, self.config.head_dim)
        # Masked future positions still participate in the value matmul.
        # Initialize them to finite values so zero probabilities cannot meet NaN.
        return [(hidden.new_zeros(shape), hidden.new_zeros(shape)) for _ in self.layers]

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
        return self._generate_frame(hidden, first, self._allocate_cache(hidden), temperature, top_k, top_p, generator)

    def _generate_frame(
        self,
        hidden: torch.Tensor,
        first: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator,
        noise: torch.Tensor | None = None,
        guidance_scale: float | torch.Tensor | None = None,
        sampling_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        branches = 2 if guidance_scale is not None else 1
        branch_first = first.repeat(branches)
        x = torch.cat((hidden.unsqueeze(1), self.embed_tokens(branch_first.reshape(batch, 1))), 1)
        frames = [first]
        start = 0
        for codebook in range(self.num_codebooks - 1):
            x = self.inputs_embeds_projector(x)
            end = start + x.shape[1]
            cos = self.rope_cos[start:end].to(x.dtype)[None, None]
            sin = self.rope_sin[start:end].to(x.dtype)[None, None]
            for layer, cache in zip(self.layers, caches, strict=True):
                args = (x, self.positions[start:end], cache, self.causal_mask[start:end][None, None], cos, sin)
                if noise is None:
                    x = layer(*args)
                else:
                    x = self._compiled_layer(layer, *args)
            logits = F.linear(self.norm(x[:, -1]).float(), self.codebooks_head[codebook].T.float())
            if guidance_scale is not None:
                cond, uncond = logits.chunk(2, dim=0)
                logits = uncond + guidance_scale * (cond - uncond)
            logits[:, self.vocab_size - 3 :] = -torch.inf
            if noise is None:
                token = sample_logits(logits, temperature, top_k, top_p, generator)
            else:
                token = self._compiled_sampler(logits, sampling_parameters, noise[:, codebook])
            frames.append(token)
            x = self.embed_tokens((token.repeat(branches) + (codebook + 1) * self.vocab_size).reshape(batch, 1))
            start = end
        return torch.stack(frames, -1)

    def generate_frames(
        self,
        hidden: torch.Tensor,
        first: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        generators: list[torch.Generator],
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        branches = 2 if guidance_scale != 1.0 else 1
        key = (batch, branches)
        parameters = (temperature, top_k, top_p)
        entry = self._graphs.get(key)
        if entry is None:
            if self._compiled_layer is None:
                self._compiled_layer = torch.compile(
                    BreezeDepthLayer.forward,
                    fullgraph=True,
                    dynamic=False,
                    options={"epilogue_fusion": False, "max_autotune": True},
                )
                self._compiled_sampler = torch.compile(sample_graph_logits, fullgraph=True, dynamic=False)
            entry = BreezeDepthGraph(self, hidden, first, parameters, guidance_scale)
            self._graphs[key] = entry
        if entry.parameter_values != parameters:
            entry.parameters.copy_(entry.parameters.new_tensor(parameters))
            entry.parameter_values = parameters
        entry.hidden.copy_(hidden)
        entry.first.copy_(first)
        if entry.guidance_scale is not None and entry.guidance_value != guidance_scale:
            entry.guidance_scale.fill_(guidance_scale)
            entry.guidance_value = guidance_scale
        if temperature > 0:
            for row, generator in enumerate(generators):
                # Match the 15 independent exponential draws used by
                # torch.multinomial(..., num_samples=1), preserving each
                # request's RNG stream even when requests are batched.
                for codebook in range(self.num_codebooks - 1):
                    entry.noise[row, codebook].exponential_(generator=generator)
        entry.graph.replay()
        # The tiny RVQ result outlives this replay in request state and the
        # transfer queue. The graph owns and overwrites its output workspace.
        return entry.output.clone()


class BreezeDepthGraph:
    def __init__(
        self,
        model: BreezeDepthDecoder,
        hidden: torch.Tensor,
        first: torch.Tensor,
        parameters: tuple[float, int, float],
        guidance_scale: float = 1.0,
    ) -> None:
        self.parameter_values = parameters
        self.parameters = hidden.new_tensor(parameters, dtype=torch.float32)
        self.guidance_value = guidance_scale
        self.guidance_scale = hidden.new_tensor(guidance_scale, dtype=torch.float32) if guidance_scale != 1.0 else None
        self.hidden = torch.zeros_like(hidden)
        self.first = torch.zeros_like(first)
        self.noise = hidden.new_ones((first.shape[0], model.num_codebooks - 1, model.vocab_size), dtype=torch.float32)
        # Captured kernels retain addresses, not the Python tensors allocated
        # before capture. Keep every KV allocation alive with its graph.
        self.caches = model._allocate_cache(hidden)
        generator = torch.Generator(device=hidden.device)
        for _ in range(3):
            model._generate_frame(
                self.hidden,
                self.first,
                self.caches,
                *parameters,
                generator,
                self.noise,
                self.guidance_scale,
                self.parameters,
            )
        current_omni_platform.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            self.graph, pool=current_platform.get_global_graph_pool(), capture_error_mode="thread_local"
        ):
            self.output = model._generate_frame(
                self.hidden,
                self.first,
                self.caches,
                *parameters,
                generator,
                self.noise,
                self.guidance_scale,
                self.parameters,
            )
