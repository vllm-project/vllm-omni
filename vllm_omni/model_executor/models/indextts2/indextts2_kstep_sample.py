# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Multi-step (K-step) device-resident sampling and decode loop for IndexTTS Talker.

Optimized with:
1. GPU-resident tensor sampling reducing per-token D2H synchronization across K-step sub-sequences.
2. Top-K Sliced Top-P fast path (sliced to 30 elements, eliminating 8194 Radix Sort).
3. Gumbel-Max in-tensor selection (eliminating aten::multinomial CPU overhead).
4. Pre-allocated device-resident static tokens buffer (zero DtoD allocation / zero memcpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_UNIFORM_EPS = 1e-6


def gumbel_max_sample(logits: torch.Tensor, uniforms: torch.Tensor | None = None) -> torch.Tensor:
    """argmax(logits - log(-log(u))), pure tensor op, zero CPU sync."""
    if uniforms is None:
        uniforms = torch.rand_like(logits)
    u = uniforms.clamp(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS)
    return (logits.float() - torch.log(-torch.log(u))).argmax(dim=-1, keepdim=True).to(torch.int64)


def sample_token_tensor(
    logits: torch.Tensor,
    *,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    repetition_penalty: float = 10.0,
    past_tokens: torch.Tensor | None = None,
    stop_token_ids: tuple[int, ...] = (8193,),
    min_tokens: int = 0,
    current_step: int = 0,
) -> torch.Tensor:
    """Apply penalties and sample token purely on GPU (returns 1D tensor [1], zero .item())."""
    filtered_logits = logits.clone()
    # Normalize logits to 2-D (batch, vocab_size) for consistent gathering/scattering
    orig_1d = filtered_logits.ndim == 1
    if orig_1d:
        filtered_logits = filtered_logits.unsqueeze(0)
    if filtered_logits.shape[0] != 1:
        raise ValueError(
            f"sample_token_tensor expects a single-row logit tensor, got batch={filtered_logits.shape[0]}"
        )

    # Repetition penalty
    if repetition_penalty != 1.0 and past_tokens is not None and past_tokens.numel() > 0:
        unique_tokens = torch.unique(past_tokens).view(1, -1)
        token_logits = filtered_logits.gather(-1, unique_tokens)
        penalized = torch.where(
            token_logits > 0,
            token_logits / repetition_penalty,
            token_logits * repetition_penalty,
        )
        filtered_logits.scatter_(-1, unique_tokens, penalized)

    # Stop-token masking before min_tokens
    if current_step < min_tokens and stop_token_ids:
        for st_id in stop_token_ids:
            if 0 <= st_id < filtered_logits.shape[-1]:
                filtered_logits[..., st_id] = -float("inf")

    # Temperature
    if temperature > 0 and temperature != 1.0:
        filtered_logits = filtered_logits / temperature

    # Top-K Slicing: Only take Top-K (e.g. 30 elements)
    vocab_size = filtered_logits.shape[-1]
    effective_k = min(top_k, vocab_size) if top_k > 0 else vocab_size

    if effective_k < vocab_size:
        topk_logits, topk_indices = torch.topk(filtered_logits.view(1, -1), effective_k, dim=-1)
    else:
        topk_logits = filtered_logits.view(1, -1)
        topk_indices = torch.arange(vocab_size, device=filtered_logits.device).unsqueeze(0)

    # Top-P Slicing + In-tensor Gumbel selection (No .item() / No multinomial CPU dispatch)
    if 0.0 < top_p < 1.0:
        probs = F.softmax(topk_logits, dim=-1)
        sorted_probs, sorted_sub_indices = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = (cum_probs - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_logits = (sorted_probs + 1e-10).log()
        sub_sample_idx = gumbel_max_sample(sorted_logits)
        orig_topk_idx = sorted_sub_indices.gather(-1, sub_sample_idx)
        out_tokens = topk_indices.gather(-1, orig_topk_idx)
    else:
        sub_sample_idx = gumbel_max_sample(topk_logits)
        out_tokens = topk_indices.gather(-1, sub_sample_idx)

    return out_tokens.squeeze(0) if orig_1d else out_tokens


@dataclass
class IndexTTSKStepResult:
    sampled_tokens: list[int]
    sampled_tensor: torch.Tensor
    accumulated_latents: list[torch.Tensor]
    finished: bool


def talker_kstep_decode(
    talker: Any,
    initial_hidden_states: torch.Tensor,
    start_pos: int,
    k_steps: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    repetition_penalty: float = 10.0,
    stop_token_ids: tuple[int, ...] = (8193,),
    use_latent: bool = False,
    past_tokens_tensor: torch.Tensor | None = None,
    max_decode_tokens: int = 2048,
) -> IndexTTSKStepResult:
    """Run K steps of decode loop on GPU using a static ring buffer for past tokens."""
    device = initial_hidden_states.device
    sampled_tokens: list[int] = []
    accumulated_latents: list[torch.Tensor] = []
    finished = False

    # Pre-allocate device-resident buffer to avoid per-step torch.cat and DtoD copy
    static_tokens_buf = torch.empty(max_decode_tokens, dtype=torch.long, device=device)
    curr_len = 0
    write_pos = 0
    if past_tokens_tensor is not None and past_tokens_tensor.numel() > 0:
        past_num = min(past_tokens_tensor.numel(), max_decode_tokens)
        static_tokens_buf[:past_num].copy_(past_tokens_tensor.view(-1)[-past_num:])
        curr_len = past_num
        write_pos = past_num

    current_hidden = initial_hidden_states
    stop_tokens_t = torch.tensor(stop_token_ids, dtype=torch.long, device=device)

    for step_idx in range(k_steps):
        logits = talker.compute_logits(current_hidden)
        if logits is None:
            break
        logits = logits.squeeze(0).squeeze(0)  # (vocab_size,)

        active_past = static_tokens_buf[:curr_len] if curr_len > 0 else None
        token_t = sample_token_tensor(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            past_tokens=active_past,
            stop_token_ids=stop_token_ids,
            current_step=curr_len,
        )

        # In-place write to static buffer with ring index (zero DtoD allocation / zero clone)
        slot = write_pos % max_decode_tokens
        static_tokens_buf[slot : slot + 1].copy_(token_t)
        write_pos += 1
        if curr_len < max_decode_tokens:
            curr_len += 1
        sampled_tokens.append(int(token_t.item()))

        if use_latent:
            accumulated_latents.append(current_hidden.detach())

        # Check termination in tensor
        if (token_t == stop_tokens_t).any():
            finished = True
            break

        if step_idx < k_steps - 1:
            mel_pos = torch.tensor([start_pos + step_idx + 1], dtype=torch.long, device=device)
            embeds = talker.mel_embedding(token_t) + talker.mel_pos_embedding.emb(mel_pos)
            hidden = embeds
            for layer in talker.h[talker.start_layer : talker.end_layer]:
                hidden = layer(hidden)
            current_hidden = talker.final_norm(talker.ln_f(hidden))

    # If buffer wrapped around, roll so earliest token is at index 0 and latest at the end
    if write_pos > max_decode_tokens:
        sampled_tensor = static_tokens_buf.roll(- (write_pos % max_decode_tokens), dims=0)
    else:
        sampled_tensor = static_tokens_buf[:curr_len]

    return IndexTTSKStepResult(
        sampled_tokens=sampled_tokens,
        sampled_tensor=sampled_tensor,
        accumulated_latents=accumulated_latents,
        finished=finished,
    )
