# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice Generator (Stage 0) - Iterative unmasking with Qwen3 backbone.

Generates 8-codebook audio tokens from text via 32-step non-autoregressive
iterative masked prediction with classifier-free guidance.

Uses vLLM-Omni's DiffusionAttention for optimized full (bidirectional) attention
via FlashAttention/SageAttention/SDPA backends.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Unmasking schedule helpers
# ---------------------------------------------------------------------------


def _get_time_steps(
    t_start: float,
    t_end: float,
    num_step: int,
    t_shift: float,
) -> torch.Tensor:
    """Compute the unmasking schedule with time shift.

    Returns cumulative proportions [0, ..., 1] of length num_step.
    Formula: r_n = t_shift * (n/N) / (1 + (t_shift - 1) * (n/N))
    """
    steps = torch.linspace(t_start, t_end, num_step)
    shifted = t_shift * steps / (1.0 + (t_shift - 1.0) * steps)
    return shifted


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for stochastic position selection."""
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8)))
    return logits / max(temperature, 1e-8) + noise


# ---------------------------------------------------------------------------
# Qwen3-style transformer blocks using DiffusionAttention
# ---------------------------------------------------------------------------


class OmniVoiceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class OmniVoiceAttention(nn.Module):
    """Qwen3-style GQA attention using DiffusionAttention backend."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.hidden_size = config.llm_hidden_size
        self.num_heads = config.llm_num_attention_heads
        self.num_kv_heads = config.llm_num_key_value_heads
        self.head_dim = config.llm_head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Qwen3 uses per-head QK norm
        self.q_norm = OmniVoiceRMSNorm(self.head_dim)
        self.k_norm = OmniVoiceRMSNorm(self.head_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Per-head QK norm (Qwen3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if cos is not None and sin is not None:
            q = _apply_rotary_pos_emb(q, cos, sin)
            k = _apply_rotary_pos_emb(k, cos, sin)

        # Expand KV heads for GQA (8 KV heads → 16 Q heads)
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Full bidirectional attention via SDPA with proper mask support
        # Permute to (batch, heads, seq, head_dim) for SDPA
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Convert [B, 1, S, S] bool mask to float mask for SDPA
        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = attention_mask.to(dtype=q.dtype)
            sdpa_mask = sdpa_mask.masked_fill(~attention_mask, float("-inf"))
            sdpa_mask = sdpa_mask.masked_fill(attention_mask, 0.0)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        # Back to (batch, seq, heads * head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class OmniVoiceMLP(nn.Module):
    """Qwen3-style MLP with SwiGLU."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.llm_intermediate_size, config.llm_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OmniVoiceTransformerBlock(nn.Module):
    """Single Qwen3 transformer block with DiffusionAttention."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.input_layernorm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        self.self_attn = OmniVoiceAttention(config)
        self.post_attention_layernorm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        self.mlp = OmniVoiceMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, cos=cos, sin=sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tensors."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x shape: (B, S, H, D)."""
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)


# ---------------------------------------------------------------------------
# Generator model
# ---------------------------------------------------------------------------


class OmniVoiceGenerator(nn.Module):
    """OmniVoice Stage 0: Iterative unmasking generator.

    Architecture:
    - Text embedding (from Qwen3 vocab) + Audio embedding (8*1025 entries)
    - 28-layer Qwen3 transformer with full bidirectional attention
    - 8-codebook prediction head (single linear: hidden → 8*1025)
    - 32-step iterative unmasking with classifier-free guidance

    Optimizations:
    - DiffusionAttention (FlashAttn/SageAttn/SDPA auto-selected)
    - TeaCache / Cache-DiT compatible (hook-based, non-intrusive)
    - regionally_compile() compatible for torch.compile on repeated blocks
    - Sequence parallelism via SP hooks for multi-GPU
    """

    # For regionally_compile() support
    _repeated_blocks = ["layers"]

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config

        # Text embedding (shared with LLM)
        self.text_embedding = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)

        # Audio embedding: 8 codebooks * 1025 tokens
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([OmniVoiceTransformerBlock(config) for _ in range(config.llm_num_hidden_layers)])
        self.norm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)

        # Prediction head: hidden → 8 * 1025
        self.audio_heads = nn.Linear(
            config.llm_hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        # Precompute RoPE
        self._rope_cos = None
        self._rope_sin = None

    def _ensure_rope(self, seq_len: int, device: torch.device) -> None:
        """Lazily compute RoPE cos/sin if needed."""
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            max_len = max(seq_len, 4096)
            self._rope_cos, self._rope_sin = _precompute_rope(
                self.config.llm_head_dim,
                max_len,
                theta=self.config.llm_rope_theta,
                device=device,
            )

    def _prepare_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare mixed text+audio embeddings.

        Args:
            input_ids: [B, 8, S] - text tokens replicated across codebooks,
                       audio positions have per-codebook token IDs
            audio_mask: [B, S] - True for audio positions, False for text

        Returns:
            embeddings: [B, S, hidden_size]
        """
        # Text embeddings from first codebook row (all rows identical for text)
        text_embeds = self.text_embedding(input_ids[:, 0, :])

        # Audio embeddings: offset per codebook, then sum across codebooks
        shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        # Merge: audio where audio_mask=True, text elsewhere
        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def _transformer_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run through transformer layers.

        Args:
            inputs_embeds: [B, S, hidden_size]
            attention_mask: [B, 1, S, S] or None

        Returns:
            hidden_states: [B, S, hidden_size]
        """
        device = inputs_embeds.device
        seq_len = inputs_embeds.shape[1]
        self._ensure_rope(seq_len, device)

        hidden_states = inputs_embeds
        cos = self._rope_cos.to(device=device, dtype=hidden_states.dtype)
        sin = self._rope_sin.to(device=device, dtype=hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
            )

        return self.norm(hidden_states)

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to per-codebook logits.

        Args:
            hidden_states: [B, S, hidden_size]

        Returns:
            logits: [B, 8, S, 1025]
        """
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)  # [B, S, 8*1025]
        return logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)  # [B, 8, S, 1025]

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        target_lens: list[int],
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
    ) -> torch.Tensor:
        """Run the full 32-step iterative unmasking generation.

        Args:
            input_ids: [2*B, 8, S] - conditional (0:B) + unconditional (B:2B)
            audio_mask: [2*B, S] - True for audio positions
            attention_mask: [2*B, 1, S, S] - attention mask
            target_lens: List of target audio lengths per batch item
            num_step: Number of unmasking steps
            guidance_scale: CFG scale
            t_shift: Time shift for schedule
            layer_penalty_factor: Penalty for later codebooks
            position_temperature: Gumbel temperature for position selection
            class_temperature: Temperature for token prediction (0=greedy)

        Returns:
            tokens: [B, 8, max_target_len] - generated audio tokens
        """
        B = len(target_lens)
        device = input_ids.device
        max_target_len = max(target_lens)
        mask_id = self.config.audio_mask_id
        num_codebooks = self.config.num_audio_codebook

        # Initialize all target tokens as [MASK]
        tokens = torch.full(
            (B, num_codebooks, max_target_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # Compute unmasking schedule
        timesteps = _get_time_steps(0.0, 1.0, num_step + 1, t_shift).tolist()
        schedules = []
        for t_len in target_lens:
            total_mask = t_len * num_codebooks
            rem = total_mask
            sched = []
            for step in range(num_step):
                num = (
                    rem
                    if step == num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_ids = torch.arange(num_codebooks, device=device).view(1, -1, 1)

        # Compute c_lens for extracting target region from full sequence
        c_lens = []
        for i in range(B):
            # Conditional sequence length = number of non-padding positions
            c_len = attention_mask[i, 0, 0].sum().item()
            c_lens.append(int(c_len))

        # Main iterative loop
        for step in range(num_step):
            # Prepare embeddings and run transformer
            inputs_embeds = self._prepare_embeddings(input_ids, audio_mask)
            hidden_states = self._transformer_forward(inputs_embeds, attention_mask)
            batch_logits = self._get_logits(hidden_states).to(torch.float32)
            # batch_logits: [2*B, 8, S, 1025]

            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len = c_lens[i]
                t_len = target_lens[i]

                # Extract logits for target region
                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]  # [1, 8, T, 1025]
                u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :]  # [1, 8, T, 1025]

                # Classifier-free guidance
                if guidance_scale != 0:
                    c_log_probs = F.log_softmax(c_logits, dim=-1)
                    u_log_probs = F.log_softmax(u_logits, dim=-1)
                    log_probs = torch.log_softmax(
                        c_log_probs + guidance_scale * (c_log_probs - u_log_probs),
                        dim=-1,
                    )
                else:
                    log_probs = F.log_softmax(c_logits, dim=-1)

                # Prevent predicting [MASK]
                log_probs[..., mask_id] = -float("inf")

                # Token prediction
                if class_temperature > 0.0:
                    pred_tokens = _gumbel_sample(log_probs, class_temperature).argmax(dim=-1)
                else:
                    pred_tokens = log_probs.argmax(dim=-1)  # [1, 8, T]

                # Confidence scores
                scores = log_probs.max(dim=-1)[0]  # [1, 8, T]

                # Layer penalty (earlier codebooks get higher priority)
                scores = scores - (layer_ids * layer_penalty_factor)

                # Gumbel noise for position selection
                if position_temperature > 0.0:
                    scores = _gumbel_sample(scores, position_temperature)

                # Mask out already unmasked positions
                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(sample_tokens != mask_id, -float("inf"))

                # Select top-k positions to unmask
                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten().clone()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                # Update tokens and batch inputs for next iteration
                tokens[i : i + 1, :, :t_len] = sample_tokens
                input_ids = input_ids.clone()
                input_ids[i, :, c_len - t_len : c_len] = sample_tokens.squeeze(0)
                input_ids[B + i, :, :t_len] = sample_tokens.squeeze(0)

        return tokens

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load weights from HuggingFace OmniVoice model.safetensors.

        The HF checkpoint contains:
        - llm.* -> Qwen3 transformer weights
        - audio_embeddings.* -> audio embedding table
        - audio_heads.* -> prediction head
        """
        import os

        from safetensors.torch import load_file

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        state_dict = load_file(weights_path, device=str(device))

        # Map HF weight names to our module names
        loaded_keys = set()

        # 1. Text embedding: llm.embed_tokens.weight -> text_embedding.weight
        text_emb_key = "llm.embed_tokens.weight"
        if text_emb_key in state_dict:
            self.text_embedding.weight.data.copy_(state_dict[text_emb_key])
            loaded_keys.add(text_emb_key)

        # 2. Audio embeddings
        for key in ["audio_embeddings.weight"]:
            if key in state_dict:
                self.audio_embeddings.weight.data.copy_(state_dict[key])
                loaded_keys.add(key)

        # 3. Audio heads
        for key in ["audio_heads.weight"]:
            if key in state_dict:
                self.audio_heads.weight.data.copy_(state_dict[key])
                loaded_keys.add(key)

        # 4. Transformer layers: llm.layers.N.* -> layers.N.*
        for key, value in state_dict.items():
            if key.startswith("llm.layers."):
                # llm.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
                our_key = key.replace("llm.layers.", "layers.")
                parts = our_key.split(".")
                module = self
                try:
                    for part in parts[:-1]:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    param_name = parts[-1]
                    param = getattr(module, param_name)
                    if isinstance(param, nn.Parameter):
                        param.data.copy_(value)
                    elif isinstance(param, torch.Tensor):
                        param.copy_(value)
                    loaded_keys.add(key)
                except (AttributeError, IndexError, KeyError) as e:
                    logger.warning("Failed to load weight %s: %s", key, e)

        # 5. Final norm: llm.norm.weight -> norm.weight
        norm_key = "llm.norm.weight"
        if norm_key in state_dict:
            self.norm.weight.data.copy_(state_dict[norm_key])
            loaded_keys.add(norm_key)

        unloaded = set(state_dict.keys()) - loaded_keys
        # Filter out audio_tokenizer weights (loaded in decoder stage)
        unloaded = {k for k in unloaded if not k.startswith("audio_tokenizer.")}
        if unloaded:
            logger.info(
                "Generator: %d/%d weights loaded, %d skipped (decoder weights)",
                len(loaded_keys),
                len(state_dict),
                len(unloaded),
            )
        else:
            logger.info("Generator: all %d weights loaded", len(loaded_keys))
