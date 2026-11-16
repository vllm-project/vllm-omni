# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only π0 (Pi-Zero) VLA math kernel for vllm-omni.

A self-contained inference kernel: only the math that turns a robot observation
into an action chunk, with no serving/request glue — validated bit-for-bit
against the LeRobot ``PI0Policy`` reference (``max|Δ| < 1e-4``).

π0 = PaliGemma (SigLIP vision + Gemma 2B LM) prefix + Gemma 300M action expert
suffix + flow-matching head. Inference:

1. Embed prefix (images + language) → bidirectional prefix tokens.
2. Forward prefix through PaliGemma → a per-layer ``list[(k, v)]`` KV cache.
3. For each Euler step ``t = 1.0, 1-dt, ..., 0``: embed the suffix
   (state + noisy actions + timestep), run the action expert with cross
   attention over the cached prefix K/V, predict velocity ``v_t``, and
   integrate ``x_t = x_t + dt * v_t``.
4. Return ``x_0`` as the action chunk ``(batch, action_horizon, action_dim)``.

We walk Gemma decoder layers ourselves in ``_compute_layer_*`` so we never go
through ``GemmaModel.forward`` (which has version-dependent mask/KV-cache
behaviour). Targets modern transformers (PaliGemmaForConditionalGeneration with
an inner PaliGemmaModel + DynamicCache).

Reference implementations:
   - OpenPI: openpi/src/openpi/models_pytorch/pi0_pytorch.py, gemma_pytorch.py
   - LeRobot: lerobot/src/lerobot/policies/pi0/modeling_pi0.py
Weight source: https://huggingface.co/lerobot/pi0_base
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM,
    apply_rotary_pos_emb,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_MAX_TOKEN_LEN = 48
DEFAULT_NUM_INFERENCE_STEPS = 10
DEFAULT_IMAGE_RESOLUTION = (224, 224)

# Large negative value to fill masked-out positions in a float attention mask.
# Matches OpenPI's constant exactly so that numerics line up during parity.
# Ref: openpi/src/openpi/models/gemma.py
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# ──────────────────────────────────────────────────────────────────────
# Gemma variant configs (matches openpi/models/gemma.py get_config)
# ──────────────────────────────────────────────────────────────────────


class GemmaVariantConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    if variant == "gemma_2b":
        return GemmaVariantConfig(2048, 18, 16384, 8, 1, 256)
    elif variant == "gemma_300m":
        return GemmaVariantConfig(1024, 18, 4096, 8, 1, 256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ──────────────────────────────────────────────────────────────────────
# Utility functions (match openpi/models_pytorch/pi0_pytorch.py)
# ──────────────────────────────────────────────────────────────────────


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute a sine/cosine positional embedding for scalar timesteps.

    Ref: openpi/models_pytorch/pi0_pytorch.py create_sinusoidal_pos_embedding
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be 1-D (batch_size,)")
    if device is None:
        device = time.device

    # Use float64 for the log-linear sweep and for the inner products, to
    # match the reference implementation's numerical behaviour exactly.
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Build a 2D attention mask from a padding mask and an autoregressive mask.

    Ref: openpi/models_pytorch/pi0_pytorch.py make_att_2d_masks
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2-D, got {att_masks.ndim}-D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2-D, got {pad_masks.ndim}-D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _build_norm_buffers(norm_stats: dict | None, key: str) -> dict[str, torch.Tensor] | None:
    """Parse a ``norm_stats[key]`` entry into CPU tensors, or ``None``.

    Matches LeRobot's ``NormalizationMode``:
      - ``mean_std`` : ``forward = (x - mean) / std``, ``inverse = x * std + mean``
      - ``min_max``  : forward maps into ``[-1, 1]``, inverse maps back.
    """
    if not norm_stats or not isinstance(norm_stats, dict):
        return None
    entry = norm_stats.get(key)
    if not entry:
        return None
    mode = str(entry.get("mode", "mean_std")).lower()
    if mode == "mean_std":
        mean = entry.get("mean")
        std = entry.get("std")
        if mean is None or std is None:
            return None
        return {
            "mode": mode,
            "mean": torch.as_tensor(mean, dtype=torch.float32),
            "std": torch.as_tensor(std, dtype=torch.float32),
        }
    if mode == "min_max":
        lo = entry.get("min")
        hi = entry.get("max")
        if lo is None or hi is None:
            return None
        return {
            "mode": mode,
            "min": torch.as_tensor(lo, dtype=torch.float32),
            "max": torch.as_tensor(hi, dtype=torch.float32),
        }
    return None


def _apply_norm(
    x: torch.Tensor,
    stats: dict[str, torch.Tensor] | None,
    inverse: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply (un)normalization using the given stats. No-op if ``stats`` is None.

    Broadcasts over leading dims; stats vectors align with the last dim. When
    ``x`` has more last-dim entries than the stats (π0 pads state/action to
    ``max_dim``), only the first ``len(stats)`` entries are transformed and
    the padded tail is left untouched.
    """
    if stats is None:
        return x
    mode = stats["mode"]
    if mode == "mean_std":
        mean = stats["mean"].to(device=x.device, dtype=x.dtype)
        std = stats["std"].to(device=x.device, dtype=x.dtype)
        valid = mean.shape[0]
        head = x[..., :valid]
        if inverse:
            head = head * std + mean
        else:
            head = (head - mean) / (std + eps)
        if valid == x.shape[-1]:
            return head
        return torch.cat([head, x[..., valid:]], dim=-1)
    if mode == "min_max":
        lo = stats["min"].to(device=x.device, dtype=x.dtype)
        hi = stats["max"].to(device=x.device, dtype=x.dtype)
        valid = lo.shape[0]
        head = x[..., :valid]
        denom = (hi - lo).clamp_min(eps)
        if inverse:
            head = (head + 1.0) * 0.5 * denom + lo
        else:
            head = 2.0 * (head - lo) / denom - 1.0
        if valid == x.shape[-1]:
            return head
        return torch.cat([head, x[..., valid:]], dim=-1)
    return x


def prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, S, S)`` bool masks to ``(B, 1, S, S)`` float masks.

    ``True`` → 0.0 (attend), ``False`` → ``OPENPI_ATTENTION_MASK_VALUE``.
    Ref: openpi PI0Pytorch._prepare_attention_masks_4d
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


# ──────────────────────────────────────────────────────────────────────
# Dual-backbone: PaliGemma + Action Expert
# Ref: openpi/models_pytorch/gemma_pytorch.py PaliGemmaWithExpertModel
# Ref: lerobot/policies/pi0/modeling_pi0.py PaliGemmaWithExpertModel
# ──────────────────────────────────────────────────────────────────────


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    ``(B, num_kv_heads, S, D) → (B, num_kv_heads * n_rep, S, D)``.
    Matches ``transformers.models.gemma.modeling_gemma.repeat_kv``.
    """
    if n_rep == 1:
        return hidden_states
    b, nh, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nh, n_rep, s, d)
    return hidden_states.reshape(b, nh * n_rep, s, d)


def _attend(query_states, key_states, value_states, attention_mask, num_kv_groups, scaling):
    """Manual eager attention: ``softmax(Q Kᵀ · scale + mask) · V``.

    Important: we slice ``attention_mask[..., :key_states.shape[-2]]`` so that
    the same ``(B, 1, Q, prefix+suffix)`` mask produced by ``denoise_step``
    can be re-used both in the prefix pass (K length = prefix) and in the
    suffix pass (K length = prefix + suffix after we concat the cached prefix).
    Stock transformers' ``eager_attention_forward`` adds the mask without
    slicing and therefore breaks when the two dims disagree.
    """
    k = _repeat_kv(key_states, num_kv_groups)
    v = _repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : k.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, v)


def _compute_layer_prefix_only(layer_idx, hidden_states, attention_mask, position_ids, paligemma):
    """Run one PaliGemma LM layer on the prefix only, returning the layer
    output **and** the post-RoPE ``(k, v)`` so the caller can cache them for
    the suffix pass."""
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]
    residual = hidden_states
    x = layer.input_layernorm(hidden_states)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(x).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(x).view(hidden_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(x).view(hidden_shape).transpose(1, 2)

    cos, sin = model.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    att = _attend(
        q,
        k,
        v,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    out = layer.self_attn.o_proj(att) + residual
    after_resid = out
    out = layer.mlp(layer.post_attention_layernorm(out)) + after_resid
    return out, (k, v)


def _compute_layer_suffix_only(
    layer_idx,
    hidden_states,
    prefix_kv,
    attention_mask,
    position_ids,
    paligemma,
    gemma_expert,
):
    """Run one action-expert layer on the suffix, attending to the cached
    prefix K/V concatenated with freshly computed suffix K/V.

    ``prefix_kv`` is a ``(k_prefix, v_prefix)`` tuple, shape
    ``(B, num_kv_heads, prefix_len, head_dim)``, post-RoPE, produced by the
    corresponding ``_compute_layer_prefix_only`` call. Bypasses
    ``GemmaModel.forward`` entirely — no HF mask rebuild, no cache format
    gymnastics.
    """
    layer = gemma_expert.model.layers[layer_idx]
    residual = hidden_states
    x = layer.input_layernorm(hidden_states)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(x).view(hidden_shape).transpose(1, 2)
    k_suf = layer.self_attn.k_proj(x).view(hidden_shape).transpose(1, 2)
    v_suf = layer.self_attn.v_proj(x).view(hidden_shape).transpose(1, 2)

    # RoPE frequencies are shared between PaliGemma and the expert.
    cos, sin = gemma_expert.model.rotary_emb(v_suf, position_ids)
    q, k_suf = apply_rotary_pos_emb(q, k_suf, cos, sin, unsqueeze_dim=1)

    # Concatenate cached prefix K/V (possibly different dtype) with suffix K/V.
    k_prefix, v_prefix = prefix_kv
    k = torch.cat([k_prefix.to(k_suf.dtype), k_suf], dim=2)
    v = torch.cat([v_prefix.to(v_suf.dtype), v_suf], dim=2)

    att = _attend(
        q,
        k,
        v,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    out = layer.self_attn.o_proj(att) + residual
    after_resid = out
    out = layer.mlp(layer.post_attention_layernorm(out)) + after_resid
    return out


class PaliGemmaWithActionExpert(nn.Module):
    """Dual-backbone transformer: PaliGemma (Gemma 2B) + Action Expert (Gemma 300M).

    ``forward`` has two inference modes:
      - **prefix_only**: ``inputs_embeds=[prefix, None]`` + ``use_cache=True``
        → compute prefix hidden states + a layer-wise K/V cache.
      - **suffix_only**: ``inputs_embeds=[None, suffix]`` + the cache from the
        previous prefix pass → compute expert hidden states with cross-attention
        over the concatenated (prefix, suffix) K/V.

    Both modes walk Gemma decoder layers manually through ``_compute_layer_*``
    instead of ``GemmaModel.forward`` — that way we own the attention mask
    handling and the KV cache format (a plain ``list[(k, v)]`` one entry
    per layer).

    Ref: lerobot/policies/pi0/modeling_pi0.py PaliGemmaWithExpertModel
    """

    def __init__(self, vlm_config, action_expert_config):
        super().__init__()

        # Build HF PaliGemma config from the variant dims we expose.
        # NOTE: we do NOT set use_adarms / adarms_cond_dim — those are
        # OpenPI-only attributes used by π0.5, not π0.
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        # transformers ≥ 5 uses ``dtype``; older versions still accept
        # ``torch_dtype``. We set the canonical name; HF's PretrainedConfig
        # normalizes one to the other internally.
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        # The action expert doesn't embed tokens — it only consumes the
        # suffix state/action embeddings we feed in.
        self.gemma_expert.model.embed_tokens = None

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with SigLIP vision tower + PaliGemma projector.

        We run the two steps explicitly instead of calling
        ``PaliGemmaModel.get_image_features`` because that convenience
        method historically has returned either the projected tensor (old)
        or the raw vision-tower output (new). Being explicit makes us
        independent of which transformers release we're on.
        """
        # Shapes: pixel_values (B, 3, 224, 224) → SigLIP (B, 256, 1152)
        #                                      → projector (B, 256, 2048)
        vision_outputs = self.paligemma.model.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state
        return self.paligemma.model.multi_modal_projector(image_features)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens with PaliGemma's embedding table, returning the
        ``* sqrt(hidden)``-scaled embedding (PaliGemma/Gemma convention).

        The scaling location moved across transformers releases:
          * ≤ 5.3: ``embed_tokens`` is a plain ``nn.Embedding`` and the
            ``* sqrt(hidden)`` normalizer lives inside ``GemmaModel.forward``
            (which we bypass) — so we apply it explicitly here.
          * ≥ 5.4: ``embed_tokens`` is a ``GemmaTextScaledWordEmbedding`` that
            self-applies ``embed_scale = hidden_size ** 0.5`` — applying it again
            would double-scale (≈45×). We detect this and skip the manual scale.
        This makes ``embed_language_tokens`` return the canonical scaled embedding
        on every transformers version.
        """
        embed_tokens = self.paligemma.model.language_model.embed_tokens
        lang_emb = embed_tokens(tokens)
        # If the embedding already self-scales (transformers ≥ 5.4), don't repeat it.
        if getattr(embed_tokens, "embed_scale", None) is None:
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        return lang_emb

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
    ):
        """Dispatch to prefix_only / suffix_only and return
        ``([prefix_out, suffix_out], past_key_values_or_None)``.
        """
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        pali_lm = self.paligemma.model.language_model
        expert_lm = self.gemma_expert.model

        if inputs_embeds[1] is None:
            # Prefix-only: PaliGemma LM on (images + language) tokens; the
            # per-layer post-RoPE K/V is collected into a list that the
            # suffix pass will consume directly.
            hidden_states = inputs_embeds[0]
            kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(num_layers):
                hidden_states, kv = _compute_layer_prefix_only(
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    paligemma=self.paligemma,
                )
                kv_list.append(kv)
            hidden_states = pali_lm.norm(hidden_states)
            return [hidden_states, None], (kv_list if use_cache else None)

        if inputs_embeds[0] is not None:
            raise ValueError(
                "PaliGemmaWithActionExpert.forward only supports prefix-only "
                "or suffix-only dispatch; got both inputs_embeds populated."
            )
        # Suffix-only: action expert on (state + noisy_actions + time),
        # attending to the cached prefix K/V.
        if not isinstance(past_key_values, list):
            raise TypeError(
                "suffix_only forward expects past_key_values to be the "
                "list[(k, v)] produced by a previous prefix_only forward; "
                f"got {type(past_key_values)}"
            )
        hidden_states = inputs_embeds[1]
        for layer_idx in range(num_layers):
            hidden_states = _compute_layer_suffix_only(
                layer_idx,
                hidden_states,
                past_key_values[layer_idx],
                attention_mask,
                position_ids,
                paligemma=self.paligemma,
                gemma_expert=self.gemma_expert,
            )
        hidden_states = expert_lm.norm(hidden_states)
        return [None, hidden_states], None


# ──────────────────────────────────────────────────────────────────────
# Main π0 Model
# ──────────────────────────────────────────────────────────────────────


class Pi0ForActionPrediction(nn.Module):
    """π0 VLA model for robot action prediction via flow matching.

    Inference flow:
      1. Embed prefix (images + language) → prefix tokens.
      2. Forward prefix through PaliGemma → layer-wise KV cache.
      3. For each denoising step ``t = 1.0, 1-dt, ..., 0``:
         a. Embed suffix (state + x_t + timestep) → suffix tokens.
         b. Forward suffix through the action expert with the prefix cache.
         c. ``x_t = x_t + dt * v_t`` (Euler integration).
      4. Return ``x_0`` as the predicted action chunk.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        # ``quant_config`` is accepted for interface compatibility but unused —
        # π0 runs in full precision for flow-matching parity.
        del quant_config
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
        # ``max_state_dim`` is independent from ``max_action_dim`` — e.g. Aloha
        # has state_dim=14, action_dim=32, both padded to their own max.  Fall
        # back to action_dim for older configs that only expose a single size.
        self.max_state_dim = getattr(config, "max_state_dim", self.action_dim)
        self.action_horizon = getattr(config, "chunk_size", DEFAULT_ACTION_HORIZON)
        self.num_inference_steps = getattr(config, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)

        paligemma_variant = getattr(config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(config, "action_expert_variant", "gemma_300m")
        vlm_config = get_gemma_config(paligemma_variant)
        expert_config = get_gemma_config(action_expert_variant)
        self.vlm_width = vlm_config.width
        self.expert_width = expert_config.width

        # Dual backbone
        self.paligemma_with_expert = PaliGemmaWithActionExpert(vlm_config, expert_config)

        # Action chunk projections (openpi pi0_pytorch.py).
        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)

        # State is projected to the expert dim once (π0 only, not π0.5).
        # Ref: lerobot modeling_pi0.py ``nn.Linear(config.max_state_dim, ...)``
        self.state_proj = nn.Linear(self.max_state_dim, self.expert_width)

        # Timestep + action fusion MLP (2W → W → W).
        self.action_time_mlp_in = nn.Linear(2 * self.expert_width, self.expert_width)
        self.action_time_mlp_out = nn.Linear(self.expert_width, self.expert_width)

        # Optional mean/std normalization stats (LeRobot NormalizerProcessorStep
        # semantics). Format on the config (all optional):
        #     norm_stats = {
        #         "state":  {"mode": "mean_std", "mean": [...], "std": [...]},
        #         "action": {"mode": "mean_std", "mean": [...], "std": [...]},
        #     }
        # ``mode`` may also be ``"min_max"``.
        #
        # Missing norm_stats is logged at INFO (not WARNING) because the most
        # common setup is "client pre-normalizes state and post-normalizes
        # actions using dataset stats" — that's a supported mode, not an error.
        self._state_norm = _build_norm_buffers(getattr(config, "norm_stats", None), "state")
        self._action_norm = _build_norm_buffers(getattr(config, "norm_stats", None), "action")
        if self._state_norm is None:
            logger.info(
                "π0: no state normalization stats on config.norm_stats — state values will pass through unchanged."
            )
        if self._action_norm is None:
            logger.info(
                "π0: no action normalization stats on config.norm_stats — "
                "returned actions are in the model's normalized space."
            )

    # ── State / action normalization ─────────────────────────────────

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return _apply_norm(state, self._state_norm, inverse=False)

    def _unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return _apply_norm(actions, self._action_norm, inverse=True)

    # ── Prefix embedding ─────────────────────────────────────────────

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the prefix embeddings, per-token padding mask, and AR mask.

        Prefix tokens form a contiguous sequence
        ``[img_cam_0..., img_cam_1..., ..., lang_tokens...]`` with
        bidirectional attention; the returned ``att_masks`` are all zeros
        because each token is free to attend to every other prefix token (the
        suffix pass will put a causal boundary right before the state token).

        Ref: openpi PI0Pytorch.embed_prefix
        """
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, image_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # ``embed_language_tokens`` already returns the ``* sqrt(hidden)``-scaled
        # embedding (version-robust: self-scaled at transformers ≥ 5.4, manually
        # scaled at ≤ 5.3). The image slice went through the projector and is
        # NOT scaled. See embed_language_tokens for the version details.
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)

        return embs, pad_masks, att_masks

    # ── Suffix embedding ─────────────────────────────────────────────

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the suffix embeddings + masks: ``[state_token, action_tokens×H]``.

        AR mask layout: ``[1, 1, 0, 0, ..., 0]`` — the state token is a causal
        boundary (no later token attends backwards through it onto prefix
        *by mistake*), the first action token starts a new causal block, and
        the rest of the action tokens attend to each other bidirectionally.

        Ref: openpi PI0Pytorch.embed_suffix
        """
        model_dtype = self.state_proj.weight.dtype
        state = state.to(dtype=model_dtype)
        noisy_actions = noisy_actions.to(dtype=model_dtype)
        device = state.device

        # State → (B, 1, W)
        state_emb = self.state_proj(state)[:, None, :]
        bsize = state_emb.shape[0]

        # Sinusoidal timestep → (B, W) → expand across the horizon.
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        ).to(dtype=model_dtype)

        # Fuse action and time via (2W → W → W) MLP with SiLU.
        action_emb = self.action_in_proj(noisy_actions)  # (B, H, W)
        action_time_emb = torch.cat([action_emb, time_emb[:, None, :].expand_as(action_emb)], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs = torch.cat([state_emb, action_time_emb], dim=1)
        pad_masks = torch.ones(bsize, embs.shape[1], dtype=torch.bool, device=device)
        att_masks = torch.tensor(
            [1] + [1] + [0] * (self.action_horizon - 1),
            dtype=embs.dtype,
            device=device,
        )[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    # ── Denoising step ───────────────────────────────────────────────

    def denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one flow-matching denoising step: predict ``v_t`` from ``x_t``.

        Uses the prefix KV cache from ``sample_actions`` and only runs the
        action expert. Ref: openpi PI0Pytorch.denoise_step.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        # Build the full (B, suffix_len, prefix_len + suffix_len) boolean mask:
        #   * suffix queries can see every *valid* prefix key (padded camera
        #     slots and padded language tokens are blocked).
        #   * within the suffix, the state token is a causal boundary; action
        #     tokens attend to each other bidirectionally.
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs continue from where the prefix's last valid token left off.
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
        )

        # Drop the state token, keep only the action tokens, project down.
        suffix_out = outputs_embeds[1][:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    # ── Full action generation ───────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk via iterative flow-matching denoising.

        Convention: ``t=1`` is noise, ``t=0`` is the target — opposite of the
        published π0 paper but matches both OpenPI and LeRobot.
        Ref: openpi PI0Pytorch.sample_actions
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = torch.randn(
                bsize,
                self.action_horizon,
                self.action_dim,
                dtype=torch.float32,
                device=device,
            )

        # 1. Prefix embeddings + mask building.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)

        # 2. Forward prefix through PaliGemma LM, producing a list[(k, v)] cache.
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 3. Euler-integrated denoising from t=1 down to t=0.
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    # ── Weight loading ───────────────────────────────────────────────

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from an ``lerobot/pi0_base`` safetensors checkpoint.

        Checkpoint keys (after stripping the leading ``model.`` prefix) look
        like ``paligemma_with_expert.paligemma.<sub>.*``. Modern transformers
        nests those sub-modules one level deeper (``paligemma.model.<sub>``),
        and PaliGemma ties ``lm_head.weight`` with ``embed_tokens.weight`` at
        ``post_init`` time. Two remap rules make the load lossless:

          1. ``paligemma.{vision_tower,multi_modal_projector,language_model}.*``
             → ``paligemma.model.{...}.*``
          2. ``paligemma.lm_head.weight``
             → ``paligemma.model.language_model.embed_tokens.weight``
             (the checkpoint does not store ``embed_tokens.weight`` on its
             own — only the tied ``lm_head.weight`` copy — and
             ``PaliGemmaForConditionalGeneration`` does not register
             ``lm_head.weight`` as a Parameter when tied, so without this
             rule the language embedding would silently remain at its
             random init.)

        On any remaining mismatch the loader logs a warning listing a sample
        of the unmatched checkpoint keys and model params; downstream parity
        tests surface these as a hard failure.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        _PALIGEMMA_SUBMODULES = (
            "vision_tower",
            "multi_modal_projector",
            "language_model",
        )

        def _remap(name: str) -> str:
            # Strip the leading "model." that LeRobot's PI0Policy wrapper adds.
            if name.startswith("model."):
                name = name[len("model.") :]

            # Historical MLP alias used by some early π0 checkpoints.
            if name.startswith("time_mlp_in."):
                name = "action_time_mlp_in." + name[len("time_mlp_in.") :]
            elif name.startswith("time_mlp_out."):
                name = "action_time_mlp_out." + name[len("time_mlp_out.") :]

            # Nested PaliGemma layout.
            for sub in _PALIGEMMA_SUBMODULES:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat) :]

            # Tied lm_head → embed_tokens redirect (see docstring).
            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

            return name

        def _fix_vision_tower(name: str) -> str:
            """Reconcile the SigLIP vision-tower nesting across transformers
            versions. transformers ≤5.3 wraps the encoder in an extra
            ``vision_tower.vision_model.*`` module; ≥5.4 flattens it to
            ``vision_tower.*``. The ``lerobot/pi0_base`` checkpoint uses the
            ``vision_model`` layout, so on newer transformers we must drop that
            infix (else 437 vision params silently load nothing). Adaptive:
            only rewrite when the rewritten key actually exists on the model.
            """
            vt = "paligemma_with_expert.paligemma.model.vision_tower."
            if not name.startswith(vt):
                return name
            rest = name[len(vt) :]
            if name in params_dict or name in buffers_dict:
                return name  # model already expects this exact layout
            if rest.startswith("vision_model."):
                # Checkpoint has the infix; model (≥5.4) doesn't → strip it.
                candidate = vt + rest[len("vision_model.") :]
            else:
                # Checkpoint lacks the infix; model (≤5.3) has it → insert it.
                candidate = vt + "vision_model." + rest
            return candidate if (candidate in params_dict or candidate in buffers_dict) else name

        loaded = 0
        skipped: list[str] = []
        filled_params: set = set()
        for name, loaded_weight in weights:
            mapped = _fix_vision_tower(_remap(name))
            if mapped in params_dict:
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            elif mapped in buffers_dict:
                buffers_dict[mapped].copy_(loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            else:
                skipped.append(mapped)

        # Reverse audit: any model param that got no checkpoint tensor at all
        # would be running with random init. ``rotary_emb.inv_freq`` and
        # friends are config-derived buffers, not stored in the checkpoint.
        missing_params: list[str] = []
        for pname in params_dict:
            if pname in filled_params:
                continue
            if "rotary_emb" in pname or pname.endswith(".inv_freq"):
                continue
            missing_params.append(pname)

        if missing_params or skipped:
            parts: list[str] = []
            if skipped:
                parts.append(f"{len(skipped)} checkpoint key(s) had no home in the model (first 5: {skipped[:5]})")
            if missing_params:
                parts.append(f"{len(missing_params)} model param(s) received NO weight (first 5: {missing_params[:5]})")
            logger.warning(
                "π0 load_weights: %d tensors loaded — %s.",
                loaded,
                "; ".join(parts),
            )
        else:
            logger.info(
                "π0 load_weights: %d tensors loaded, 0 skipped, 0 missing.",
                loaded,
            )
        return filled_params
