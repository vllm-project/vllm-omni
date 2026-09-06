# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only π0.5 VLA math kernel for vllm-omni.

Only the math that turns a robot observation into an action chunk; no serving or
request glue. Deliberately shaped like ``models/pi0/modeling_pi0.py`` so the two
can later be factored into a shared Pi-family module (RFC step 2) on a
"behaviour unchanged" review.

π0.5 = PaliGemma (SigLIP vision + Gemma 2B LM) prefix + Gemma 300M action expert
suffix + flow-matching head. Where π0 projects the robot state through
``state_proj``, π0.5 discretizes it into prompt tokens — so there is no
``state_proj`` layer, ``sample_actions`` takes no ``state`` argument, and the
suffix is action tokens only, which drops π0's leading state-token boundary from
the suffix attention mask and leaves ``[1] + [0] * (horizon - 1)``.

π0.5 is nonetheless the *larger* model: the 37 AdaRMS ``dense`` projections add
~116M parameters against the ~8K that ``state_proj`` saves. (LeRobot's README
says otherwise; the checkpoint disagrees.)

Reference implementations:
   - OpenPI: openpi/src/openpi/models_pytorch/pi0_pytorch.py, gemma_pytorch.py
   - LeRobot: lerobot/src/lerobot/policies/pi05/modeling_pi05.py
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
DEFAULT_MAX_TOKEN_LEN = 200  # π0 uses 48
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


def prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, S, S)`` bool masks to ``(B, 1, S, S)`` float masks.

    ``True`` → 0.0 (attend), ``False`` → ``OPENPI_ATTENTION_MASK_VALUE``.
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


def _build_norm_buffers(norm_stats: dict | None, key: str) -> dict[str, torch.Tensor] | None:
    """Parse a ``norm_stats[key]`` entry into CPU tensors, or ``None``.

    ``quantile`` is the one to get right: LeRobot defaults π0.5 to QUANTILES
    where π0 uses MEAN_STD, so a π0.5 checkpoint usually ships ``q01``/``q99``.
    Returning ``None`` for it would leave actions in normalized space, which
    reads as a badly tuned policy rather than as a failure.
    """
    if not norm_stats or not isinstance(norm_stats, dict):
        return None
    entry = norm_stats.get(key)
    if not entry:
        return None

    mode = entry.get("mode")
    if mode is None:
        if "mean" in entry and "std" in entry:
            mode = "mean_std"
        elif "min" in entry and "max" in entry:
            mode = "min_max"
        elif ("q01" in entry and "q99" in entry) or ("low" in entry and "high" in entry):
            mode = "quantile"
    mode = str(mode).lower() if mode is not None else None

    if mode == "mean_std":
        mean, std = entry.get("mean"), entry.get("std")
        if mean is None or std is None:
            return None
        return {
            "mode": mode,
            "mean": torch.as_tensor(mean, dtype=torch.float32),
            "std": torch.as_tensor(std, dtype=torch.float32),
        }
    if mode in ("min_max", "quantile"):
        if mode == "min_max":
            lo, hi = entry.get("min"), entry.get("max")
        else:
            lo = entry.get("q01", entry.get("low"))
            hi = entry.get("q99", entry.get("high"))
        if lo is None or hi is None:
            return None
        return {
            "mode": "min_max",  # same arithmetic; quantile only changes the bounds
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
    ``x`` has more last-dim entries than the stats (π0.5 pads actions to
    ``max_action_dim``), only the first ``len(stats)`` entries are transformed
    and the padded tail is left untouched.
    """
    if stats is None:
        return x
    mode = stats["mode"]
    if mode == "mean_std":
        mean = stats["mean"].to(device=x.device, dtype=x.dtype)
        std = stats["std"].to(device=x.device, dtype=x.dtype)
        valid = mean.shape[0]
        head = x[..., :valid]
        head = head * std + mean if inverse else (head - mean) / (std + eps)
        if valid == x.shape[-1]:
            return head
        return torch.cat([head, x[..., valid:]], dim=-1)
    if mode == "min_max":
        lo = stats["min"].to(device=x.device, dtype=x.dtype)
        hi = stats["max"].to(device=x.device, dtype=x.dtype)
        valid = lo.shape[0]
        head = x[..., :valid]
        denom = (hi - lo).clamp_min(eps)
        head = (head + 1.0) * 0.5 * denom + lo if inverse else 2.0 * (head - lo) / denom - 1.0
        if valid == x.shape[-1]:
            return head
        return torch.cat([head, x[..., valid:]], dim=-1)
    return x


# ──────────────────────────────────────────────────────────────────────
# AdaRMS — the π0.5-specific norm in the action expert
# ──────────────────────────────────────────────────────────────────────
class Pi05AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm conditioned on the flow-matching timestep.

    π0 conditions on time by *concatenating* a time embedding onto each action
    embedding. π0.5 instead feeds the time embedding into every action-expert
    norm, which produces a per-layer ``(scale, shift, gate)`` triple::

        y    = norm(x) * (1 + scale) + shift
        out  = residual + gate * sublayer(y)

    ``dense`` is zero-initialized, so an untrained model starts as the identity
    modulation with a closed gate — matching OpenPI's parameterization.

    Note the shape of the unconditioned branch: ``normed * (1 + weight)`` with
    ``weight`` zero-initialized, which is exactly ``transformers``'
    ``GemmaRMSNorm``. That equivalence is why only the *expert* norms need
    replacing here and the PaliGemma prefix can keep stock Gemma layers.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(var + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(normed, gate)``; ``gate`` is ``None`` in the unconditioned case."""
        dtype = x.dtype
        normed = self._norm(x)
        if self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.to(dtype), None

        if cond is None:
            # A conditioned norm silently falling back to an unconditioned one
            # would drop the entire timestep signal and still return a
            # well-shaped, finite tensor.
            raise ValueError(
                "Pi05AdaRMSNorm was built with cond_dim="
                f"{self.cond_dim} but called without an AdaRMS conditioning vector."
            )

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected AdaRMS cond dim {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if x.ndim == 3:
            # (B, 3*dim) → (B, 1, 3*dim), broadcast across the token axis: the
            # timestep is a per-sample scalar, identical for every action token.
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _gated_residual(residual: torch.Tensor, out: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return residual + out
    return residual + gate * out


# ──────────────────────────────────────────────────────────────────────
# Dual-backbone: PaliGemma + AdaRMS action expert
# ──────────────────────────────────────────────────────────────────────
def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    ``(B, num_kv_heads, S, D) → (B, num_kv_heads * n_rep, S, D)``.
    """
    if n_rep == 1:
        return hidden_states
    b, nh, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nh, n_rep, s, d)
    return hidden_states.reshape(b, nh * n_rep, s, d)


def _attend(query_states, key_states, value_states, attention_mask, num_kv_groups, scaling):
    """Manual eager attention: ``softmax(Q Kᵀ · scale + mask) · V``.

    The mask is sliced to ``key_states.shape[-2]`` so the same
    ``(B, 1, Q, prefix+suffix)`` mask works in both the prefix pass (K length =
    prefix) and the suffix pass (K length = prefix + suffix).
    """
    k = _repeat_kv(key_states, num_kv_groups)
    v = _repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : k.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, v)


def _match(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast ``tensor`` to the dtype ``module``'s weight expects."""
    return tensor.to(module.weight.dtype) if tensor.dtype != module.weight.dtype else tensor


def _compute_layer_prefix_only(layer_idx, hidden_states, attention_mask, position_ids, paligemma):
    """Run one PaliGemma LM layer on the prefix, returning the layer output and
    the post-RoPE ``(k, v)`` for the suffix pass.

    Identical to π0: the prefix backbone is unchanged in π0.5 (no AdaRMS —
    there is no timestep in the prefix).
    """
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]
    residual = hidden_states
    x = _match(layer.input_layernorm(hidden_states), layer.self_attn.q_proj)

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

    out = layer.self_attn.o_proj(_match(att, layer.self_attn.o_proj)) + residual
    after_resid = out
    normed = layer.post_attention_layernorm(out)
    out = layer.mlp(_match(normed, layer.mlp.up_proj)) + after_resid
    return out, (k, v)


def _compute_layer_suffix_only(
    layer_idx,
    hidden_states,
    prefix_kv,
    attention_mask,
    position_ids,
    gemma_expert,
    adarms_cond,
):
    """Run one action-expert layer on the suffix with AdaRMS conditioning.

    This is where π0.5 diverges from π0. Both norms are
    :class:`Pi05AdaRMSNorm` and each returns a gate that scales its sublayer's
    contribution to the residual stream.
    """
    layer = gemma_expert.model.layers[layer_idx]

    residual = hidden_states
    x, gate = layer.input_layernorm(hidden_states, adarms_cond)
    x = _match(x, layer.self_attn.q_proj)

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

    hidden_states = _gated_residual(residual, layer.self_attn.o_proj(_match(att, layer.self_attn.o_proj)), gate)

    residual = hidden_states
    x, gate = layer.post_attention_layernorm(hidden_states, adarms_cond)
    return _gated_residual(residual, layer.mlp(_match(x, layer.mlp.up_proj)), gate)


class PaliGemmaWithActionExpertPi05(nn.Module):
    """Dual-backbone transformer: PaliGemma (Gemma 2B) + AdaRMS expert (300M).

    Same two-mode dispatch as π0 (``prefix_only`` / ``suffix_only``), with one
    structural change: after building a stock ``GemmaForCausalLM`` expert, every
    norm in it is swapped for a :class:`Pi05AdaRMSNorm` carrying a ``dense``
    conditioning projection.

    Swapping in place — rather than subclassing ``GemmaModel`` as #4419 does —
    keeps the module tree, and therefore the checkpoint key layout, identical to
    the expert's stock layout apart from the norms themselves.
    """

    def __init__(self, vlm_config, action_expert_config):
        super().__init__()

        # PaliGemma prefix: identical to π0. It sees no timestep, so no AdaRMS.
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
        # suffix action embeddings we feed in.
        self.gemma_expert.model.embed_tokens = None

        self.adarms_cond_dim = action_expert_config.width
        self._install_adarms_norms(action_expert_config)

    def _install_adarms_norms(self, action_expert_config) -> None:
        """Replace every action-expert RMSNorm with a conditioned AdaRMS norm."""
        expert = self.gemma_expert.model
        eps = getattr(self.gemma_expert.config, "rms_norm_eps", 1e-6)
        width = action_expert_config.width
        for layer in expert.layers:
            layer.input_layernorm = Pi05AdaRMSNorm(width, eps=eps, cond_dim=self.adarms_cond_dim)
            layer.post_attention_layernorm = Pi05AdaRMSNorm(width, eps=eps, cond_dim=self.adarms_cond_dim)
        expert.norm = Pi05AdaRMSNorm(width, eps=eps, cond_dim=self.adarms_cond_dim)

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with SigLIP vision tower + PaliGemma projector.

        The two steps are run explicitly rather than via
        ``PaliGemmaModel.get_image_features`` because that helper divides the
        projector output by ``sqrt(text_hidden_size)``. Being explicit keeps the
        scale unambiguous and matches π0 exactly (SigLIP is unchanged in π0.5).
        """
        # Shapes: pixel_values (B, 3, 224, 224) → SigLIP (B, 256, 1152)
        #                                      → projector (B, 256, 2048)
        vision_outputs = self.paligemma.model.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state
        return self.paligemma.model.multi_modal_projector(image_features)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens, returning the ``* sqrt(hidden)``-scaled embedding.

        The scaling location moved across transformers releases: at ≤5.3 the
        normalizer lives inside ``GemmaModel.forward`` (which we bypass), and at
        ≥5.4 ``GemmaTextScaledWordEmbedding`` self-applies it. Detect and avoid
        double-scaling.
        """
        embed_tokens = self.paligemma.model.language_model.embed_tokens
        lang_emb = embed_tokens(tokens)
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
        adarms_cond: torch.Tensor | None = None,
    ):
        """Dispatch to prefix_only / suffix_only and return
        ``([prefix_out, suffix_out], past_key_values_or_None)``.
        """
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        pali_lm = self.paligemma.model.language_model
        expert_lm = self.gemma_expert.model

        if inputs_embeds[1] is None:
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
                "PaliGemmaWithActionExpertPi05.forward only supports prefix-only "
                "or suffix-only dispatch; got both inputs_embeds populated."
            )
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
                gemma_expert=self.gemma_expert,
                adarms_cond=adarms_cond,
            )
        hidden_states, _ = expert_lm.norm(hidden_states, adarms_cond)
        return [None, hidden_states], None


# ──────────────────────────────────────────────────────────────────────
# Main π0.5 Model
# ──────────────────────────────────────────────────────────────────────
class Pi05ForActionPrediction(nn.Module):
    """π0.5 VLA model for robot action prediction via flow matching.

    Inference flow:
      1. Embed prefix (images + language, where the language already carries the
         discretized state) → prefix tokens.
      2. Forward prefix through PaliGemma → layer-wise KV cache.
      3. For each denoising step ``t = 1.0, 1-dt, ..., 0``:
         a. Embed the timestep → an AdaRMS conditioning vector.
         b. Embed the suffix (action tokens only).
         c. Forward the suffix through the AdaRMS action expert.
         d. ``x_t = x_t + dt * v_t`` (Euler integration).
      4. Return ``x_0`` as the predicted action chunk.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        # ``quant_config`` is accepted for interface compatibility but unused —
        # π0.5 runs in full precision for flow-matching parity.
        del quant_config
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
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
        self.paligemma_with_expert = PaliGemmaWithActionExpertPi05(vlm_config, expert_config)

        # Action chunk projections.
        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)

        # π0.5 timestep MLP: (W → W → W) with SiLU, feeding AdaRMS.
        # π0 instead has action_time_mlp_{in,out} of shape (2W → W → W) because
        # it concatenates the time embedding onto the action embedding.
        # NOTE: there is deliberately **no** ``state_proj`` here — that is the
        # π0-only continuous-state path.
        self.time_mlp_in = nn.Linear(self.expert_width, self.expert_width)
        self.time_mlp_out = nn.Linear(self.expert_width, self.expert_width)

        # π0.5 checkpoints commonly carry quantile stats; see _build_norm_buffers.
        self._action_norm = _build_norm_buffers(getattr(config, "norm_stats", None), "action")
        if self._action_norm is None:
            logger.info(
                "π0.5: no action normalization stats on config.norm_stats — "
                "returned actions are in the model's normalized space."
            )

    # ── Action normalization ─────────────────────────────────────────
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
        """Build prefix embeddings, per-token padding mask, and AR mask.

        Prefix tokens form ``[img_cam_0..., ..., lang_tokens...]`` with fully
        bidirectional attention (all-zero ``att_masks``). Identical to π0 — the
        state is inside ``lang_tokens``, so nothing here changes shape-wise.

        Cameras are embedded one at a time; each call is ``(B, 3, 224, 224)``
        regardless of camera count, so 1/2/3 views need no separate code path.
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

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)

        return embs, pad_masks, att_masks

    # ── Timestep + suffix embedding ──────────────────────────────────
    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Timestep → AdaRMS conditioning vector ``(B, expert_width)``.

        ``silu(time_mlp_out(silu(time_mlp_in(sinusoid(t)))))``. The trailing
        SiLU is part of the reference implementation — dropping it is a silent
        numerical error, not a crash.
        """
        model_dtype = self.action_in_proj.weight.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=getattr(self.config, "min_period", 4e-3),
            max_period=getattr(self.config, "max_period", 4.0),
            device=timestep.device,
        ).to(dtype=model_dtype)
        time_cond = self.time_mlp_in(time_emb)
        time_cond = F.silu(time_cond)
        time_cond = self.time_mlp_out(time_cond)
        return F.silu(time_cond)

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the suffix: **action tokens only**, plus the AdaRMS condition.

        π0's suffix is ``[state_token, action_tokens×H]`` with an AR mask of
        ``[1, 1, 0...]``. π0.5 has no state token, so the suffix is
        ``[action_tokens×H]`` and the mask is ``[1] + [0]*(H-1)``: the first
        action token opens a causal block and the rest attend bidirectionally
        within it.
        """
        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)

        time_cond = self.embed_timestep(timestep)
        action_emb = self.action_in_proj(noisy_actions)  # (B, H, W)

        bsize, action_len = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_len, dtype=torch.bool, device=action_emb.device)
        att_masks = torch.tensor(
            [1] + [0] * (self.action_horizon - 1),
            dtype=action_emb.dtype,
            device=action_emb.device,
        )[None, :].expand(bsize, -1)
        return action_emb, pad_masks, att_masks, time_cond

    # ── Denoising step ───────────────────────────────────────────────
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one flow-matching denoising step: predict ``v_t`` from ``x_t``.

        Signature differs from π0's by exactly one argument: no ``state``.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, time_cond = self.embed_suffix(x_t, timestep)

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
            adarms_cond=time_cond,
        )

        # Every suffix token is an action token in π0.5 (no state token to drop).
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
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk via iterative flow-matching denoising.

        Convention: ``t=1`` is noise, ``t=0`` is the target — opposite of the
        published π0 paper but matching both OpenPI and LeRobot.

        Takes no ``state``: π0.5's state rides inside ``lang_tokens``.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
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
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    # ── Weight loading ───────────────────────────────────────────────
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        strict: bool = True,
    ):
        """Load and audit a LeRobot π0.5 safetensors checkpoint.

        Same remap rules as π0 (strip the ``model.`` prefix, flatten→nested
        PaliGemma submodules, tied ``lm_head`` → ``embed_tokens``, version-robust
        SigLIP nesting), plus two π0.5-specific ones:

          * ``action_time_mlp_{in,out}`` → ``time_mlp_{in,out}``: some
            checkpoints were exported under the π0 parameter names.
          * ``state_proj.*`` is reported, not silently dropped. A π0.5
            checkpoint should not contain it; its presence usually means a π0
            checkpoint was pointed at the π0.5 model class, which would
            otherwise run happily with a randomly-initialized action expert.

        The action-expert norms are AdaRMS here, so they expose ``dense.weight``
        / ``dense.bias`` and no plain ``weight``. A checkpoint that carries a
        plain expert-norm ``weight`` is a π0-shaped checkpoint; that too is
        rejected rather than skipped. ``strict=False`` exists only for focused
        remapping unit tests that intentionally provide a partial state dict;
        the serving path always uses the strict default.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        _PALIGEMMA_SUBMODULES = ("vision_tower", "multi_modal_projector", "language_model")
        _EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."

        def _remap(name: str) -> str:
            # Strip the leading "model." that LeRobot's PI05Policy wrapper adds.
            if name.startswith("model."):
                name = name[len("model.") :]

            # π0-style timestep MLP names → π0.5 names.
            if name.startswith("action_time_mlp_in."):
                name = "time_mlp_in." + name[len("action_time_mlp_in.") :]
            elif name.startswith("action_time_mlp_out."):
                name = "time_mlp_out." + name[len("action_time_mlp_out.") :]

            # Nested PaliGemma layout.
            for sub in _PALIGEMMA_SUBMODULES:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat) :]

            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

            return name

        def _fix_vision_tower(name: str) -> str:
            """Reconcile SigLIP nesting across transformers versions (≤5.3 wraps
            the encoder in ``vision_tower.vision_model.*``; ≥5.4 flattens it)."""
            vt = "paligemma_with_expert.paligemma.model.vision_tower."
            if not name.startswith(vt):
                return name
            rest = name[len(vt) :]
            if name in params_dict or name in buffers_dict:
                return name
            if rest.startswith("vision_model."):
                candidate = vt + rest[len("vision_model.") :]
            else:
                candidate = vt + "vision_model." + rest
            return candidate if (candidate in params_dict or candidate in buffers_dict) else name

        loaded = 0
        skipped: list[str] = []
        pi0_shaped: list[str] = []
        filled_params: set = set()

        for name, loaded_weight in weights:
            mapped = _fix_vision_tower(_remap(name))

            # Diagnose π0-shaped keys instead of dropping them quietly.
            is_expert_norm_weight = mapped.startswith(_EXPERT_PREFIX) and (
                mapped.endswith("input_layernorm.weight")
                or mapped.endswith("post_attention_layernorm.weight")
                or mapped == _EXPERT_PREFIX + "norm.weight"
            )
            if mapped.startswith("state_proj.") or is_expert_norm_weight:
                pi0_shaped.append(mapped)
                continue

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

        # LeRobot stores PaliGemma's tied text embedding as lm_head.weight; keep
        # lm_head filled so the tied-weight state stays consistent.
        embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
        if embed_key in filled_params and lm_head_key in params_dict and lm_head_key not in filled_params:
            params_dict[lm_head_key].data.copy_(params_dict[embed_key].data)
            filled_params.add(lm_head_key)

        # Reverse audit: any model param that got no checkpoint tensor at all
        # would be running with random init.
        missing_params: list[str] = []
        for pname in params_dict:
            if pname in filled_params:
                continue
            if "rotary_emb" in pname or pname.endswith(".inv_freq"):
                continue
            missing_params.append(pname)

        parts: list[str] = []
        if pi0_shaped:
            parts.append(
                f"{len(pi0_shaped)} checkpoint key(s) are π0-shaped, not π0.5-shaped (first 5: {pi0_shaped[:5]})"
            )
        if skipped:
            parts.append(f"{len(skipped)} checkpoint key(s) had no model target (first 5: {skipped[:5]})")
        if missing_params:
            parts.append(f"{len(missing_params)} model param(s) received no weight (first 5: {missing_params[:5]})")

        if parts and strict:
            raise RuntimeError("Incomplete or incompatible π0.5 checkpoint: " + "; ".join(parts))
        if parts:
            logger.debug("π0.5 partial test load: %d tensors loaded — %s.", loaded, "; ".join(parts))
        else:
            logger.info("π0.5 load_weights: %d tensors loaded, 0 skipped, 0 missing.", loaded)
        return filled_params


EntryClass = Pi05ForActionPrediction
