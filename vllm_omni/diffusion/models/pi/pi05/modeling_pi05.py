# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only π0.5 VLA math kernel for vllm-omni.

Only the math that turns a robot observation into an action chunk; no serving or
request glue. Deliberately shaped like ``models/pi/pi0/modeling_pi0.py`` so the two
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
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.pi.common import attention, backbone, checkpoint, flow_matching

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 50
DEFAULT_MAX_TOKEN_LEN = 200  # π0 uses 48
DEFAULT_NUM_INFERENCE_STEPS = 10
DEFAULT_IMAGE_RESOLUTION = (224, 224)  # openpi/models/model.py IMAGE_RESOLUTION
DEFAULT_STATE_NUM_BINS = 256  # openpi PaliGemmaTokenizer.tokenize()

# Large negative value to fill masked-out positions in a float attention mask.
# Matches OpenPI's constant exactly so that numerics line up during parity.
# Ref: openpi/src/openpi/models/gemma.py
OPENPI_ATTENTION_MASK_VALUE = attention.OPENPI_ATTENTION_MASK_VALUE
make_att_2d_masks = attention.make_att_2d_masks
prepare_attention_masks_4d = attention.prepare_attention_masks_4d
create_sinusoidal_pos_embedding = flow_matching.create_sinusoidal_pos_embedding


# ──────────────────────────────────────────────────────────────────────
# Gemma variant configs (matches openpi/models/gemma.py get_config)
# ──────────────────────────────────────────────────────────────────────
GemmaVariantConfig = backbone.GemmaVariantConfig
get_gemma_config = backbone.get_gemma_config


# ──────────────────────────────────────────────────────────────────────
# Utility functions (match openpi/models_pytorch/pi0_pytorch.py)
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
def _match(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast ``tensor`` to the dtype ``module``'s weight expects."""
    return tensor.to(module.weight.dtype) if tensor.dtype != module.weight.dtype else tensor


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

    att = attention.eager_attention(
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
        self.paligemma, self.gemma_expert = backbone.build_backbones(vlm_config, action_expert_config)

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
        expert_lm = self.gemma_expert.model

        if inputs_embeds[1] is None:
            hidden_states, kv_list = backbone.execute_prefix(
                inputs_embeds[0],
                attention_mask,
                position_ids,
                self.paligemma,
                align_module_input=_match,
            )
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
        # quant_config is not plumbed through; the weight dtype comes from the pipeline.
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

    # ── Timestep + suffix embedding ──────────────────────────────────
    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Timestep → AdaRMS conditioning vector ``(B, expert_width)``.

        ``silu(time_mlp_out(silu(time_mlp_in(sinusoid(t)))))``. The trailing
        SiLU is part of the reference implementation — dropping it is a silent
        numerical error, not a crash.
        """
        model_dtype = self.action_in_proj.weight.dtype
        time_emb = flow_matching.create_sinusoidal_pos_embedding(
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
        generator: torch.Generator | list[torch.Generator] | None = None,
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
            noise_shape = (self.action_horizon, self.action_dim)
            if isinstance(generator, list):
                if len(generator) != bsize:
                    raise ValueError(f"Expected {bsize} generators, got {len(generator)}.")
                noise = torch.stack(
                    [torch.randn(noise_shape, dtype=torch.float32, device=device, generator=item) for item in generator]
                )
            else:
                noise = torch.randn(
                    bsize,
                    *noise_shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )

        # 1. Compose the ordered camera and language prefix. The deployed
        # π0.5 layout always retains max_cameras slots; missing cameras occupy
        # their slot with a false image mask. State is already in lang_tokens.
        prefix_embs, prefix_pad_masks, prefix_att_masks = backbone.embed_multimodal_prefix(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            paligemma=self.paligemma_with_expert.paligemma,
            expected_num_views=int(self.config.max_cameras),
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
        x_t = noise
        for t, dt in flow_matching.make_euler_schedule(num_steps):
            time_tensor = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = flow_matching.euler_step(x_t, v_t, dt)
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

        _EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
        model_keys = params_dict.keys() | buffers_dict.keys()
        prefix_aliases = (
            ("action_time_mlp_in.", "time_mlp_in."),
            ("action_time_mlp_out.", "time_mlp_out."),
        )

        loaded = 0
        skipped: list[str] = []
        pi0_shaped: list[str] = []
        filled_params: set = set()

        for name, loaded_weight in weights:
            mapped = checkpoint.resolve_parameter_name(name, model_keys, prefix_aliases=prefix_aliases)

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
