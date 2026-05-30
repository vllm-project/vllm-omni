# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE-Step 1.5 DiT Model for vLLM-Omni.

Ported from diffusers PR #13095:
src/diffusers/models/transformers/ace_step_transformer.py

Known limitation (tracked separately):
    The DiT uses sliding-window self-attention. vllm-omni's flash backend has no
    ``window_size`` plumbing today, so flash silently drops the window
    constraint and the model produces noise. ``AceStepAttention`` hard-pins
    SDPA for sliding-window self-attention sites at construction time
    (cross-attention and full self-attention sites keep flash). The pin is
    automatic — no user flag required — and will be removed in a follow-up PR
    once ``window_size`` plumbing lands in
    ``vllm_omni/diffusion/attention/backends/flash_attn.py``.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import (
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
#                                attention-mask                                #
# --------------------------------------------------------------------------- #


def _create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    """Build a `[B, 1, seq_len, seq_len]` additive mask (0.0 kept, -inf masked).

    Mirrors the mask construction in the original ACE-Step DiT so the model sees
    identical attention coverage regardless of the attention backend selected.
    """
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    valid_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    if is_causal:
        valid_mask = valid_mask & (diff >= 0)

    if is_sliding_window and sliding_window is not None:
        if is_causal:
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            valid_mask = valid_mask & (torch.abs(diff) <= sliding_window)

    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is not None:
        padding_mask_4d = attention_mask.view(attention_mask.shape[0], 1, 1, seq_len).to(torch.bool)
        valid_mask = valid_mask & padding_mask_4d

    min_dtype = torch.finfo(dtype).min
    mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)
    mask_tensor.masked_fill_(valid_mask, 0.0)
    return mask_tensor


# --------------------------------------------------------------------------- #
#                                 RoPE helpers                                 #
# --------------------------------------------------------------------------- #


def _ace_step_rotary_freqs(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) freqs for ACE-Step RoPE.

    ACE-Step reuses Qwen3's rotary layout: ``freqs = cat([freq_half, freq_half], dim=-1)``
    (not interleaved). That matches ``get_1d_rotary_pos_embed(..., use_real=True,
    repeat_interleave_real=False)`` combined with ``apply_rotary_emb(..., use_real_unbind_dim=-2)``.
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    cos, sin = get_1d_rotary_pos_embed(head_dim, positions, theta=theta, use_real=True, repeat_interleave_real=False)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


# --------------------------------------------------------------------------- #
#                                building blocks                               #
# --------------------------------------------------------------------------- #


class AceStepMLP(nn.Module):
    """SwiGLU MLP used in ACE-Step transformer blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AceStepTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP + 6-way AdaLN scale/shift projection.

    Matches the original ACE-Step checkpoint layout (``linear_1``, ``linear_2``, ``time_proj``).
    """

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 2048, scale: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.time_sinusoid = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_freq = self.time_sinusoid(t * self.scale)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj


class AceStepAttention(nn.Module):
    """GQA attention with RMSNorm on query/key for ACE-Step 1.5.

    Self-attention applies RoPE on query/key; cross-attention reads K/V from
    ``encoder_hidden_states`` and does not apply RoPE. GQA is implemented by
    manually expanding K/V heads to match Q heads (vllm-omni's Attention layer
    expects matched head counts), mirroring the stable_audio pattern.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-6,
        sliding_window: int | None = None,
        is_cross_attention: bool = False,
        role: str = "self",
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.sliding_window = sliding_window
        self.is_cross_attention = is_cross_attention
        self.role = role
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        q_dim = num_attention_heads * head_dim
        kv_dim = num_key_value_heads * head_dim

        self.to_q = ReplicatedLinear(hidden_size, q_dim, bias=bias)
        self.to_k = ReplicatedLinear(hidden_size, kv_dim, bias=bias)
        self.to_v = ReplicatedLinear(hidden_size, kv_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(q_dim, hidden_size, bias=bias),
                nn.Dropout(dropout),
            ]
        )
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)

        # vllm-omni Attention; K/V are expanded to match Q heads before dispatch,
        # so num_kv_heads equals num_heads here.
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
            num_kv_heads=num_attention_heads,
            role=role,
        )

        # Mask-shape compatibility per backend (mirrors the branching the
        # diffusers source did inside its AceStepAttnProcessor2_0):
        #
        #   - SDPA / NPU / others accept the 4D additive mask we build via
        #     ``_create_4d_mask``. Sliding window is enforced by the mask.
        #   - Flash attention (vllm-omni's variant) only accepts a 2D padding
        #     mask and has NO ``window_size`` kwarg, so sliding-window cannot
        #     be enforced with flash on this codebase. For sliding-window
        #     self-attention layers we hard-pin the wrapper's inner backend to
        #     SDPA (via the existing ``sdpa_fallback`` instance) — verified
        #     empirically that without this the model produces noise on
        #     alternating layers. Flash is still used for cross-attention and
        #     full self-attention layers.
        backend_name = self.attn.attn_backend.get_name().upper()
        self._uses_flash = "FLASH" in backend_name
        if self._uses_flash and sliding_window is not None and not is_cross_attention:
            self.attn.attention = self.attn.sdpa_fallback
            self._uses_flash = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        is_cross = self.is_cross_attention and encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        kv_seq_len = kv_input.shape[1]

        # Project. Q has ``heads * head_dim``; K/V have ``kv_heads * head_dim``.
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(kv_input)
        value, _ = self.to_v(kv_input)

        query = query.view(batch_size, seq_len, self.heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.kv_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.kv_heads, self.head_dim)

        # Per-head RMSNorm on Q/K before RoPE.
        query = self.norm_q(query)
        key = self.norm_k(key)

        # RoPE on self-attention only.
        if not is_cross and image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2, sequence_dim=1)

        # GQA expansion: [B, S, kv_heads, D] -> [B, S, heads, D].
        if self.num_kv_groups > 1:
            key = key.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            key = key.reshape(batch_size, kv_seq_len, self.heads, self.head_dim)
            value = value.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            value = value.reshape(batch_size, kv_seq_len, self.heads, self.head_dim)

        # Backend-specific mask shape:
        #   - flash: 2D padding mask `[B, S_kv]` (any positive value = keep).
        #     ``_create_4d_mask`` produces `[B, 1, S_q, S_kv]` with `0.0` for
        #     keep and `-inf` for mask; collapse along the query and singleton
        #     dims with ``any`` to recover the padding axis.
        #   - others: pass the 4D additive mask through unchanged.
        # When the model code passes ``None`` (full-attention layers / cross-attn)
        # we pass ``attn_metadata=None`` so the backend takes its unmasked fast path.
        dispatch_mask = attention_mask
        if dispatch_mask is not None and self._uses_flash and dispatch_mask.ndim == 4:
            keep_mask = dispatch_mask if dispatch_mask.dtype == torch.bool else (dispatch_mask == 0)
            dispatch_mask = keep_mask.any(dim=(1, 2))

        if dispatch_mask is not None:
            attn_metadata = AttentionMetadata(attn_mask=dispatch_mask)
            hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)
        else:
            hidden_states = self.attn(query, key, value)

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class AceStepTransformerBlock(nn.Module):
    """ACE-Step DiT block: self-attn (AdaLN) → cross-attn → MLP (AdaLN).

    AdaLN parameters come from ``scale_shift_table + temb`` chunked into 6
    (3 for self-attn + 3 for MLP).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int | None = None,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.self_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = AceStepAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            bias=attention_bias,
            dropout=attention_dropout,
            eps=rms_norm_eps,
            sliding_window=sliding_window,
            is_cross_attention=False,
            role="self",
        )

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = AceStepAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                bias=attention_bias,
                dropout=attention_dropout,
                eps=rms_norm_eps,
                is_cross_attention=True,
                role="cross",
            )

        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = AceStepMLP(hidden_size, intermediate_size)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb).chunk(
            6, dim=1
        )

        # Self-attention with AdaLN.
        norm_hidden_states = (self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # Cross-attention (no AdaLN, plain residual).
        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
            attn_output = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # MLP with AdaLN.
        norm_hidden_states = (self.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
#                                 main DiT model                               #
# --------------------------------------------------------------------------- #


class AceStepTransformer1DModel(nn.Module):
    """Diffusion Transformer for ACE-Step 1.5 music generation.

    Generates audio latents conditioned on text, lyrics, and timbre. Uses 1D
    patch embedding (``Conv1d`` with stride ``patch_size``) followed by a stack
    of ``AceStepTransformerBlock`` with alternating sliding-window / full
    attention on the self-attention branch. Cross-attention consumes the packed
    ``encoder_hidden_states`` produced by ``AceStepConditionEncoder``.
    """

    _repeated_blocks = ["AceStepTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]

    @staticmethod
    def _is_ace_step_layer(name: str, module: object) -> bool:
        """HSDP shard predicate.

        ACE-Step's DiT calls its transformer block list ``layers`` (mirroring
        the diffusers source — renaming would require remapping every weight
        name in the checkpoint). The shared ``is_transformer_block_module``
        helper hardcodes ``transformer_blocks``, so we provide a model-local
        predicate that matches ``layers.<int>`` instead.
        """
        return "layers" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_ace_step_layer]

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        in_channels: int = 192,
        audio_acoustic_hidden_dim: int = 64,
        patch_size: int = 2,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 128,
        layer_types: list | None = None,
        # Equal to ``hidden_size`` on base/turbo models; the XL turbo has a
        # smaller condition encoder feeding a wider DiT, so we project up.
        encoder_hidden_size: int | None = None,
        # Variant metadata. Turbo models distill guidance into the weights and
        # run without CFG; base/SFT models require CFG with the learned
        # ``AceStepConditionEncoder.null_condition_emb``.
        is_turbo: bool = False,
        model_version: str | None = None,
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config is not None else None

        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size
        self.patch_size = patch_size
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(num_hidden_layers)
            ]
        self.layer_types = list(layer_types)

        # Stash config so the pipeline can introspect it (mirrors stable_audio).
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "in_channels": in_channels,
                "audio_acoustic_hidden_dim": audio_acoustic_hidden_dim,
                "patch_size": patch_size,
                "rope_theta": rope_theta,
                "sliding_window": sliding_window,
                "encoder_hidden_size": encoder_hidden_size,
                "is_turbo": is_turbo,
                "model_version": model_version,
            },
        )()

        self.layers = nn.ModuleList(
            [
                AceStepTransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=sliding_window if layer_types[i] == "sliding_attention" else None,
                    use_cross_attention=True,
                )
                for i in range(num_hidden_layers)
            ]
        )

        # Patchify: Conv1d(stride=patch_size) lifts (B, T, in_channels) ->
        # (B, T/patch_size, hidden_size). The input is `cat(context_latents, hidden_states)`
        # along the channel dim.
        self.proj_in_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Dual-timestep conditioning: one path for ``t``, one for ``(t - r)``.
        self.time_embed = AceStepTimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.time_embed_r = AceStepTimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)

        self.condition_embedder = nn.Linear(encoder_hidden_size, hidden_size, bias=True)

        self.norm_out = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj_out_conv = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=audio_acoustic_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size**0.5)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """The [`AceStepTransformer1DModel`] forward method.

        Args:
            hidden_states: Noisy latent input `(batch, seq_len, channels)`.
            timestep: Current diffusion timestep `t` `(batch,)`.
            timestep_r: Reference timestep `r` `(batch,)` (set equal to `t`
                for standard inference).
            encoder_hidden_states: Conditioning embeddings from the condition
                encoder (text + lyrics + timbre), shape
                `(batch, encoder_seq_len, encoder_hidden_size)`.
            context_latents: Context latents (source latents concatenated with
                chunk masks) — fed alongside ``hidden_states`` to the patchify
                conv, shape `(batch, seq_len, context_dim)`.
            return_dict: Whether to return ``Transformer2DModelOutput`` or a tuple.

        Returns:
            Predicted velocity field (flow matching).
        """
        # Dual timestep embedding: t and (t - r). Sum the AdaLN projections.
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concat context latents on the channel dim, pad to patch boundary, patchify.
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden_states.shape[1]
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode="constant", value=0)
        hidden_states = self.proj_in_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device

        cos, sin = _ace_step_rotary_freqs(seq_len, self.head_dim, self.rope_theta, device, dtype)
        position_embeddings = (cos, sin)

        # Sliding-window self-attention mask. Only the sliding-attention layers
        # use it; full-attention layers see no mask. Cross-attention is unmasked.
        sliding_attn_mask = _create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            sliding_window=self.config.sliding_window,
            is_sliding_window=True,
            is_causal=False,
        )

        for i, layer_module in enumerate(self.layers):
            layer_attn_mask = sliding_attn_mask if self.layer_types[i] == "sliding_attention" else None
            hidden_states = layer_module(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                temb=timestep_proj,
                attention_mask=layer_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
            )

        # Adaptive output norm + de-patchify back to original sequence length.
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states[:, :original_seq_len, :]

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a diffusers checkpoint.

        Diffusers uses ``layers.N.self_attn.processor.*`` for processor state
        (we no longer have a separate processor here) but the parameter names
        on the AceStepAttention / MLP modules match ours 1:1.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                except AssertionError as err:
                    raise AssertionError(f"Failed to load ACE-Step weight {name!r}") from err
                loaded_params.add(name)
            else:
                logger.debug("Skipping weight %s - not found in model", name)

        return loaded_params
