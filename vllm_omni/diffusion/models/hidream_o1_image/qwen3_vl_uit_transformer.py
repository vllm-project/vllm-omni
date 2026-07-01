# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image backbone: a Qwen3-VL multimodal LLM extended with a
pixel-space flow-matching diffusion head ("Pixel-level Unified Transformer").

Design decisions (Phase 0 design lock -- see the HiDream-O1-Image roadmap):

1. Attention masking: every forward call builds BOTH a dense boolean
   attention mask and a ``full_attn_spans`` list (see
   ``utils_hidream_o1.build_packed_attention_metadata``) and passes both via
   a single ``AttentionMetadata``. vLLM-Omni's flash-attn backend consumes
   ``full_attn_spans`` (efficient piecewise dispatch); the SDPA backend
   consumes ``attn_mask`` (dense, always correct). This avoids the silent
   wrong-output trap of relying on ``full_attn_spans`` alone, which only the
   flash backend honors.
2. Position ids: 3D MRoPE (T, H, W), produced by
   ``utils_hidream_o1.get_rope_index_fix_point``. The rotary embedding
   module itself is HF transformers' stock ``Qwen3VLTextRotaryEmbedding``,
   used unmodified -- it was diffed against the reference repo's own copy
   during Phase 0 and found functionally identical (same
   ``apply_interleaved_mrope`` channel-order convention).
3. The reference repo's hand-rolled two-pass flash-attention closure
   (``_custom_flash_attn``) is NOT ported. Its net effect -- text attends
   causally to text-only, generation tokens (timestep + image patches)
   attend bidirectionally to everything -- is reproduced by vLLM-Omni's
   existing ``Attention`` layer + ``AttentionMetadata`` plumbing instead.
4. Everything *not* related to diffusion (vision tower, text decoder
   skeleton, RMSNorm, MLP, rotary embeddings) is imported from HF
   `transformers`' Qwen3-VL implementation rather than copied wholesale,
   following the precedent in
   ``vllm_omni/diffusion/models/internvla_a1/adapter_qwen3_vl.py``. Only the
   genuinely new diffusion-only modules (``BottleneckPatchEmbed``,
   ``TimestepEmbedder``, ``FinalLayer``) and the modified decoder forward
   path (attention masking, generation-mode forward) are written here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_MIN_TRANSFORMERS_VERSION = "4.57.1"


def _load_qwen3_vl_classes() -> dict[str, type]:
    """Import Qwen3-VL lazily so a missing/old `transformers` fails with a
    clear, actionable error instead of an opaque ImportError deep in some
    other module's import chain.

    Mirrors ``vllm_omni/diffusion/models/gr00t/modeling/gr00t_n1d7.py``'s
    ``_load_qwen3_vl_cls()`` pattern.
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLModel,
            Qwen3VLPreTrainedModel,
            Qwen3VLTextAttention,
            Qwen3VLTextConfig,
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextModel,
            Qwen3VLTextRMSNorm,
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionModel,
            apply_rotary_pos_emb,
        )
    except ImportError as exc:
        raise ImportError(
            f"HiDream-O1-Image requires transformers>={_MIN_TRANSFORMERS_VERSION} "
            "for transformers.models.qwen3_vl (vllm-omni's declared floor is "
            "4.56.0, which predates Qwen3-VL's addition upstream). "
            "Please upgrade transformers."
        ) from exc
    return {
        "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
        "Qwen3VLModel": Qwen3VLModel,
        "Qwen3VLPreTrainedModel": Qwen3VLPreTrainedModel,
        "Qwen3VLTextAttention": Qwen3VLTextAttention,
        "Qwen3VLTextConfig": Qwen3VLTextConfig,
        "Qwen3VLTextDecoderLayer": Qwen3VLTextDecoderLayer,
        "Qwen3VLTextModel": Qwen3VLTextModel,
        "Qwen3VLTextRMSNorm": Qwen3VLTextRMSNorm,
        "Qwen3VLTextRotaryEmbedding": Qwen3VLTextRotaryEmbedding,
        "Qwen3VLVisionModel": Qwen3VLVisionModel,
        "apply_rotary_pos_emb": apply_rotary_pos_emb,
    }


# ---------------------------------------------------------------------------
# New, diffusion-only modules -- no upstream Qwen3-VL equivalent.
# Ported near-verbatim from the reference repo's
# ``models/qwen3_vl_transformers.py``.
# ---------------------------------------------------------------------------


class BottleneckPatchEmbed(nn.Module):
    """Projects raw pixel patches (``patch_size**2 * in_chans`` values per
    token) through a low-rank bottleneck into the transformer's hidden size.
    This is the model's only "patch embedder" -- there is no VAE.
    """

    def __init__(self, patch_size: int, in_chans: int, pca_dim: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.proj1 = nn.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class FinalLayer(nn.Module):
    """Output head: hidden states -> pixel-patch predictions.

    No AdaLN is used (unlike most DiT final layers) -- the ``adaln_input``
    parameter is accepted for signature parity with the reference repo but
    is never consumed there either. Timestep conditioning enters this model
    via token substitution (see ``Qwen3VLUiTModel.forward_generation``), not
    via modulation of this layer.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, adaln_input: torch.Tensor | None = None) -> torch.Tensor:
        del adaln_input  # unused -- kept for signature parity, see class docstring
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP, in the style of DiT/ADM."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# Decoder-layer port: same as upstream Qwen3-VL except the attention
# computation is rewired to vLLM-Omni's Attention layer + AttentionMetadata.
# ---------------------------------------------------------------------------


def _build_attention_subclasses() -> dict[str, type]:
    """Build the attention/decoder-layer/text-model classes that need the
    masking rewire. Deferred into a function (rather than module-level
    classes) because the HF base classes are only available once
    `transformers` is confirmed importable.
    """
    hf = _load_qwen3_vl_classes()
    HFQwen3VLTextRMSNorm = hf["Qwen3VLTextRMSNorm"]
    HFQwen3VLTextAttention = hf["Qwen3VLTextAttention"]
    HFQwen3VLTextDecoderLayer = hf["Qwen3VLTextDecoderLayer"]
    HFQwen3VLTextModel = hf["Qwen3VLTextModel"]
    Qwen3VLPreTrainedModel = hf["Qwen3VLPreTrainedModel"]
    apply_rotary_pos_emb = hf["apply_rotary_pos_emb"]

    class Qwen3VLTextRMSNorm(HFQwen3VLTextRMSNorm):
        """Float32-upcast RMSNorm, matching the existing internvla_a1 adapter
        precedent and the reference repo's own numerics."""

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * hidden_states).to(input_dtype)

    class Qwen3VLTextAttention(HFQwen3VLTextAttention):
        """Same Q/K/V projections, Q/K-norm, and RoPE as upstream Qwen3-VL,
        but dispatches the actual attention computation through vLLM-Omni's
        ``Attention`` layer instead of HF's native eager/SDPA path, so that
        the mixed causal(text)/full(image) mask (Phase 0 design lock) and
        future backend selection / parallelism hooks are available.
        """

        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.attn = Attention(
                num_heads=self.config.num_attention_heads,
                head_size=self.head_dim,
                softmax_scale=self.scaling,
                causal=False,  # masking is fully expressed via AttentionMetadata
                num_kv_heads=self.config.num_key_value_heads,
                prefix=f"hidream_o1.language_model.layers.{layer_idx}.self_attn",
                role="self",
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attn_metadata: AttentionMetadata,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # NOTE: deliberately NOT transposed to [B, H, S, D] here -- vLLM-Omni's
            # Attention layer expects [B, S, H, D]. RoPE is applied in-place below
            # with unsqueeze_dim=2 (heads at dim 2, not dim 1) to match.
            q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
            k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
            v = self.v_proj(hidden_states).view(hidden_shape)

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

            attn_output = self.attn(q, k, v, attn_metadata=attn_metadata)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

    class Qwen3VLTextDecoderLayer(HFQwen3VLTextDecoderLayer):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
            self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attn_metadata: AttentionMetadata,
            **kwargs: Any,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

    class Qwen3VLTextModel(HFQwen3VLTextModel):
        """Text decoder stack. ``forward()`` is rewritten (rather than
        inheriting HF's) because HF's version builds a causal-only mask via
        ``create_causal_mask`` and threads scalar/2D position ids -- neither
        fits this model's mixed-mask, 3D-MRoPE generation path.
        """

        _repeated_blocks = ["Qwen3VLTextDecoderLayer"]

        def __init__(self, config):
            Qwen3VLPreTrainedModel.__init__(self, config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = hf["Qwen3VLTextRotaryEmbedding"](config=config)
            self.gradient_checkpointing = False
            self.post_init()

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            position_ids: torch.Tensor,
            attn_metadata: AttentionMetadata,
            visual_pos_masks: torch.Tensor | None = None,
            deepstack_visual_embeds: list[torch.Tensor] | None = None,
        ) -> torch.Tensor:
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(self.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attn_metadata=attn_metadata,
                )
                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

            return self.norm(hidden_states)

        @staticmethod
        def _deepstack_process(
            hidden_states: torch.Tensor,
            visual_pos_masks: torch.Tensor,
            visual_embeds: torch.Tensor,
        ) -> torch.Tensor:
            """Residually add deepstack vision features at VLM-ref-image positions.

            Near-verbatim port of the reference repo's
            ``Qwen3VLTextModel._deepstack_process`` (qwen3_vl_transformers.py:938-944).
            `visual_pos_masks` is True at every image-token-id position in the
            (text + vinputs) sequence; `visual_embeds` is the merged intermediate
            vision feature for the same set of positions.
            """
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds.to(hidden_states.dtype)
            hidden_states[visual_pos_masks, :] = local_this
            return hidden_states

    return {
        "Qwen3VLTextRMSNorm": Qwen3VLTextRMSNorm,
        "Qwen3VLTextAttention": Qwen3VLTextAttention,
        "Qwen3VLTextDecoderLayer": Qwen3VLTextDecoderLayer,
        "Qwen3VLTextModel": Qwen3VLTextModel,
    }


class HiDreamO1ModelOutput:
    """Lightweight output container -- avoids depending on HF's
    ``Qwen3VLModelOutputWithPast`` dataclass, whose fields we don't need.
    """

    def __init__(self, x_pred: torch.Tensor, cond_image_embeds=None, cond_deepstack_image_embeds=None):
        self.x_pred = x_pred
        self.cond_image_embeds = cond_image_embeds
        self.cond_deepstack_image_embeds = cond_deepstack_image_embeds


class HiDreamO1UiTModel(nn.Module):
    """The full Pixel-level Unified Transformer: Qwen3-VL vision tower +
    text decoder (masking-rewired, see module docstring) + the new
    diffusion-only modules (``x_embedder``, ``t_embedder1``,
    ``final_layer2``).

    Only ``forward_generation()`` is implemented -- this model is never
    asked to autoregressively decode text in this pipeline, so the
    `transformers`-style causal-LM ``forward()``/``generate()`` entry points
    are intentionally not ported.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int = 32,
        in_channels: int = 3,
        bottleneck_dim: int = 1024,
        tms_token_id: int | None = None,
    ):
        super().__init__()
        hf = _load_qwen3_vl_classes()
        subclasses = _build_attention_subclasses()

        config_cls = None
        model_path = od_config.model if od_config is not None else None
        if model_path is None:
            raise ValueError("HiDreamO1UiTModel requires od_config.model (a HF config/checkpoint path).")
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        self.config = config
        del config_cls

        self.visual = hf["Qwen3VLVisionModel"]._from_config(config.vision_config)
        self.language_model = subclasses["Qwen3VLTextModel"]._from_config(config.text_config)

        hidden_size = config.text_config.hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.t_embedder1 = TimestepEmbedder(hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            patch_size=patch_size, in_chans=in_channels, pca_dim=bottleneck_dim, embed_dim=hidden_size
        )
        self.final_layer2 = FinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels)

        if tms_token_id is None:
            raise ValueError(
                "HiDreamO1UiTModel requires tms_token_id (the checkpoint's timestep "
                "placeholder special-token id) -- resolve it from the tokenizer, "
                "do not hardcode it."
            )
        self.tms_token_id = tms_token_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the Qwen3-VL vision tower and return (merged_patches, deepstack_features).

        HF's ``Qwen3VLVisionModel.forward()`` returns a
        ``BaseModelOutputWithDeepstackFeatures`` with:
        - ``pooler_output``: spatially-merged patch embeddings,
          shape ``[total_merged_patches, hidden_size]``.
        - ``deepstack_features``: list of intermediate-layer features
          (one per ``config.deepstack_visual_indexes`` entry), each
          ``[total_merged_patches, hidden_size]``.
        """
        out = self.visual(pixel_values, grid_thw=image_grid_thw)
        return out.pooler_output, out.deepstack_features

    def forward_generation(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        attn_metadata: AttentionMetadata,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        precomputed_image_embeds: torch.Tensor | None = None,
        precomputed_deepstack_image_embeds: list[torch.Tensor] | None = None,
    ) -> HiDreamO1ModelOutput:
        """One denoising-step forward pass.

        Args:
            input_ids: ``(B, txt_seq_len)`` -- text tokens only (no image
                placeholders included in the *embedding lookup*; image
                positions get their embeddings overwritten/appended below).
            position_ids: ``(3, B, total_seq_len)`` -- 3D MRoPE, covering
                text + image positions (see
                ``utils_hidream_o1.get_rope_index_fix_point``).
            vinputs: ``(B, img_tokens, patch_dim)`` -- patchified noisy
                target image (and, from Phase 2 onward, reference images).
            timestep: ``(B,)`` scalar flow-matching timestep per sample.
            token_types: ``(B, total_seq_len)`` -- 0 text, >0 generation.
            attn_metadata: built once by the pipeline via
                ``utils_hidream_o1.build_packed_attention_metadata`` and
                passed straight through to every decoder layer.
            pixel_values / image_grid_thw: reference-image conditioning
                (Phase 2+; unused for plain text-to-image).
            precomputed_image_embeds / precomputed_deepstack_image_embeds:
                Phase 2+ vision-tower embedding cache, reused across
                denoising steps and CFG branches instead of recomputing the
                vision tower every call.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)

        cond_image_embeds_out = None
        cond_deepstack_image_embeds_out = None
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if pixel_values is not None:
            if precomputed_image_embeds is not None and precomputed_deepstack_image_embeds is not None:
                image_embeds = precomputed_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_visual_embeds = [
                    e.to(inputs_embeds.device, inputs_embeds.dtype) for e in precomputed_deepstack_image_embeds
                ]
            else:
                image_embeds, deepstack_visual_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_token_id = self.config.image_token_id
            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
            cond_image_embeds_out = image_embeds
            cond_deepstack_image_embeds_out = deepstack_visual_embeds

        # Timestep conditioning enters via token substitution, not AdaLN:
        # every position tagged as the timestep placeholder gets its
        # embedding replaced by t_embedder1(timestep).
        if isinstance(timestep, list):
            timestep = torch.cat(timestep, dim=0)
        timestep = timestep.to(inputs_embeds.device)
        t_emb = self.t_embedder1(timestep)
        tms_mask = (input_ids == self.tms_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask, t_emb.unsqueeze(1).expand_as(inputs_embeds), inputs_embeds)

        if isinstance(vinputs, list):
            vinputs = torch.cat(vinputs, dim=0)
        vinputs = vinputs.to(inputs_embeds.device)
        vinputs_embedded = self.x_embedder(vinputs).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        if visual_pos_masks is not None:
            pad = torch.zeros(
                visual_pos_masks.shape[0],
                vinputs_embedded.shape[1],
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        x_pred = self.final_layer2(hidden_states)
        return HiDreamO1ModelOutput(
            x_pred=x_pred,
            cond_image_embeds=cond_image_embeds_out,
            cond_deepstack_image_embeds=cond_deepstack_image_embeds_out,
        )
