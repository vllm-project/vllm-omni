import math

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .rope_real import RotaryPosEmbedReal

try:
    from transformers.modeling_flash_attention_utils import (  # type: ignore
        flash_attn_varlen_func,  # pyright: ignore[reportAttributeAccessIssue]
        is_flash_attn_available,
    )
except Exception:  # pragma: no cover - best-effort compatibility
    flash_attn_varlen_func = None  # type: ignore[assignment]

    def is_flash_attn_available() -> bool:  # type: ignore[override]
        return False


from .rope_real import apply_real_rotary_emb

_HAS_FLASH_ATTN_VARLEN = bool(is_flash_attn_available()) and flash_attn_varlen_func is not None


class LuminaRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )

        self.norm = Qwen2RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int | None = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()

        # custom hidden_size factor multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )

    def swiglu(self, x, y):
        return F.silu(x.float(), inplace=False).to(x.dtype) * y

    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        return self.linear_2(self.swiglu(h1, h2))


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: int | None = None,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = Qwen2RMSNorm(embedding_dim, eps=eps)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2520,
        text_feat_dim: int = 3584,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=timestep_scale
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(text_feat_dim, eps=norm_eps),
            nn.Linear(text_feat_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(text_hidden_states)
        return time_embed, caption_embed


class SimpleQFormerImageRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 128,
        num_layers: int = 2,
        num_heads: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        # ensure num_heads divides hidden_size
        if num_heads is None:
            num_heads = max(1, hidden_size // 128)
        self.num_heads = self._choose_valid_num_heads(hidden_size, num_heads)
        self.input_proj = nn.Sequential(
            Qwen2RMSNorm(hidden_size, eps=norm_eps),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # Learnable query embeddings
        scale = hidden_size**-0.5
        self.query = nn.Parameter(scale * torch.randn(1, num_queries, hidden_size))

        # Decoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    dict(
                        ln_q1=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        self_attn=nn.MultiheadAttention(
                            embed_dim=hidden_size, num_heads=self.num_heads, dropout=dropout, batch_first=True
                        ),
                        ln_q2=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        cross_attn=nn.MultiheadAttention(
                            embed_dim=hidden_size, num_heads=self.num_heads, dropout=dropout, batch_first=True
                        ),
                        ln_ffn=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        ffn=LuminaFeedForward(dim=hidden_size, inner_dim=4 * hidden_size),
                    )
                )
            )

    @staticmethod
    def _choose_valid_num_heads(hidden_size: int, proposed_heads: int, preferred_head_dim: int = 128) -> int:
        """Pick a number of heads that divides hidden_size, close to proposed or preferred."""
        # If proposed is valid, use it
        if proposed_heads > 0 and hidden_size % proposed_heads == 0:
            return proposed_heads
        # target based on preferred head dim
        target = max(1, round(hidden_size / preferred_head_dim))
        # collect divisors up to 128 heads (more than enough)
        max_heads_cap = min(128, hidden_size)
        divisors = [d for d in range(1, max_heads_cap + 1) if hidden_size % d == 0]
        # choose closest to target
        best = min(divisors, key=lambda d: (abs(d - target), -d))
        return best

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, num_queries, hidden_size)
        """
        batch, _, _ = x.shape
        kv = self.input_proj(x)
        q = self.query.repeat(batch, 1, 1).to(kv.dtype)

        for layer in self.layers:
            # Self-attention on queries
            q_norm = layer["ln_q1"](q)
            attn_out, _ = layer["self_attn"](q_norm, q_norm, q_norm, need_weights=False)
            q = q + attn_out

            # Cross-attention: queries attend to inputs
            q_norm = layer["ln_q2"](q)
            cross_out, _ = layer["cross_attn"](q_norm, kv, kv, need_weights=False, key_padding_mask=attention_mask)
            q = q + cross_out

            # Feed-forward
            q = q + layer["ffn"](layer["ln_ffn"](q))

        return q


class AttnProcessor:
    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0+ (F.scaled_dot_product_attention).")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_real_rotary_emb(query, image_rotary_emb[0], image_rotary_emb[1])
            key = apply_real_rotary_emb(key, image_rotary_emb[0], image_rotary_emb[1])

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        if _HAS_FLASH_ATTN_VARLEN and attention_mask is not None and hidden_states.is_cuda:
            # Flash-Attn varlen expects packed tokens + cu_seqlens. Here we only need
            # the self-attention case (q/k/v share the same padding mask).
            attention_mask = attention_mask.to(torch.bool)
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen = int(seqlens.max().item())
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

            query_states = query.reshape(batch_size * sequence_length, attn.heads, head_dim)[indices]
            key_states = key.reshape(batch_size * sequence_length, kv_heads, head_dim)[indices]
            value_states = value.reshape(batch_size * sequence_length, kv_heads, head_dim)[indices]

            if kv_heads < attn.heads:
                key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
                value_states = repeat(value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )

            out = torch.zeros(
                (batch_size * sequence_length, attn.heads, head_dim),
                device=attn_output_unpad.device,
                dtype=attn_output_unpad.dtype,
            )
            out[indices] = attn_output_unpad
            hidden_states = out.view(batch_size, sequence_length, attn.heads, head_dim).flatten(-2)
            hidden_states = hidden_states.type_as(query)
        else:
            # PyTorch SDPA path.
            attn_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
                attn_mask = attention_mask.view(batch_size, 1, 1, -1)

            query = query.transpose(1, 2)  # [B, H, S, D]
            key = key.transpose(1, 2)  # [B, H_kv, S, D]
            value = value.transpose(1, 2)

            if kv_heads < attn.heads:
                key = key.repeat_interleave(attn.heads // kv_heads, dim=1)
                value = value.repeat_interleave(attn.heads // kv_heads, dim=1)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )

            if attention_mask is not None:
                # Keep padding tokens consistent with the flash-varlen path (zero output).
                hidden_states = hidden_states * attention_mask[:, None, :, None]

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        processor = AttnProcessor()

        # Initialize attention layer
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm=None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )
        # 显式使用 transformers 的 Qwen2RMSNorm，避免依赖 diffusers 内部创建的 `RMSNorm` 再做递归替换。
        self.attn.norm_q = Qwen2RMSNorm(self.head_dim, eps=1e-5)
        self.attn.norm_k = Qwen2RMSNorm(self.head_dim, eps=1e-5)

        # Initialize feed-forward network
        self.feed_forward = LuminaFeedForward(
            dim=dim, inner_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.norm1 = Qwen2RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = Qwen2RMSNorm(dim, eps=norm_eps)
        self.norm2 = Qwen2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = Qwen2RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")

            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class Transformer2DModel(ModelMixin, ConfigMixin):
    """MammothModa2 DiT transformer"""

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: tuple[int, int, int] = (32, 32, 32),
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        text_feat_dim: int = 1024,
        timestep_scale: float = 1.0,
    ) -> None:
        """Initialize the  transformer model."""
        super().__init__()
        self.hidden_size = hidden_size

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        self.out_channels = out_channels or in_channels
        # Initialize embeddings
        self.rope_embedder = RotaryPosEmbedReal(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            text_feat_dim=text_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        # Initialize transformer blocks
        self.noise_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # 3. Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Add learnable embeddings to distinguish different images
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))  # support max 5 ref images

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None,
        return_dict: bool,
    ) -> tuple[int, int, int]:
        if return_dict:
            raise ValueError("return_dict=True is not supported in vLLM inference.")
        if ref_image_hidden_states is not None:
            raise ValueError("ref_image_hidden_states is not supported in vLLM inference.")
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected hidden_states to be 4D [B,C,H,W], got shape={tuple(hidden_states.shape)}")

        batch_size, _channels, height, width = hidden_states.shape
        if batch_size != text_hidden_states.shape[0] or batch_size != text_attention_mask.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"hidden_states={batch_size}, text_hidden_states={text_hidden_states.shape[0]}, "
                f"text_attention_mask={text_attention_mask.shape[0]}"
            )

        p = self.config.patch_size
        if height % p != 0 or width % p != 0:
            raise ValueError(f"Input latent H/W must be divisible by patch_size={p}, got {height}x{width}")
        return batch_size, height, width

    def _prepare_embeddings(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ):
        device = hidden_states.device
        p = self.config.patch_size

        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states.dtype)

        img_tokens = rearrange(hidden_states, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        img_tokens = self.x_embedder(img_tokens)

        img_len = (height // p) * (width // p)
        img_mask = torch.ones((batch_size, img_len), dtype=torch.bool, device=device)
        l_effective_img_len = [img_len for _ in range(batch_size)]
        img_sizes = [(height, width) for _ in range(batch_size)]

        l_effective_ref_img_len = [[] for _ in range(batch_size)]
        ref_img_sizes = [None for _ in range(batch_size)]

        (
            context_rotary_emb,
            _ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            text_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        return (
            temb,
            text_hidden_states,
            img_tokens,
            img_mask,
            img_len,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        )

    def _apply_refiners(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        context_rotary_emb: torch.Tensor,
        img_tokens: torch.Tensor,
        img_mask: torch.Tensor,
        noise_rotary_emb: torch.Tensor,
        temb: torch.Tensor,
    ):
        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

        for layer in self.noise_refiner:
            img_tokens = layer(img_tokens, img_mask, noise_rotary_emb, temb)

        return text_hidden_states, img_tokens

    def _apply_transformer_layers(self, hidden_states, attention_mask, rotary_emb, temb):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        batch_size, height, width = self._validate_inputs(
            hidden_states, text_hidden_states, text_attention_mask, ref_image_hidden_states, return_dict
        )

        (
            temb,
            text_hidden_states,
            img_tokens,
            img_mask,
            img_len,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self._prepare_embeddings(
            hidden_states,
            timestep,
            text_hidden_states,
            text_attention_mask,
            freqs_cis,
            batch_size,
            height,
            width,
        )

        text_hidden_states, img_tokens = self._apply_refiners(
            text_hidden_states,
            text_attention_mask,
            context_rotary_emb,
            img_tokens,
            img_mask,
            noise_rotary_emb,
            temb,
        )

        max_seq_len = max(seq_lengths)
        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len : encoder_seq_len + img_len] = img_tokens[i, :img_len]

        hidden_states = self._apply_transformer_layers(joint_hidden_states, attention_mask, rotary_emb, temb)

        hidden_states = self.norm_out(hidden_states, temb)

        p = self.config.patch_size
        img_hidden_states = torch.stack(
            [
                hidden_states[i, encoder_seq_len : encoder_seq_len + img_len]
                for i, encoder_seq_len in enumerate(encoder_seq_lengths)
            ],
            dim=0,
        )
        output = rearrange(
            img_hidden_states,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=height // p,
            w=width // p,
            p1=p,
            p2=p,
        )
        return output
