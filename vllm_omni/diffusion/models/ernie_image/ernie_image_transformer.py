# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ernie_image.py

from collections.abc import Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def _get_sequence_parallel_world_size_or_one() -> int:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size

        return max(1, int(get_sequence_parallel_world_size()))
    except Exception:
        return 1


def _get_ring_parallel_info() -> tuple[int, int]:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_ring_parallel_rank,
            get_ring_parallel_world_size,
        )

        return max(1, int(get_ring_parallel_world_size())), int(get_ring_parallel_rank())
    except Exception:
        return 1, 0


def validate_ernie_image_tp_constraints(*, heads: int, tensor_parallel_size: int) -> int:
    """Validate ErnieImage TP constraints without requiring a distributed context.

    Args:
        heads: Number of attention heads
        tensor_parallel_size: Tensor parallel size

    Returns:
        Number of heads per GPU after TP sharding

    Raises:
        ValueError: If heads is not divisible by tensor_parallel_size
    """
    tp_size = int(tensor_parallel_size)
    if tp_size <= 0:
        raise ValueError(f"tensor_parallel_size must be > 0, got {tp_size}")
    if heads % tp_size != 0:
        raise ValueError(f"num_attention_heads ({heads}) must be divisible by tensor_parallel_size ({tp_size})")
    return heads // tp_size


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return out.float()


def _apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    cos_ = freqs_cos.unsqueeze(2).to(x.dtype)
    sin_ = freqs_sin.unsqueeze(2).to(x.dtype)
    cos_ = cos_.repeat_interleave(2, dim=-1)
    sin_ = sin_.repeat_interleave(2, dim=-1)
    rot_dim = cos_.shape[-1]
    x_rot = x[..., :rot_dim]
    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * cos_ + x_rotated * sin_, x[..., rot_dim:]), dim=-1)


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(3):
            emb = rope(pos[..., i], self.axes_dim[i], self.theta)
            cos_out.append(emb.cos())
            sin_out.append(emb.sin())
        freqs_cos = torch.cat(cos_out, dim=-1)
        freqs_sin = torch.cat(sin_out, dim=-1)
        return freqs_cos, freqs_sin


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()


class UnifiedPrepare(nn.Module):
    """Prepares hidden_states, RoPE embeddings, and attention mask for sequence parallel.

    This module encapsulates the input projection, RoPE, and attention_mask computation.
    This creates a module boundary where _sp_plan can shard outputs via split_output=True.

    The key insight is that hidden_states, freqs_cos, and freqs_sin
    must be sharded together to maintain dimension alignment for attention layers.

    Note: Our _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism).
    """

    def __init__(
        self,
        x_embedder: nn.Module,
        text_proj: nn.Module | None,
        pos_embed: ErnieImageEmbedND3,
    ):
        super().__init__()
        self.x_embedder = x_embedder
        self.text_proj = text_proj
        self.pos_embed = pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare hidden_states, RoPE embeddings, and attention mask.

        Args:
            hidden_states: Image latent [B, C, H, W]
            text_bth: Text embeddings [B, Tmax, text_in_dim]
            text_lens: Text sequence lengths [B]

        Returns:
            hidden_states: [S, B, dim] where S = N_img + Tmax
            freqs_cos: [B, seq, head_dim] cosine component of RoPE
            freqs_sin: [B, seq, head_dim] sine component of RoPE
            attention_mask: [B, seq] boolean mask
        """
        device = hidden_states.device
        B, C, H, W = hidden_states.shape
        Hp, Wp = H, W
        N_img = Hp * Wp

        img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]
        text_sbh = text_bth.transpose(0, 1).contiguous()

        hidden_states = torch.cat([img_sbh, text_sbh], dim=0)

        text_ids = (
            torch.cat(
                [
                    torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
                    torch.zeros((B, Tmax, 2), device=device),
                ],
                dim=-1,
            )
            if Tmax > 0
            else torch.zeros((B, 0, 3), device=device)
        )
        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(Hp, device=device, dtype=torch.float32),
                torch.arange(Wp, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        image_ids = torch.cat(
            [text_lens.float().view(B, 1, 1).expand(-1, N_img, -1), grid_yx.view(1, N_img, 2).expand(B, -1, -1)],
            dim=-1,
        )
        freqs_cos, freqs_sin = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        valid_text = (
            torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
            if Tmax > 0
            else torch.zeros((B, 0), device=device, dtype=torch.bool)
        )
        attention_mask = torch.cat([torch.ones((B, N_img), device=device, dtype=torch.bool), valid_text], dim=1)
        sp_size = _get_sequence_parallel_world_size_or_one()
        pad_size = (-hidden_states.shape[0]) % sp_size
        if pad_size:
            pad_hidden = torch.zeros((pad_size, B, hidden_states.shape[-1]), device=device, dtype=hidden_states.dtype)
            pad_freq = torch.zeros((B, pad_size, freqs_cos.shape[-1]), device=device, dtype=freqs_cos.dtype)
            pad_mask = torch.zeros((B, pad_size), device=device, dtype=attention_mask.dtype)
            hidden_states = torch.cat([hidden_states, pad_hidden], dim=0)
            freqs_cos = torch.cat([freqs_cos, pad_freq], dim=1)
            freqs_sin = torch.cat([freqs_sin, pad_freq], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        return hidden_states, freqs_cos, freqs_sin, attention_mask


class ErnieImageAttention(nn.Module):
    def __init__(
        self,
        parallel_config: DiffusionParallelConfig,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: str = "rms_norm",
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.parallel_config = parallel_config
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        tp_size = get_tensor_model_parallel_world_size()
        if heads % tp_size != 0:
            raise ValueError(f"num_attention_heads ({heads}) must be divisible by tensor_parallel_size ({tp_size})")
        self.heads = heads // tp_size
        self.dropout = dropout

        self.to_q = ColumnParallelLinear(
            input_size=query_dim,
            output_size=self.inner_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
        )
        self.to_k = ColumnParallelLinear(
            input_size=query_dim,
            output_size=self.inner_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
        )
        self.to_v = ColumnParallelLinear(
            input_size=query_dim,
            output_size=self.inner_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
        )

        self.kv_num_heads = self.heads

        if qk_norm is not None:
            if qk_norm == "layer_norm":
                self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
                self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            elif qk_norm == "rms_norm":
                self.norm_q = RMSNorm(dim_head, eps=eps)
                self.norm_k = RMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}")

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                ),
                nn.Dropout(dropout),
            ]
        )

        self.attn = Attention(
            num_heads=self.heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.kv_num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        if hasattr(self, "norm_q"):
            query = self.norm_q(query)
            key = self.norm_k(key)

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            query = _apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = _apply_rotary_emb(key, freqs_cos, freqs_sin)

        attn_metadata = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        hidden_states = self.to_out[0](hidden_states.contiguous())
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class ErnieImageFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        bias: bool = False,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
        )
        self.linear_fc2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out, _ = self.gate_proj(x)
        up_out, _ = self.up_proj(x)
        return self.linear_fc2(up_out * torch.nn.functional.gelu(gate_out))


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(
        self,
        parallel_config: DiffusionParallelConfig,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageAttention(
            parallel_config=parallel_config,
            query_dim=hidden_size,
            dim_head=hidden_size // num_heads,
            heads=num_heads,
            qk_norm="rms_norm" if qk_layernorm else None,
            eps=eps,
            bias=False,
            out_bias=False,
            quant_config=quant_config,
        )
        self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size, quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        temb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb
        residual = hidden_states
        hidden_states = self.adaLN_sa_ln(hidden_states)
        hidden_states = (hidden_states.float() * (1 + scale_msa.float()) + shift_msa.float()).to(hidden_states.dtype)
        hidden_states_bsh = hidden_states.permute(1, 0, 2)

        attn_out = self.self_attention(
            hidden_states_bsh, attention_mask=attention_mask, image_rotary_emb=rotary_pos_emb
        )
        attn_out = attn_out.permute(1, 0, 2)
        hidden_states = residual + (gate_msa.float() * attn_out.float()).to(hidden_states.dtype)
        residual = hidden_states
        hidden_states = self.adaLN_mlp_ln(hidden_states)
        hidden_states = (hidden_states.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(hidden_states.dtype)
        return residual + (gate_mlp.float() * self.mlp(hidden_states).float()).to(hidden_states.dtype)


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class ErnieImageTransformer2DModel(nn.Module):
    _repeated_blocks = ["ErnieImageSharedAdaLNBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]

    @staticmethod
    def _is_layer_block(name: str, module) -> bool:
        return "layers" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_layer_block]

    _sp_plan = {
        "unified_prepare": {
            0: SequenceParallelInput(split_dim=0, expected_dims=3, split_output=True, auto_pad=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
        },
        "final_linear": SequenceParallelOutput(gather_dim=0, expected_dims=3),
    }

    @staticmethod
    def _slice_attention_mask_for_ring(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        ring_size, ring_rank = _get_ring_parallel_info()
        if ring_size <= 1:
            return attention_mask
        seq_len = attention_mask.shape[1]
        if seq_len % ring_size != 0:
            raise ValueError(
                "ERNIE-Image hybrid SP requires attention_mask length to be divisible by ring_degree, "
                f"but got seq_len={seq_len}, ring_degree={ring_size}."
            )
        return attention_mask.chunk(ring_size, dim=1)[ring_rank].contiguous()

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 24,
        num_attention_heads: int = 24,
        ffn_hidden_size: int = 8192,
        hidden_size: int = 3072,
        text_in_dim: int = 2560,
        rope_theta: int = 256,
        rope_axes_dim: tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        od_config: OmniDiffusionConfig = None,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.text_in_dim = text_in_dim

        if od_config is not None:
            self.parallel_config = od_config.parallel_config
        else:
            from vllm_omni.diffusion.data import DiffusionParallelConfig

            self.parallel_config = DiffusionParallelConfig()

        self.config = SimpleNamespace(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.out_channels,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            hidden_size=hidden_size,
            text_in_dim=text_in_dim,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            eps=eps,
            qk_layernorm=qk_layernorm,
        )

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.pos_embed = ErnieImageEmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.unified_prepare = UnifiedPrepare(self.x_embedder, self.text_proj, self.pos_embed)
        self.layers = nn.ModuleList(
            [
                ErnieImageSharedAdaLNBlock(
                    parallel_config=self.parallel_config,
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    eps=eps,
                    qk_layernorm=qk_layernorm,
                    quant_config=quant_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        dtype = hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p

        N_img = Hp * Wp
        hidden_states, freqs_cos, freqs_sin, attention_mask = self.unified_prepare(hidden_states, text_bth, text_lens)
        attention_mask = self._slice_attention_mask_for_ring(attention_mask)
        rotary_pos_emb = (freqs_cos, freqs_sin)
        S = hidden_states.shape[0]

        sample = self.time_proj(timestep)
        sample = sample.to(dtype=dtype)
        c = self.time_embedding(sample)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.unsqueeze(0).expand(S, -1, -1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]

        for layer in self.layers:
            temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    rotary_pos_emb,
                    temb,
                    attention_mask,
                )
            else:
                hidden_states = layer(hidden_states, rotary_pos_emb, temb, attention_mask)
        hidden_states = self.final_norm(hidden_states, c).type_as(hidden_states)
        patches = self.final_linear(hidden_states)[:N_img].transpose(0, 1).contiguous()
        output = (
            patches.view(B, Hp, Wp, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, self.out_channels, H, W)
        )

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace("transformer.", "", 1)
            if name not in params_dict and ".to_out.0." in name:
                name = name.replace(".to_out.0.", ".to_out.")
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
