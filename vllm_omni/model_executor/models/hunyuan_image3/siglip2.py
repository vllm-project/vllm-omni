# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SigLIP2 Vision Transformer for HunyuanImage3, rewritten in vLLM style.

Key optimizations over the original HuggingFace-style implementation:
- QKVParallelLinear: fused QKV projection with tensor parallelism support
- MMEncoderAttention: FlashAttention / xFormers backend
- ColumnParallelLinear / RowParallelLinear: TP-aware MLP layers
- Packed sequence processing: eliminates padding waste via cu_seqlens
- Data parallel support for multi-GPU ViT inference
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.vision import is_vit_use_data_parallel


class Config:
    """Convert dict config to object with attribute access."""

    def __init__(self, config):
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Process packed pixel values with per-image position embeddings.

        Args:
            pixel_values: Packed pixel values
                (total_real_patches, num_channels * patch_size * patch_size)
            spatial_shapes: Per-image spatial shapes (B, 2) as [(h, w), ...]

        Returns:
            Packed embeddings (total_real_patches, embed_dim)
        """
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Resize position embeddings per image and concatenate (packed)
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        # (H_pe, W_pe, embed_dim) → (1, embed_dim, H_pe, W_pe) for interpolation
        pe_for_resize = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        # Upcast on CPU since antialias is not supported for bfloat16/float16
        if pe_for_resize.device.type == "cpu":
            pe_for_resize = pe_for_resize.to(torch.float32)

        position_embs: list[torch.Tensor] = []
        for i in range(spatial_shapes.shape[0]):
            height, width = int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1])
            resized = F.interpolate(
                pe_for_resize,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # (1, embed_dim, h, w) → (h*w, embed_dim)
            resized = resized.reshape(self.embed_dim, height * width).transpose(0, 1)
            position_embs.append(resized.to(target_dtype))

        packed_position_embs = torch.cat(position_embs, dim=0)
        return patch_embeds + packed_position_embs


class Siglip2Attention(nn.Module):
    """Multi-headed attention using QKVParallelLinear and MMEncoderAttention."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        use_data_parallel = is_vit_use_data_parallel()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )

        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.scale,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Packed input (total_tokens, embed_dim)
            cu_seqlens: Cumulative sequence lengths (B+1,)
        """
        seq_length = hidden_states.shape[0]

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(seq_length, self.num_heads_per_partition, self.head_dim)
        k = k.view(seq_length, self.num_heads_per_partition, self.head_dim)
        v = v.view(seq_length, self.num_heads_per_partition, self.head_dim)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output = self.attn(
            query=q.unsqueeze(0),
            key=k.unsqueeze(0),
            value=v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_output = attn_output.reshape(seq_length, self.num_heads_per_partition * self.head_dim)

        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class Siglip2MLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Packed input (total_tokens, embed_dim)
            cu_seqlens: Cumulative sequence lengths (B+1,)
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    """Transformer encoder with packed sequence processing."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Packed input (total_tokens, embed_dim)
            cu_seqlens: Cumulative sequence lengths (B+1,)
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = Config(config)
        self.config = config
        self.embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder" if prefix else "encoder",
        )
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Batched pixel values
                (B, max_num_patches, num_channels * patch_size * patch_size)
            attention_mask: (B, max_num_patches) with 1 for real, 0 for padding
            spatial_shapes: (B, 2) with (height, width) per image

        Returns:
            (B, max_num_patches, hidden_size) with zeros at padding positions
        """
        batch_size, max_patches, _ = pixel_values.shape

        # Pack: extract real tokens using attention_mask
        mask_bool = attention_mask.bool()
        packed_pixels = pixel_values[mask_bool]  # (total_real_patches, C*P*P)

        # Compute cu_seqlens from spatial_shapes (h * w per image)
        seq_lens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(torch.int32)
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=pixel_values.device)
        cu_seqlens[1:] = seq_lens.cumsum(0)

        # Embeddings (packed)
        hidden_states = self.embeddings(packed_pixels, spatial_shapes)

        # Encoder (packed)
        hidden_states = self.encoder(hidden_states, cu_seqlens)

        # Post layernorm
        hidden_states = self.post_layernorm(hidden_states)

        # Unpack: scatter back to (B, max_patches, hidden_size)
        output = torch.zeros(
            batch_size,
            max_patches,
            self.embed_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output[mask_bool] = hidden_states

        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    # Skip weights for removed modules (e.g., pooling head)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LightProjector(nn.Module):
    def __init__(self, config):
        config = Config(config)
        super().__init__()

        if config.projector_type == "linear":
            modules = nn.Linear(config.input_dim, config.n_embed)

        elif config.projector_type == "mlp_gelu":
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        self.layers = modules

    def forward(self, x):
        return self.layers(x)


__all__ = [
    "Siglip2VisionTransformer",
    "LightProjector",
]
