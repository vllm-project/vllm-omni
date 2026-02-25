from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO replace this with vLLM implementation
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states):
        hidden_states, _ = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        else:
            raise ValueError(f"Unsupported activation function type: {activation_fn}")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(RowParallelLinear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for layer in self.net:
            output = layer(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states


class SD3PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for SD3.

    Args:
        patch_size (`int`, defaults to `16`): The size of the patches.
        in_channels (`int`, defaults to `3`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
    """

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=True
        )

    def forward(self, latent):
        x = self.proj(latent)  # [B, embed_dim, patch_size, patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class SD3CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,  # query_dim
        num_heads: int,
        head_dim: int,
        added_kv_proj_dim: int = 0,
        out_bias: bool = True,
        qk_norm=True,  # rmsnorm
        eps=1e-6,
        pre_only=False,
        context_pre_only: bool = False,
        parallel_attention=False,
        out_dim: int = 0,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
        )
        self.norm_q = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.inner_dim = out_dim if out_dim is not None else head_dim * num_heads
        self.inner_kv_dim = self.inner_dim
        if added_kv_proj_dim is not None:
            self.add_kv_proj = QKVParallelLinear(
                added_kv_proj_dim,
                head_size=self.inner_kv_dim // self.num_heads,
                total_num_heads=self.num_heads,
            )
        else:
            self.add_kv_proj = None

        if not context_pre_only:
            self.to_add_out = RowParallelLinear(self.inner_dim, self.dim, bias=out_bias)
        else:
            self.to_add_out = None

        if not pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(RowParallelLinear(self.inner_dim, self.dim, bias=out_bias))
        else:
            self.to_out = None

        self.norm_added_q = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_added_k = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ):
        # Compute QKV for image stream (sample projections)
        qkv = self.to_qkv(hidden_states)
        qkv = qkv[0]
        img_query, img_key, img_value = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        local_num_heads = self.to_qkv.num_heads
        img_query = img_query.unflatten(-1, (local_num_heads, -1))
        img_key = img_key.unflatten(-1, (local_num_heads, -1))
        img_value = img_value.unflatten(-1, (local_num_heads, -1))

        # Apply QK normalization
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)

        if encoder_hidden_states is not None:
            # Compute QKV for text stream (context projections)
            qkv_add = self.add_kv_proj(encoder_hidden_states)
            qkv_add = qkv_add[0]
            txt_query, txt_key, txt_value = qkv_add.chunk(3, dim=-1)

            txt_query = txt_query.unflatten(-1, (local_num_heads, -1))
            txt_key = txt_key.unflatten(-1, (local_num_heads, -1))
            txt_value = txt_value.unflatten(-1, (local_num_heads, -1))

            txt_query = self.norm_added_q(txt_query)
            txt_key = self.norm_added_k(txt_key)

            # Concatenate for joint attention
            # Order: [text, image]
            query = torch.cat([txt_query, img_query], dim=1)
            key = torch.cat([txt_key, img_key], dim=1)
            value = torch.cat([txt_value, img_value], dim=1)
        else:
            query = img_query
            key = img_key
            value = img_value

        hidden_states = self.attn(
            query,
            key,
            value,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split attention outputs back
            context_seqlen = encoder_hidden_states.shape[1]
            hidden_states, encoder_hidden_states = (
                hidden_states[:, context_seqlen:, :],  # Image part
                hidden_states[:, :context_seqlen, :],  # Text part
            )
            if self.to_add_out is not None:
                encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)

        # Apply output projections
        if self.to_out is not None:
            hidden_states, _ = self.to_out[0](hidden_states)

        if encoder_hidden_states is None:
            return hidden_states
        else:
            return hidden_states, encoder_hidden_states


class SD3TransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://huggingface.co/papers/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
        use_dual_attention: bool = False,
    ):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continuous" if context_pre_only else "ada_norm_zero"

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continuous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently "
                f"only support `ada_norm_continuous`, `ada_norm_zero`"
            )

        self.attn = SD3CrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            added_kv_proj_dim=dim,
            context_pre_only=context_pre_only,
            out_dim=dim,
            qk_norm=True if qk_norm == "rms_norm" else False,
            eps=1e-6,
        )

        if use_dual_attention:
            self.attn2 = SD3CrossAttention(
                dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                out_dim=dim,
                qk_norm=True if qk_norm == "rms_norm" else False,
                eps=1e-6,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states = encoder_hidden_states[0]
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


class SD3Transformer2DModel(nn.Module):
    """
    The Transformer model introduced in [Stable Diffusion 3](https://huggingface.co/papers/2403.03206).
    """

    _repeated_blocks = ["SD3TransformerBlock"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
    ):
        super().__init__()
        model_config = od_config.tf_model_config
        self.num_layers = model_config.num_layers
        self.parallel_config = od_config.parallel_config
        self.sample_size = model_config.sample_size
        self.in_channels = model_config.in_channels
        self.out_channels = model_config.out_channels
        self.num_attention_heads = model_config.num_attention_heads
        self.attention_head_dim = model_config.attention_head_dim
        self.inner_dim = model_config.num_attention_heads * model_config.attention_head_dim
        self.caption_projection_dim = model_config.caption_projection_dim
        self.pooled_projection_dim = model_config.pooled_projection_dim
        self.joint_attention_dim = model_config.joint_attention_dim
        self.patch_size = model_config.patch_size
        self.dual_attention_layers = (
            model_config.dual_attention_layers if hasattr(model_config, "dual_attention_layers") else ()
        )
        self.qk_norm = model_config.qk_norm if hasattr(model_config, "qk_norm") else ""
        self.pos_embed_max_size = model_config.pos_embed_max_size

        self.pos_embed = PatchEmbed(
            height=self.sample_size,
            width=self.sample_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=self.pos_embed_max_size,
        )

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.pooled_projection_dim
        )
        self.context_embedder = ReplicatedLinear(self.joint_attention_dim, self.caption_projection_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SD3TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    context_pre_only=i == self.num_layers - 1,
                    qk_norm=self.qk_norm,
                    use_dual_attention=True if i in self.dual_attention_layers else False,
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = ReplicatedLinear(
            self.inner_dim, self.patch_size * self.patch_size * self.out_channels, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".pos_embed"):
                params_dict[name] = buffer

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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
