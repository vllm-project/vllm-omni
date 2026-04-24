# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from vllm_omni.diffusion.attention.layer import Attention

try:
    from dreamid_omni.modules.model import (
        ChannelLastConv1d,
        ConvMLP,
        Head,
        MLPProj,
        ModulationAdd,
        WanLayerNorm,
        WanRMSNorm,
        rope_apply,  # diff from  wan2.2
        rope_params,
        sinusoidal_embedding_1d,
    )
except ImportError:
    raise ImportError("Failed to import from dependency 'dreamid_omni'.")


class WanSelfAttention(nn.Module):
    """Optimized self-attention module using vLLM layers."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # Unified attention layer
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, seq_lens, grid_sizes, freqs, ref_lengths=None, freqs_scaling=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ref_lengths(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x)

        x = self.attn(
            rope_apply(q, grid_sizes, freqs, ref_lengths=ref_lengths, freqs_scaling=freqs_scaling),
            rope_apply(k, grid_sizes, freqs, ref_lengths=ref_lengths, freqs_scaling=freqs_scaling),
            v,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context)

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("WanT2VCrossAttention is currently disabled and should not be used.")


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        additional_emb_length=None,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        if cross_attn_type == "i2v_cross_attn":
            assert additional_emb_length is not None, "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, additional_emb_length)
        else:
            assert additional_emb_length is None, "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WanT2VCrossAttention(
                dim,
                num_heads,
                (-1, -1),
                qk_norm,
                eps,
            )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = ModulationAdd(dim, 6)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.bfloat16
        assert len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1], f"{e.shape}, {x.shape}"
        with amp.autocast("cuda", dtype=torch.bfloat16):
            e = self.modulation(e).chunk(6, dim=2)
        assert e[0].dtype == torch.bfloat16

        # self-attention
        y = self.self_attn(
            self.norm1(x).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), seq_lens, grid_sizes, freqs
        )
        with amp.autocast("cuda", dtype=torch.bfloat16):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).bfloat16() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video, text-to-audio.
    """

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        additional_emb_dim=None,
        additional_emb_length=None,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        temporal_rope_scaling_factor=1.0,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in [
            "t2v",
            "i2v",
            "t2a",
            "tt2a",
            "ti2v",
        ]  ## tt2a means text transcript + text description to audio (to support both TTS and T2A
        self.model_type = model_type
        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert is_audio_type ^ is_video_type, "Either audio or video model should be specified"
        if is_audio_type:
            ## audio model
            assert len(patch_size) == 1 and patch_size[0] == 1, (
                "Audio model should only accept 1 dimensional input, and we dont do patchify"
            )

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        self.is_audio_type = is_audio_type
        self.is_video_type = is_video_type
        # embeddings
        if is_audio_type:
            ## hardcoded to MMAudio
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        else:
            self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        ## so i2v and tt2a share the same cross attention while t2v and t2a share the same cross attention
        cross_attn_type = "t2v_cross_attn" if model_type in ["t2v", "t2a", "ti2v"] else "i2v_cross_attn"

        if cross_attn_type == "t2v_cross_attn":
            assert additional_emb_dim is None and additional_emb_length is None, (
                "additional_emb_length should be None for t2v and t2a model"
            )
        else:
            assert additional_emb_dim is not None and additional_emb_length is not None, (
                "additional_emb_length should be specified for i2v and tt2a model"
            )

        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    additional_emb_length,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        self.set_rope_params()

        if model_type in ["i2v", "tt2a"]:
            self.img_emb = MLPProj(additional_emb_dim, dim)

    def set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        if self.is_audio_type:
            ## to be determined
            # self.freqs = rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
            self.freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)
        else:
            self.freqs = torch.cat(
                [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
                dim=1,
            )

    def prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        ref_lengths=None,
        freqs_scaling=None,
    ):
        # params
        ## need to change!
        device = next(self.patch_embedding.parameters()).device

        if self.is_audio_type and freqs_scaling is not None:
            if isinstance(freqs_scaling, torch.Tensor):
                scale_val = freqs_scaling.item()
            else:
                scale_val = freqs_scaling
            d = self.dim // self.num_heads

            current_freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=scale_val).to(device)
        else:
            current_freqs = self.freqs
            if current_freqs.device != device:
                current_freqs = current_freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  ## x is list of [B L D] or [B C F H W]
        if self.is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack([torch.tensor(u.shape[1:2], dtype=torch.long) for u in x])
        else:
            # [B, 3]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]  # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
        )  # single [B, L, C]

        # time embeddings
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with amp.autocast("cuda", dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).bfloat16())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))  # [1, 26784, 6, 3072] - B, seq_len, 6, dim
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=current_freqs,
            context=context,
            context_lens=context_lens,
            ref_lengths=ref_lengths,
            freqs_scaling=freqs_scaling,
        )

        return x, e, kwargs

    def post_transformer_block_out(self, x, grid_sizes, e):
        # head
        x = self.head(x, e)
        # unpatchify
        if self.is_audio_type:
            ## grid_sizes is [B 1] where 1 is L,
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self.unpatchify(x, grid_sizes)

        return [u.bfloat16() for u in x]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("WanModel model does not support forward pass.")

    def unpatchify(self, x, grid_sizes) -> list[torch.Tensor]:
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20.
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out
