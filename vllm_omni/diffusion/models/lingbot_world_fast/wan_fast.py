"""Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing)."""

import logging
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.lingbot_world_fast.state_lingbot_world_fast import CacheIndex
from vllm_omni.platforms import current_omni_platform

from .wan_model import rope_params, sinusoidal_embedding_1d

logger = logging.getLogger(__name__)

# AR-Diffusion paged self-attention (in-tree experimental engine). Import at
# module level so the isinstance check + custom-op call trace cleanly inside
# the fullgraph-compiled DiT block (an import inside the traced region would
# graph-break). The model still works without the engine: the payload type is
# only ever constructed by the AR-Diffusion runner.
try:
    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
        ARDiffusionPagedLayerInputs,
        paged_write_attn,
    )
except ImportError:  # pragma: no cover - experimental package always ships in-tree
    ARDiffusionPagedLayerInputs = None
    paged_write_attn = None


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = 1
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.local_attn_size = local_attn_size
        self.max_attention_size = 1_000_000
        self.num_frame_per_block = 4

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: torch.Tensor | Any | None = None,
        local_end_index=None,
        global_end_index=None,
        current_start=0,
        max_attention_size=1_000_000,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Inference-only forward (KV cache path)."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        updated_kv_cache: torch.Tensor | None = None

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen

        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        num_new_tokens = roped_query.shape[1]
        current_end = current_start + roped_query.shape[1]

        if ARDiffusionPagedLayerInputs is not None and isinstance(kv_cache, ARDiffusionPagedLayerInputs):
            # Fused write+attend custom op: one opaque node in the compiled
            # graph (slot writes + FlashAttention block-table kernel inside).
            # Metadata tensors were prepared once per forward in _forward_blocks.
            q_cat = roped_query

            x = paged_write_attn(
                kv_cache,
                q_cat[0],
                roped_key[0],
                v[0],
                None,
                None,
                self.head_dim**-0.5,
            ).unsqueeze(0)
        else:
            new_local_end_index = local_end_index.item() + current_end - global_end_index.item()
            local_start_index = new_local_end_index - num_new_tokens
            kv_cache[CacheIndex.K][:, local_start_index:new_local_end_index] = roped_key
            kv_cache[CacheIndex.V][:, local_start_index:new_local_end_index] = v

            k_cache = kv_cache[CacheIndex.K][:, max(0, new_local_end_index - max_attention_size) : new_local_end_index]
            v_cache = kv_cache[CacheIndex.V][:, max(0, new_local_end_index - max_attention_size) : new_local_end_index]
            x = self.attn(roped_query, k_cache, v_cache)
            updated_kv_cache = torch.stack([k_cache, v_cache], dim=0)

            global_end_index.fill_(current_end)
            local_end_index.fill_(new_local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class WanCrossAttention(nn.Module):
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
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        del context_lens
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        # if crossattn_cache is not None:
        #     if not crossattn_cache["is_init"]:
        #         crossattn_cache["is_init"] = True
        #         k = self.norm_k(self.k(context)).unflatten(2, (n, d))
        #         v = self.v(context).unflatten(2, (n, d))
        #         crossattn_cache[CacheIndex.K] = k
        #         crossattn_cache[CacheIndex.V] = v
        #     else:
        #         k = crossattn_cache[CacheIndex.K]
        #         v = crossattn_cache[CacheIndex.V]
        # else:
        #     k = self.norm_k(self.k(context)).unflatten(2, (n, d))
        #     v = self.v(context).unflatten(2, (n, d))

        k = crossattn_cache["k"]
        v = crossattn_cache["v"]

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = CausalWanSelfAttention(
            dim=dim, num_heads=num_heads, local_attn_size=local_attn_size, sink_size=sink_size, qk_norm=qk_norm, eps=eps
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        local_end_index=None,
        global_end_index=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        # self-attention
        y, updated_kv_cache = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            grid_sizes,
            freqs,
            kv_cache,
            local_end_index,
            global_end_index,
            current_start,
            max_attention_size,
        )
        with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cam injection (only if dit_cond_dict is provided and contains c2ws_plucker_emb)
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_hidden_states = self.cam_injector_layer2(torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
            c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden_states)
            cam_shift = self.cam_shift_layer(c2ws_hidden_states)
            x = (1.0 + cam_scale) * x + cam_shift

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x, updated_kv_cache


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModelFast(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="",
        control_type="cam",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        num_frame_per_block=4,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to ''):
            control_type (`str`, *optional*, defaults to 'cam'):
               Type of conditioning control signal - 'cam' (6-dim camera Plucker
               embeddings) or 'act' (7-dim action embeddings including WASD movement)
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
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.model_type = model_type

        self.task_type = "t2v"
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
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.frame_seqlen = 1296
        self.local_branches = 1

        if control_type == "cam":
            control_dim = 6
        elif control_type == "act":
            control_dim = 7

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim
        )
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        kv_cache=None,
        local_end_index=None,
        global_end_index=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        **kwargs,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            dit_cond_dict (`dict`, *optional*, defaults to None):
                Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
            kv_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, kv_size, num_heads, head_dim]), ``global_end_index``,
                and ``local_end_index`` (scalar Tensors tracking cache position).
            crossattn_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current chunk in the full sequence. Used to index
                into the KV cache and compute positional embeddings correctly.
            max_attention_size (`int`, *optional*, defaults to 1_000_000):
                Maximum number of KV tokens each query can attend to. Limits the
                effective context window of self-attention to control memory usage.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.task_type == "i2v":
            assert y is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_lens)
        with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_lens)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        # cam
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_plucker_emb = [
                rearrange(
                    i,
                    "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                )
                for i in c2ws_plucker_emb
            ]
            c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)  # [1, (L1+...+Ln), C]

            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))
            )
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb + c2ws_hidden_states

        if kv_cache and getattr(kv_cache[0], "is_ar_diffusion_paged_context", False):
            fctx = kv_cache[0].forward_ctx
            if seq_len != fctx.seq_len:
                raise RuntimeError(
                    f"AR-Diffusion paged context seq_len={fctx.seq_len} but current video KV has {seq_len} tokens"
                )
            fctx.prepare(
                device=x.device,
                action_len=0,
                query_len=int(x.shape[1]),
            )
            kv_cache = [c.to_layer_inputs() for c in kv_cache]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
            max_attention_size=max_attention_size,
        )

        updated_kv_caches: list[torch.Tensor | None] = []
        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "local_end_index": local_end_index[block_index],
                    "global_end_index": global_end_index[block_index],
                    "current_start": current_start,
                }
            )
            x, updated_kv_cache = block(x, **kwargs)
            updated_kv_caches.append(updated_kv_cache)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x], updated_kv_caches

    def unpatchify(self, x, grid_sizes):
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
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
