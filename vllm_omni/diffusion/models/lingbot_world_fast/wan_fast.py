"""Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

from .state_lingbot_world_fast import CacheIndex
from .wan_model import WanLayerNorm, WanRMSNorm, WanSelfAttention, rope_params, sinusoidal_embedding_1d


def causal_rope_apply(x, grid_sizes, freqs, start_frames=0):
    """Apply causal rotary position embedding per batch row.

    start_frames: int or list[int] of per-row frame offsets. 
    An int broadcasts to all rows.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    if isinstance(start_frames, int):
        start_frames = [start_frames] * grid_sizes.shape[0]

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        sf = start_frames[i]
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][sf : sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
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
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        local_end_index=None,
        global_end_index=None,
        current_starts=0,
        max_attention_size=1_000_000,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            current_starts(int | list[int]): per-row absolute token offset; an int
                broadcasts to all rows.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if isinstance(current_starts, int):
            current_starts = [current_starts] * b
        assert len(current_starts) == b

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        start_frames = [cs // frame_seqlen for cs in current_starts]
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frames=start_frames).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frames=start_frames).type_as(v)
        num_new_tokens = roped_query.shape[1]

        if self.local_attn_size != -1:
            # Cache-rolling path only supports single-row processing.
            assert b == 1, "local_attn_size != -1 requires batch_size=1"
            current_start = current_starts[0]
            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache[CacheIndex.K].shape[1]

            if (current_end > global_end_index.item()) and (
                num_new_tokens + local_end_index.item() > kv_cache_size
            ):
                num_evicted_tokens = num_new_tokens + local_end_index.item() - kv_cache_size
                num_rolled_tokens = local_end_index.item() - num_evicted_tokens - sink_tokens
                kv_cache[CacheIndex.K][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache[CacheIndex.K][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache[CacheIndex.V][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache[CacheIndex.V][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                new_local_end_index = (
                    local_end_index.item() + current_end - global_end_index.item() - num_evicted_tokens
                )
            else:
                new_local_end_index = local_end_index.item() + current_end - global_end_index.item()

            local_start_index = new_local_end_index - num_new_tokens
            kv_cache[CacheIndex.K][:, local_start_index:new_local_end_index] = roped_key
            kv_cache[CacheIndex.V][:, local_start_index:new_local_end_index] = v

            k_cache = kv_cache[CacheIndex.K][:, max(0, new_local_end_index - max_attention_size) : new_local_end_index]
            v_cache = kv_cache[CacheIndex.V][:, max(0, new_local_end_index - max_attention_size) : new_local_end_index]
            out = self.attn(roped_query, k_cache, v_cache)

            global_end_index.fill_(current_end)
            local_end_index.fill_(new_local_end_index)
        else:
            # local_attn_size == -1: per-row writes to non-overlapping cache slots,
            # per-row attention reads sized by max_attention_size. Loops once per
            # batch row inside attention to avoid needing a key-padding mask.
            outs = []
            max_end = 0
            for i in range(b):
                cs_i = current_starts[i]
                ce_i = cs_i + num_new_tokens
                kv_cache[CacheIndex.K][:, cs_i:ce_i] = roped_key[i : i + 1]
                kv_cache[CacheIndex.V][:, cs_i:ce_i] = v[i : i + 1]

                kv_start_i = max(0, ce_i - max_attention_size)
                k_cache_i = kv_cache[CacheIndex.K][:, kv_start_i:ce_i]
                v_cache_i = kv_cache[CacheIndex.V][:, kv_start_i:ce_i]

                outs.append(self.attn(roped_query[i : i + 1], k_cache_i, v_cache_i))
                if ce_i > max_end:
                    max_end = ce_i

            out = torch.cat(outs, dim=0)
            global_end_index.fill_(max_end)
            local_end_index.fill_(max_end)

        # output
        out = out.flatten(2)
        out = self.o(out)
        return out


class WanCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

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
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache.get("is_init", False):
                crossattn_cache["is_init"] = True
                # Cache at B=1 (text context is shared across chunks in a batch);
                # expand on retrieval to match q's batch size for variable-B calls.
                k = self.norm_k(self.k(context[:1])).view(1, -1, n, d)
                v = self.v(context[:1]).view(1, -1, n, d)
                crossattn_cache[CacheIndex.K] = k
                crossattn_cache[CacheIndex.V] = v
            else:
                k = crossattn_cache[CacheIndex.K]
                v = crossattn_cache[CacheIndex.V]
            if k.shape[0] != b:
                k = k.expand(b, *k.shape[1:])
                v = v.expand(b, *v.shape[1:])
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

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
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim, num_heads=num_heads, local_attn_size=local_attn_size, sink_size=sink_size, qk_norm=qk_norm, eps=eps
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
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
        current_starts=0,
        max_attention_size=1_000_000,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            current_starts(int | list[int]): per-row absolute token offset; int broadcasts.
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache,
            local_end_index,
            global_end_index,
            current_starts,
            max_attention_size,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
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
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
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
        with torch.amp.autocast("cuda", dtype=torch.float32):
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
        model_type="t2v",
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
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
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

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

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

        # PP layout — defaults to single-stage; apply_pp_split() refines after loading.
        self.start_layer = 0
        self.end_layer = num_layers

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        # initialize weights
        self.init_weights()

    def apply_pp_split(self) -> None:
        """Partition the model across PP ranks. Called after weight loading.

        After this returns, blocks outside this rank's [start_layer, end_layer)
        slice are replaced with PPMissingLayer(); embeddings/head are kept only
        on the first/last stage. KV-cache sizing (in the pipeline state) reads
        end_layer - start_layer to allocate just for the owned slice.
        """
        pp_world = get_pipeline_parallel_world_size()
        if pp_world <= 1:
            self.start_layer = 0
            self.end_layer = self.num_layers
            return

        rank = get_pipeline_parallel_rank()
        per_rank = self.num_layers // pp_world
        rem = self.num_layers % pp_world
        # Even split: extra layers go to the first `rem` ranks.
        self.start_layer = rank * per_rank + min(rank, rem)
        self.end_layer = self.start_layer + per_rank + (1 if rank < rem else 0)

        for i in range(self.num_layers):
            if not (self.start_layer <= i < self.end_layer):
                self.blocks[i] = PPMissingLayer()

        if not is_pipeline_first_stage():
            self.patch_embedding = PPMissingLayer()
            self.patch_embedding_wancamctrl = PPMissingLayer()
            self.c2ws_hidden_states_layer1 = PPMissingLayer()
            self.c2ws_hidden_states_layer2 = PPMissingLayer()

        if not is_pipeline_last_stage():
            self.head = PPMissingLayer()

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
        current_starts=0,
        max_attention_size=1_000_000,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        r"""
        Run the diffusion model with kv caching.

        On the first PP stage, ``x``/``y``/``dit_cond_dict`` are consumed to build
        the token sequence; non-first stages take ``hidden_states`` (and the
        camera-conditioned ``c2ws_plucker_emb`` if used) from ``intermediate_tensors``.
        Non-last stages return an ``IntermediateTensors`` carrying ``hidden_states``
        (plus ``c2ws_plucker_emb`` so downstream stages can do cam injection).

        Args:
            current_starts (int | list[int]): per-row absolute token offset.
                int broadcasts to all rows.
            intermediate_tensors: per-stage hidden state from the previous PP rank.

        Returns:
            list[Tensor] on last PP stage; IntermediateTensors elsewhere.
        """

        if self.model_type == "i2v" and is_pipeline_first_stage():
            assert y is not None

        # params
        first_stage = is_pipeline_first_stage()
        last_stage = is_pipeline_last_stage()
        # `freqs` lives as a plain attribute (not a buffer) — move it to the
        # device of the first parameter we can find on this stage.
        first_param = next(self.parameters())
        device = first_param.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if first_stage:
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
            )
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
            assert seq_lens.max() <= seq_len
            x = torch.cat(x)
        else:
            assert intermediate_tensors is not None, "non-first PP stage requires intermediate_tensors"
            x = intermediate_tensors["hidden_states"]
            grid_sizes = intermediate_tensors["grid_sizes"]
            seq_lens = intermediate_tensors["seq_lens"]

        B = x.shape[0]
        s = x.shape[1]

        # Per-row time embeddings: same timestep replicated across this row's tokens.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if t.dim() == 1:
                t_full = t.unsqueeze(1).expand(B, s).contiguous()
            else:
                t_full = t
            bt, btn = t_full.shape
            t_flat = t_full.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat).unflatten(0, (bt, btn)).float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context — text embedding runs on every stage (each block has cross-attn).
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        # cam Plucker — processed on first stage, then forwarded via intermediate_tensors
        # so downstream stages re-use the same embedding for in-block cam injection.
        if first_stage:
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
                c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=0)

                c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
                c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                    torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))
                )
                dit_cond_dict = dict(dit_cond_dict)
                dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb + c2ws_hidden_states
        else:
            if "c2ws_plucker_emb" in intermediate_tensors.tensors:
                dit_cond_dict = {"c2ws_plucker_emb": intermediate_tensors["c2ws_plucker_emb"]}
            else:
                dit_cond_dict = None

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

        # Iterate this rank's blocks. kv_cache / crossattn_cache / *_end_index are
        # sized to (end_layer - start_layer) — index locally.
        for local_idx, block in enumerate(self.blocks[self.start_layer : self.end_layer]):
            kwargs.update(
                {
                    "kv_cache": kv_cache[local_idx],
                    "crossattn_cache": crossattn_cache[local_idx],
                    "local_end_index": local_end_index[local_idx],
                    "global_end_index": global_end_index[local_idx],
                    "current_starts": current_starts,
                }
            )
            x = block(x, **kwargs)

        if not last_stage:
            model_dtype = next(self.parameters()).dtype
            it = {
                "hidden_states": x.to(model_dtype),
                "grid_sizes": grid_sizes,
                "seq_lens": seq_lens,
            }
            if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
                it["c2ws_plucker_emb"] = dit_cond_dict["c2ws_plucker_emb"].to(model_dtype)
            return IntermediateTensors(it)

        # head + unpatchify only on the last PP stage
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

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
