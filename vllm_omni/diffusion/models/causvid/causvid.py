"""Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing)."""

import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform

from .state_causvid import CacheIndex
from .wan_model import rope_params, sinusoidal_embedding_1d

logger = logging.getLogger(__name__)


_ROPE_FREQ_CACHE: OrderedDict = OrderedDict()
_ROPE_FREQ_CACHE_SIZE = 16


def _get_rope_freqs(freqs_parts, freqs_id, f, h, w, sf, device):
    key = (freqs_id, device.type, device.index, f, h, w, sf)
    cached = _ROPE_FREQ_CACHE.get(key)
    if cached is not None:
        _ROPE_FREQ_CACHE.move_to_end(key)
        return cached
    freqs_i = torch.cat(
        [
            freqs_parts[0][sf : sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    _ROPE_FREQ_CACHE[key] = freqs_i
    if len(_ROPE_FREQ_CACHE) > _ROPE_FREQ_CACHE_SIZE:
        _ROPE_FREQ_CACHE.popitem(last=False)
    return freqs_i


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply causal RoPE per batch row.

    ``start_frame`` may be a single int (broadcast to all rows) or a list/tuple
    of ints, one per row.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs_parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_id = id(freqs)
    device = freqs.device

    if isinstance(start_frame, int):
        start_frames = [start_frame] * grid_sizes.shape[0]
    else:
        start_frames = list(start_frame)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        sf = start_frames[i]
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = _get_rope_freqs(freqs_parts, freqs_id, f, h, w, sf, device)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        tail = x[i, seq_len:]
        if tail.numel():
            x_i = torch.cat([x_i, tail])

        # append to collection
        output.append(x_i)
    out = output[0].unsqueeze(0) if len(output) == 1 else torch.stack(output)
    return out.type_as(x)


class WanLayerNorm(nn.LayerNorm):
    # Plain LayerNorm in the working dtype (bf16); no fp32 upcast.
    def forward(self, x):
        return super().forward(x).type_as(x)


class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.sink_threshold = 0.0
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

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        local_end_index=None,
        global_end_index=None,
        current_start=0,
        current_end=0,
        slot_idxs=None,
        evict_queues=None,
    ):
        r"""
        Two execution modes:

        - **single-slot** (``slot_idxs is None``): mirrors causvid
          ``forward()`` — single ``kv_cache`` Tensor, scalar ``current_start``
          / ``current_end``, writes ``[cs:ce]`` and reads ``[0:ce]``.
        - **Stream-batch rolling + adaptive sinks** (``slot_idxs`` provided):
          bounded per-slot cache holds ``(sink_size + local_attn_size)``
          chunks. ``evict_queues`` is a per-slot ``deque[int]`` of
          end-positions to overwrite next; sinks stay stable unless the
          adaptive cosine-sim check flags one for one-shot eviction.

        ``local_end_index`` / ``global_end_index`` are per-slot mutable
        single-element int lists (``[int]``) — read via ``lst[0]``, write via
        ``lst[0] = val``. No GPU sync.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()

        if isinstance(current_start, int):
            current_starts = [current_start] * b
        else:
            current_starts = list(current_start)
        start_frames = [cs // frame_seqlen for cs in current_starts]

        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frames).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frames).type_as(v)
        num_new_tokens = roped_query.shape[1]

        # ── Without stream-batch ─────────────────────────────────────────────
        if slot_idxs is None:
            cs = current_starts[0]
            ce = current_end if isinstance(current_end, int) else list(current_end)[0]
            kv_cache[CacheIndex.K][:, cs:ce] = roped_key
            kv_cache[CacheIndex.V][:, cs:ce] = v
            k_cache = kv_cache[CacheIndex.K][:, 0:ce]
            v_cache = kv_cache[CacheIndex.V][:, 0:ce]
            x = self.attn(roped_query, k_cache, v_cache)
            x = x.flatten(2)
            x = self.o(x)
            return x

        # ── Stream-batch stratified path ────────────────────────────────
        slot_list = [slot_idxs] * b if isinstance(slot_idxs, int) else list(slot_idxs)
        assert self.local_attn_size != -1, "stream-batch requires a bounded KV cache (local_attn_size > 0)"
        assert evict_queues is not None, "rolling KV cache requires evict_queues"

        k_all = kv_cache[CacheIndex.K]  # [num_slots, kv_size, n, d]
        v_all = kv_cache[CacheIndex.V]
        kv_cache_size = k_all.shape[1]
        chunk_tokens = num_new_tokens
        sink_tokens = self.sink_size * chunk_tokens

        # Per-row cache writes + bookkeeping; the attention read follows.
        row_lens: list[int] = [0] * b
        for i in range(b):
            slot_i = slot_list[i]
            if slot_i == -1:
                continue
            slot_local_end = local_end_index[slot_i]
            slot_global_end = global_end_index[slot_i]
            slot_evict_queue = evict_queues[slot_i]

            cs_i = current_starts[i]
            ce_i = cs_i + num_new_tokens

            cur_local_end = slot_local_end[0]
            cur_global_end = slot_global_end[0]

            # 1. Adaptive sink check.
            if self.sink_size > 0 and self.sink_threshold > 0.0 and cur_local_end >= sink_tokens:
                new_pool_k = roped_key[i : i + 1].mean(dim=1).flatten(1)  # [1, n*d]
                new_pool_v = v[i : i + 1].mean(dim=1).flatten(1)
                sink_k_pool = k_all[slot_i, :sink_tokens].reshape(self.sink_size, chunk_tokens, n * d).mean(dim=1)
                sink_v_pool = v_all[slot_i, :sink_tokens].reshape(self.sink_size, chunk_tokens, n * d).mean(dim=1)
                k_cos = torch_F.cosine_similarity(sink_k_pool, new_pool_k, dim=-1)
                v_cos = torch_F.cosine_similarity(sink_v_pool, new_pool_v, dim=-1)
                avg_cos = (k_cos + v_cos) / 2  # [sink_size]
                if avg_cos.min().item() < self.sink_threshold:
                    min_idx = int(avg_cos.argmin().item())
                    sink_end = (min_idx + 1) * chunk_tokens
                    if sink_end not in slot_evict_queue:
                        slot_evict_queue.appendleft(sink_end)

            # 2. Wrap if cache is full or this chunk overruns it.
            needs_wrap = (ce_i > kv_cache_size) or (cur_local_end >= kv_cache_size)

            if needs_wrap and slot_evict_queue:
                target_end = slot_evict_queue.popleft()
                if target_end > sink_tokens:
                    slot_evict_queue.append(target_end)
                k_all[slot_i, target_end - num_new_tokens : target_end] = roped_key[i]
                v_all[slot_i, target_end - num_new_tokens : target_end] = v[i]
                new_local_end_i = kv_cache_size
            else:
                new_local_end_i = cur_local_end + ce_i - cur_global_end
                local_start_i = new_local_end_i - num_new_tokens
                k_all[slot_i, local_start_i:new_local_end_i] = roped_key[i]
                v_all[slot_i, local_start_i:new_local_end_i] = v[i]
                rolling_end = new_local_end_i + num_new_tokens
                if (
                    rolling_end > sink_tokens
                    and rolling_end <= kv_cache_size
                    and (not slot_evict_queue or slot_evict_queue[-1] != rolling_end)
                ):
                    slot_evict_queue.append(rolling_end)

            slot_global_end[0] = ce_i
            slot_local_end[0] = new_local_end_i
            row_lens[i] = new_local_end_i

        # 3. Attention over the bounded caches, one read per active slot.
        outs = []
        for i in range(b):
            s = slot_list[i]
            if s == -1:
                outs.append(roped_query[i : i + 1])
                continue
            le = row_lens[i]
            outs.append(self.attn(roped_query[i : i + 1], k_all[s : s + 1, :le], v_all[s : s + 1, :le]))
        x = torch.cat(outs, dim=0)

        x = x.flatten(2)
        x = self.o(x)
        return x


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
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache.get("is_init", False):
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context[:1])).view(1, -1, n, d)
                v = self.v(context[:1]).view(1, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
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
        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = CausalWanSelfAttention(
            dim=dim, num_heads=num_heads, local_attn_size=local_attn_size, sink_size=sink_size, qk_norm=qk_norm, eps=eps
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

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
        current_end=None,
        slot_idxs=None,
        evict_queues=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache,
            local_end_index=local_end_index,
            global_end_index=global_end_index,
            current_start=current_start,
            current_end=current_end,
            slot_idxs=slot_idxs,
            evict_queues=evict_queues,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )

            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
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
        self.norm = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


def balance_layers_by_cost(num_layers: int, fixed_cost_blocks: list[float]) -> list[int]:
    pp = len(fixed_cost_blocks)
    counts = [0] * pp
    for _ in range(num_layers):
        r = min(range(pp), key=lambda i: (fixed_cost_blocks[i] + counts[i], i))
        counts[r] += 1
    return counts


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _repeated_blocks = ["CausalWanAttentionBlock"]

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

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

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

        # PP layout
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

    def apply_pp_split(self, layer_counts: list[int] | None = None, free_blocks: bool = True) -> None:
        pp_world = get_pipeline_parallel_world_size()
        if pp_world <= 1:
            self.start_layer = 0
            self.end_layer = self.num_layers
            return

        rank = get_pipeline_parallel_rank()
        if layer_counts is not None:
            # Explicit per-rank counts (dynamic block scheduling).
            assert len(layer_counts) == pp_world and sum(layer_counts) == self.num_layers, (
                f"layer_counts {layer_counts} must have {pp_world} entries summing to {self.num_layers}"
            )
            self.start_layer = sum(layer_counts[:rank])
            self.end_layer = self.start_layer + layer_counts[rank]
        else:
            # Honors VLLM_PP_LAYER_PARTITION; default keeps the remainder off the head stage.
            self.start_layer, self.end_layer = get_pp_indices(self.num_layers, rank, pp_world)

        # The forward only runs blocks[start:end]; freeing the rest reclaims memory.
        # Dynamic scheduling keeps all blocks resident (free_blocks=False) so a later
        # rebalance can re-own any block, then frees on the final split.
        if free_blocks:
            for i in range(self.num_layers):
                if not (self.start_layer <= i < self.end_layer):
                    self.blocks[i] = PPMissingLayer()

        if not is_pipeline_first_stage():
            self.patch_embedding = PPMissingLayer()

        if not is_pipeline_last_stage():
            self.head = PPMissingLayer()

    def forward(
        self,
        x,
        t,
        context,
        seq_len=1_000_000,
        y=None,
        grid_sizes=None,
        dit_cond_dict=None,
        kv_cache=None,
        local_end_index=None,
        global_end_index=None,
        crossattn_cache=None,
        current_start=0,
        current_end=None,
        slot_idxs=None,
        evict_queues=None,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        r"""
        Run the diffusion model with KV caching, optionally split across PP ranks.

        On the first PP stage, ``x``/``y`` are consumed to build the token sequence;
        non-first stages take ``hidden_states`` from ``intermediate_tensors``.
        Non-last stages return an ``IntermediateTensors`` with ``hidden_states``.

        Stream-batch kwargs (``slot_idxs`` / ``local_end_index`` /
        ``global_end_index`` / ``evict_queues``) are forwarded per block.
        """

        if self.model_type == "i2v" and is_pipeline_first_stage():
            assert y is not None

        first_stage = is_pipeline_first_stage()
        last_stage = is_pipeline_last_stage()

        # ``freqs`` is a plain attribute (not a buffer) — move it to this
        # rank's device on the first call.
        first_param = next(self.parameters())
        device = first_param.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if first_stage:
            stream_xt = torch.stack(x)
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = [self.patch_embedding(u.unsqueeze(0).to(first_param.dtype)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat(x)
        else:
            assert intermediate_tensors is not None, "non-first PP stage requires intermediate_tensors"
            assert grid_sizes is not None, "non-first PP stage requires grid_sizes kwarg"
            t = intermediate_tensors["t"]
            x = intermediate_tensors["hidden_states"]
            stream_xt = intermediate_tensors["xt"]
            seq_lens = None

        # time embeddings (``t`` is ``[B, F]`` per-row, per-frame).
        with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        e0 = e0.to(first_param.dtype)
        e = e.to(first_param.dtype)

        # context
        context_lens = None
        if crossattn_cache is None or any(not c.get("is_init", False) for c in crossattn_cache):
            context = self.text_embedding(
                torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
            ).to(first_param.dtype)
        else:
            context = None

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
        )

        # Owned blocks index caches locally; a global (all-blocks) cache from dynamic
        # block scheduling is indexed by ``start_layer + local_idx``.
        cache_off = self.start_layer if (kv_cache is not None and len(kv_cache) == self.num_layers) else 0
        for local_idx, block in enumerate(self.blocks[self.start_layer : self.end_layer]):
            ci = cache_off + local_idx
            kwargs.update(
                {
                    "kv_cache": kv_cache[ci],
                    "crossattn_cache": crossattn_cache[ci],
                    "local_end_index": local_end_index[ci] if local_end_index is not None else None,
                    "global_end_index": global_end_index[ci] if global_end_index is not None else None,
                    "current_end": current_end,
                    "current_start": current_start,
                    "slot_idxs": slot_idxs,
                    "evict_queues": evict_queues[ci] if evict_queues is not None else None,
                }
            )
            x = block(x, **kwargs)

        if not last_stage:
            model_dtype = next(self.parameters()).dtype
            return IntermediateTensors({"hidden_states": x.to(model_dtype), "t": t, "xt": stream_xt})

        # Last stage: expose the t this forward used (carrying the admitted chunk's
        # first_timestep, which only rank 0 computes and propagates via the IT) so
        # step_scheduler reuses it instead of reconstructing it from a stale value.
        if is_forward_context_available():
            ctx = get_forward_context()
            ctx.stream_t = t
            if not first_stage:
                ctx.stream_xt = stream_xt

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
