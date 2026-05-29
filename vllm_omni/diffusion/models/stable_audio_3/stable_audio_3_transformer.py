# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 DiT (Diffusion Transformer) for vLLM-Omni.

PORT_FROM: stable-audio-3
  - models/transformer.py (1272 lines, near-verbatim port)
  - models/dit.py (642 lines, near-verbatim port)
  - models/blocks.py FourierFeatures + ExpoFourierFeatures

vLLM-Omni adaptations applied to DiffusionTransformer (at the bottom):
  - Accept od_config: OmniDiffusionConfig | None = None
  - Add load_weights() entry for AutoWeightsLoader
  - Add _repeated_blocks / _layerwise_offload_blocks_attr class attrs
  - Export StableAudio3DiTModel = DiffusionTransformer alias

Stage 1 keeps upstream's Attention verbatim (uses F.scaled_dot_product_attention).
Stage 2 (follow-up PR) will swap to vllm_omni.diffusion.attention.layer.Attention.
"""

from __future__ import annotations

import math
import typing as tp
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.stable_audio_3.conditioners import ExpoFourierFeatures


# Stub LoRA helpers (model-side no-ops; vllm-omni framework handles LoRA at serving layer)
class _NoOpLoRAParametrization:
    pass


def has_lora(module):
    return False


def enable_lora(module):
    pass


def disable_lora(module):
    pass


def set_lora_strength(module, strength, lora_index=0):
    pass


def filter_lora_layers(module):
    return []


LoRAParametrization = _NoOpLoRAParametrization



# ===========================================================================
# Fourier helpers (PORT_FROM: models/blocks.py)
# ===========================================================================

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=16.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# ===========================================================================
# Transformer building blocks (PORT_FROM: models/transformer.py)
# ===========================================================================
from functools import reduce, partial
from packaging import version
import logging
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from torch.nn.utils.parametrizations import weight_norm
from typing import Callable, Literal, Optional
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    flex_attention_available = False

try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
except ImportError as e:
    print(e)
    print('flash_attn varlen/bert_padding not available, disabling varlen attention')
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    index_first_axis = None


def precompute_varlen_metadata(padding_mask: torch.Tensor):
    """
    Precompute varlen attention metadata once to avoid recomputation in every attention layer.

    Args:
        padding_mask: Boolean tensor of shape (batch, seq_len) where True = valid

    Returns:
        Dict with cu_seqlens, max_seqlen, indices, batch_size, seq_len for use in attention
    """
    if padding_mask is None or unpad_input is None:
        return None

    batch_size, seq_len = padding_mask.shape

    # Compute cumulative sequence lengths (same for all of q, k, v)
    seqlens = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()

    # Compute indices for gathering valid tokens
    # indices maps from packed position -> original (batch, seq) position
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()

    return {
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "indices": indices,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

# PORT_FROM: models/utils.py compile helper (inlined to avoid extra file)
import os as _os

torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
torch._dynamo.config.suppress_errors = True

_enable_torch_compile = _os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"


def compile(function, *args, **kwargs):
    """torch.compile wrapper with ENABLE_TORCH_COMPILE env gate. PORT_FROM: models/utils.py."""
    if _enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function
    return function


def _left_pad_to_match(emb, target_len):
    """Left-pad or right-trim emb along seq dim to match target_len.

    Used for local conditioning embeddings that need to align with x
    without affecting prepended tokens (memory tokens, global cond, etc.).
    """
    emb_len = emb.shape[-2]
    if emb_len < target_len:
        return F.pad(emb, (0, 0, target_len - emb_len, 0), value=0.)
    elif emb_len > target_len:
        return emb[:, -target_len:, :]
    return emb

if flex_attention_available:
    try:
        torch._dynamo.config.cache_size_limit = 5000
        flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    except Exception as e:
        logging.debug(f"Could not compile flex_attention, using uncompiled version: {e}")
        flex_attention_compiled = flex_attention
else:
    flex_attention_compiled = None


# Cache band block_masks for sliding-window attention fallback (flex_attention path).
# Keyed by (seq_q, seq_k, w_left, w_right, device). create_block_mask is expensive
# but the result is reused across all transformer layers and forward passes.
_SLIDING_WINDOW_BLOCK_MASK_CACHE = {}

def _get_sliding_window_block_mask(seq_q, seq_k, w_left, w_right, device):
    key = (seq_q, seq_k, int(w_left), int(w_right), str(device))
    bm = _SLIDING_WINDOW_BLOCK_MASK_CACHE.get(key)
    if bm is None:
        wl, wr = int(w_left), int(w_right)
        def _band_mod(b, h, q_idx, kv_idx):
            delta = kv_idx - q_idx
            return (delta >= -wl) & (delta <= wr)
        bm = create_block_mask(_band_mod, B=None, H=None, Q_LEN=seq_q, KV_LEN=seq_k, device=device)
        _SLIDING_WINDOW_BLOCK_MASK_CACHE[key] = bm
    return bm

def _sliding_window_additive_mask(seq_q, seq_k, w_left, w_right, device, dtype):
    """Build a (seq_q, seq_k) additive mask for masked SDPA fallback.
    0 inside the band [i - w_left, i + w_right], -inf outside.
    """
    ii = torch.arange(seq_q, device=device)
    jj = torch.arange(seq_k, device=device)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
    mask = torch.zeros((seq_q, seq_k), dtype=dtype, device=device)
    return mask.masked_fill(~in_band, float('-inf'))


# Chunked-halo SDPA fallback. Math-equivalent to masked SDPA with a band
# mask, but processes queries in non-overlapping chunks with a (w_left,
# w_right) halo of keys/values on each side — every query stays inside its
# chunk's softmax. Avoids materializing the O(N^2) mask.
#
# At realistic SAME-L decoder shapes (N=69632, W=17, packed sequence is
# latent_length * (stride+1)): ~34x faster than full masked SDPA, and
# ~140x less peak mask memory (~1 MB per chunk vs 9.7 GB for one N x N mask).
# Chunk size is a tunable; 1024 is a good default at typical pretransform
# decoder shapes. Larger chunks waste more compute on out-of-band tiles;
# smaller chunks suffer from launch overhead.
_SLIDING_WINDOW_CHUNK_SIZE = 1024

def _sliding_window_chunked_halo_sdpa(q, k, v, w_left, w_right, chunk_size=_SLIDING_WINDOW_CHUNK_SIZE):
    B, H, N, D = q.shape
    outs = []
    for q_start in range(0, N, chunk_size):
        q_end = min(q_start + chunk_size, N)
        k_start = max(0, q_start - int(w_left))
        k_end = min(N, q_end + int(w_right))
        q_c = q[..., q_start:q_end, :]
        k_c = k[..., k_start:k_end, :]
        v_c = v[..., k_start:k_end, :]
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        delta = k_idx[None, :] - q_idx[:, None]
        in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
        mask = torch.zeros(delta.shape, dtype=q.dtype, device=q.device).masked_fill(~in_band, float('-inf'))
        outs.append(F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=mask, is_causal=False))
    return torch.cat(outs, dim=-2)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    # Preserve autocast context during recomputation to avoid dtype mismatches
    if "context_fn" not in kwargs:
        from torch.amp import autocast
        import functools
        # Get current autocast state
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype('cuda')
            def get_contexts():
                return (
                    autocast('cuda', dtype=dtype),
                    autocast('cuda', dtype=dtype),
                )
            kwargs["context_fn"] = get_contexts
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast("cuda", enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


@autocast("cuda", enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

# norms
class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class RunningInstanceNorm(nn.Module):
    def __init__(self, dim, momentum = 0.99, eps = 1e-4, saturate = True, trainable_gain = True):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1,1,dim))
        self.register_buffer("running_std", torch.ones(1,1,dim))
        self.saturate = saturate
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.trainable_gain = trainable_gain
        if self.trainable_gain:
            self.gain = nn.Parameter(torch.ones(1))
    
    def _update_stats(self, x):
        self.running_mean = self.running_mean * self.momentum + x.detach().mean(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)
        self.running_std  = (self.running_std * self.momentum + x.detach().std(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)).clip(min = self.eps)

    def forward(self, x):
        if self.training:
            self._update_stats(x)
        x = (x - self.running_mean) / self.running_std
        if self.saturate:
            x = torch.asinh(x)
        if self.trainable_gain:
            x = x * self.gain
        return x
        
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    #@autocast("cuda", enabled = False)
    def forward(self, x):
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        else:
            output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
            return output.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, fix_scale=False, force_fp32=False, eps=1e-5):
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.rms_norm(x, x.shape[-1:], weight=self.gamma, eps=self.eps)
        else:
            output = F.rms_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), eps=self.eps)
            return output.to(x.dtype)

class LayerScale(nn.Module):
    def __init__(self, dim, init_val = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(3.14159265359 * x)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
        sinusoidal = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU() if not sinusoidal else Sin()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    #@compile
    def forward(self, x, varlen_metadata=None):
        if varlen_metadata is not None and index_first_axis is not None and pad_input is not None:
            # Pack valid tokens for efficient FFN computation (skip padding tokens)
            # Padding positions become zeros after unpack, which is fine since FFN output
            # is added to residual, preserving values at padding positions
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            indices = varlen_metadata["indices"]
            dim = x.shape[-1]

            # Pack to (N_valid, D)
            x_packed = index_first_axis(x.reshape(-1, dim), indices)

            # FFN on packed representation with pseudo-batch dim
            x_packed = self.ff(x_packed.unsqueeze(0)).squeeze(0)

            # Unpack back to (B, T, D)
            return pad_input(x_packed, indices, batch_size, seq_len)
        else:
            return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm_eps = 1e-6,
        qk_norm: Literal['l2', 'ln', 'rms', 'dyt', 'none'] = 'none',
        differential = False,
        feat_scale = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads

        self.differential = differential

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            if differential:
                self.to_q = nn.Linear(dim, dim * 2, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
            else:
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            if differential:
                self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
            else:
                self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'rms', 'dyt','none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "rms" ,"dyt", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif self.qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

        self.causal = causal
        
    @compile
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return q, k


    def apply_attn(self, q, k, v, causal = None, flex_attention_block_mask = None, flex_attention_score_mod = None, flash_attn_sliding_window = None, padding_mask = None, varlen_metadata = None):

        if self.num_heads != self.kv_heads:
             # Repeat interleave kv_heads to match q_heads for grouped query attention
             heads_per_kv_head = self.num_heads // self.kv_heads
             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        flash_attn_available = flash_attn_func is not None
        flash_attn_varlen_available = flash_attn_varlen_func is not None and index_first_axis is not None

        if causal and (flex_attention_block_mask is not None or flex_attention_score_mod is not None):
            flex_attention_block_mask = None
            flex_attention_score_mod = None

        if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
            # Flex attention path - use V-zeroing for padding mask
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            out = flex_attention_compiled(q,k,v,
                block_mask = flex_attention_block_mask,
                score_mod = flex_attention_score_mod)
        elif flash_attn_available and varlen_metadata is not None and flash_attn_varlen_available:
            # Flash attention with varlen using precomputed metadata (fast path)
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            cu_seqlens = varlen_metadata["cu_seqlens"]
            max_seqlen = varlen_metadata["max_seqlen"]
            indices = varlen_metadata["indices"]

            fa_dtype_in = q.dtype
            # Rearrange to (B, T, H, D) for flash_attn
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))

            # Pack q, k, v using precomputed indices (much faster than calling unpad_input 3x)
            num_heads, head_dim = q.shape[2], q.shape[3]
            q_unpad = index_first_axis(q.reshape(-1, num_heads, head_dim), indices)
            k_unpad = index_first_axis(k.reshape(-1, num_heads, head_dim), indices)
            v_unpad = index_first_axis(v.reshape(-1, num_heads, head_dim), indices)

            out_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                causal=causal if causal is not None else False,
                window_size=flash_attn_sliding_window if flash_attn_sliding_window is not None else (-1, -1),
            )

            # Pad output back to original shape
            out = pad_input(out_unpad, indices, batch_size, seq_len)
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        elif flash_attn_available:
            # Standard flash attention (no padding mask, or varlen imports not available)
            # Apply V-zeroing fallback if padding_mask provided but we couldn't use varlen
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))

            out = flash_attn_func(q, k, v, causal = causal, window_size=flash_attn_sliding_window if (flash_attn_sliding_window is not None) else [-1,-1])

            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        else:
            # No flash-attn available. Sliding-window fallback cascade:
            #   Tier 2: flex_attention with band block_mask (best when torch.compile works)
            #   Tier 3: chunked-halo masked SDPA           (math-equivalent, ~30x faster than tier 4)
            #   Tier 4: full masked SDPA (N x N mask)      (last resort; high memory)
            # For the no-sliding-window case, fall through to plain SDPA full attention.
            # All apply V-zeroing for padding masks (cheap and equivalent to masking
            # those positions out of attention output).
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            if flash_attn_sliding_window is not None:
                seq_q, seq_k = q.shape[2], k.shape[2]
                wl, wr = flash_attn_sliding_window
                handled = False
                if flex_attention_available and flex_attention_compiled is not None:
                    try:
                        bm = _get_sliding_window_block_mask(seq_q, seq_k, wl, wr, q.device)
                        out = flex_attention_compiled(q, k, v, block_mask=bm)
                        handled = True
                    except Exception as _flex_err:
                        logging.debug(f"flex_attention failed, trying chunked-halo SDPA: {_flex_err}")
                if not handled:
                    try:
                        out = _sliding_window_chunked_halo_sdpa(q, k, v, wl, wr)
                        handled = True
                    except Exception as _chunk_err:
                        logging.debug(f"chunked-halo SDPA failed, falling back to full masked SDPA: {_chunk_err}")
                if not handled:
                    add_mask = _sliding_window_additive_mask(seq_q, seq_k, wl, wr, q.device, q.dtype)
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask, is_causal=False)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal if causal is not None else False)
        return out


    #@compile
    def forward(
        self,
        x,
        context = None,
        rotary_pos_emb = None,
        rotary_pos_emb_k = None,
        causal = None,
        flex_attention_block_mask = None,
        flex_attention_score_mod = None,
        flash_attn_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            if self.differential:
                q, q_diff = self.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, q_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k, k_diff, v = self.to_kv(kv_input).chunk(3, dim=-1)
                k, k_diff, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, k_diff, v))
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
                k, v = self.to_kv(kv_input).chunk(2, dim=-1)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            if self.differential:
                q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
                q, k, v, q_diff, k_diff  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, q_diff, k_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)
            # q = self.q_norm(q)
            # k = self.k_norm(k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
        
            q_freqs = freqs

            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs

                if q.shape[-2] >= k.shape[-2]:
                    ratio = q.shape[-2] / k.shape[-2]
                    q_freqs, k_freqs = freqs, ratio * freqs
                else:
                    ratio = k.shape[-2] / q.shape[-2]
                    q_freqs, k_freqs = ratio * freqs, freqs

            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        
        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.differential:
            q, q_diff = q.unbind(dim = 1)
            k, k_diff = k.unbind(dim = 1)
            out = self.apply_attn(q, k, v,  causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            out_diff = self.apply_attn(q_diff, k_diff, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            out = out - out_diff
        else:
            out = self.apply_attn(q, k, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if self.feat_scale:
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(out.dtype)  # (b, n, 1)
                out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1)
                out_hf = out - out_dc
                out = out + (self.lambda_dc * out_dc + self.lambda_hf * out_hf) * mask
            else:
                out_dc = out.mean(dim=-2, keepdim=True)
                out_hf = out - out_dc
                out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    #@compile
    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            local_add_cond_dim = None,
            modular_local_cond_configs = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            add_rope = False,
            layer_scale = False,
            norm_type = 'layer_norm',
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads,dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
       
        if layer_scale and zero_init_branch_outputs:
            print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
            zero_init_branch_outputs = False
        
        if norm_type not in ['layer_norm', 'rms_norm', 'dyt']:
            raise ValueError(f'norm_type must be one of ["layer_norm", "rms_norm", "dyt"], got {norm_type}')

        norm_layer_map = {
            'layer_norm': LayerNorm,
            'rms_norm': RMSNorm,
            'dyt': DynamicTanh
        }
        norm_layer = norm_layer_map[norm_type]

        self.pre_norm = norm_layer(dim,**norm_kwargs)
        self.add_rope = add_rope

        self.self_attn = Attention(
            dim,
            dim_heads = self.dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = norm_layer(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads = self.dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.layer_ix = layer_ix

        self.conformer = None
        if conformer:
            self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
            self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

        self.local_add_cond_dim = local_add_cond_dim

        if local_add_cond_dim is not None:
            self.to_local_embed = nn.Sequential(
                nn.Linear(local_add_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

            nn.init.zeros_(self.to_local_embed[-1].weight)
            nn.init.zeros_(self.to_local_embed[-1].bias)

        else:
            self.to_local_embed = None

        # Modular local conditioning - independent projections per conditioning ID
        self.modular_local_cond_configs = modular_local_cond_configs or []
        self.modular_local_embeds = nn.ModuleDict()

        for config in self.modular_local_cond_configs:
            cond_id = config["id"]
            cond_dim = config["dim"]
            proj = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )
            # Zero-init output layer so new conditioning doesn't affect model initially
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.modular_local_embeds[cond_id] = proj

        self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None

    def _apply_local_conditioning(self, x, local_add_cond, modular_local_cond):
        """Apply local additive and modular local conditioning to x."""
        if local_add_cond is not None and self.to_local_embed is not None:
            local_emb = self.to_local_embed(local_add_cond)
            x = x + _left_pad_to_match(local_emb, x.shape[-2])

        if modular_local_cond is not None and len(self.modular_local_embeds) > 0:
            modular_sum = None
            for cond_id, proj in self.modular_local_embeds.items():
                if cond_id in modular_local_cond:
                    local_emb = proj(modular_local_cond[cond_id])
                    local_emb = _left_pad_to_match(local_emb, x.shape[-2])
                    modular_sum = local_emb if modular_sum is None else modular_sum + local_emb
            if modular_sum is not None:
                x = x + modular_sum

        return x

    @compile
    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        rotary_pos_emb = None,
        cross_attn_rotary_pos_emb = None,
        self_attention_block_mask = None,
        self_attention_score_mod = None,
        cross_attention_block_mask = None,
        cross_attention_score_mod = None,
        self_attention_flash_sliding_window = None,
        cross_attention_flash_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
    ):
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            x = x * torch.sigmoid(1 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual

            if context is not None and self.cross_attend:
                if cross_attn_rotary_pos_emb is not None:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_k = cross_attn_rotary_pos_emb, context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                else:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x, varlen_metadata=varlen_metadata)
            x = x * torch.sigmoid(1 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual

        else:
            x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata))

            if context is not None and self.cross_attend:
                if cross_attn_rotary_pos_emb is not None:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_k = cross_attn_rotary_pos_emb, context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                else:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                    
            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            x = x + self.ff_scale(self.ff(self.ff_norm(x), varlen_metadata=varlen_metadata))
            

        return x
        
class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        final_cross_attn_ix=-1,
        global_cond_dim=None,
        local_add_cond_dim=None,
        modular_local_cond_configs=None,
        causal=False,
        rotary_pos_emb=True,
        cross_attn_rotary_pos_emb=False,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        num_memory_tokens=0,
        sliding_window=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        if cross_attn_rotary_pos_emb:
            self.cross_attn_rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.cross_attn_rotary_pos_emb = None

        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6)
            )

        self.final_cross_attn_ix = final_cross_attn_ix

        self.sliding_window = sliding_window

        for i in range(depth):
            should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = should_cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    local_add_cond_dim = local_add_cond_dim,
                    modular_local_cond_configs = modular_local_cond_configs,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        context = None,
        prepend_embeds = None,
        global_cond = None,
        local_add_cond = None,
        modular_local_cond = None,
        return_info = False,
        use_checkpointing = True,
        exit_layer_ix = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(batch, -1, -1)
            x = torch.cat((memory_tokens, x), dim=1)

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.cross_attn_rotary_pos_emb is not None:
            cross_attn_rotary_pos_emb = self.cross_attn_rotary_pos_emb.forward_from_seq_len(context.shape[-1])
        else:
            cross_attn_rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        # Extend padding mask for prepended tokens if provided
        extended_padding_mask = None
        varlen_metadata = None
        if padding_mask is not None:
            # Compute total prepend length (memory tokens + prepend_embeds)
            prepend_length = self.num_memory_tokens
            if prepend_embeds is not None:
                prepend_length += prepend_embeds.shape[1]

            # Prepend tokens are always valid for attention
            if prepend_length > 0:
                prepend_valid = torch.ones(batch, prepend_length, device=device, dtype=torch.bool)
                extended_padding_mask = torch.cat([prepend_valid, padding_mask], dim=-1)
            else:
                extended_padding_mask = padding_mask

            # Precompute varlen metadata once for all layers (major performance optimization)
            # Only compute if varlen attention is actually available
            if flash_attn_varlen_func is not None and index_first_axis is not None:
                varlen_metadata = precompute_varlen_metadata(extended_padding_mask)

        # Iterate over the transformer layers
        for layer_ix, layer in enumerate(self.layers):

            layer_kwargs = {
                "context": context,
                "rotary_pos_emb": rotary_pos_emb,
                "cross_attn_rotary_pos_emb": cross_attn_rotary_pos_emb,
                "global_cond": global_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "self_attention_flash_sliding_window": self.sliding_window,
                "padding_mask": extended_padding_mask,
                "varlen_metadata": varlen_metadata
            }

            if use_checkpointing:
                x = checkpoint(layer, x, **layer_kwargs, **kwargs)
            else:
                x = layer(x, **layer_kwargs, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens:, :]

                if return_info:
                    return x, info
                
                return x

        x = x[:, self.num_memory_tokens:, :]

        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x


# ===========================================================================
# DiffusionTransformer (PORT_FROM: models/dit.py)
# ===========================================================================

class DiffusionTransformer(nn.Module):
    def __init__(self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer", "mm_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        timestep_features_type: tp.Literal["learned", "expo"] = "learned",
        timestep_features_dim = 256,
        timestep_features_logsnr: bool = False,
        modular_local_cond_configs = None,
        **kwargs):

        super().__init__()

        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type
        self.timestep_features_logsnr = timestep_features_logsnr

        timestep_features_dim = timestep_features_dim

        if timestep_features_type == "expo":
            self.timestep_features = ExpoFourierFeatures(timestep_features_dim, 0.5, 10000.0)
        else:
            self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        transformer_dim_out = io_channels * patch_size

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=transformer_dim_out,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                modular_local_cond_configs=modular_local_cond_configs,
                **kwargs
            )
      
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    # Fixed logsnr normalization range: maps logsnr to [0, 1] preserving direction (t=0→0, t=1→1)
    _LOGSNR_MIN = -12.0
    _LOGSNR_MAX = 5.0
    _LOGSNR_RANGE = _LOGSNR_MAX - _LOGSNR_MIN

    def _t_to_logsnr_cond(self, t: torch.Tensor) -> torch.Tensor:
        """Convert t to normalized logsnr in [0, 1] for timestep conditioning.

        Maps t through logsnr = log((1-t)/t), clamps to fixed range,
        then normalizes to [0, 1] preserving direction (t=0→0, t=1→1).
        """
        t_clamped = t.float().clamp(1e-7, 1 - 1e-7)
        logsnr = torch.log((1 - t_clamped) / t_clamped)
        logsnr = logsnr.clamp(self._LOGSNR_MIN, self._LOGSNR_MAX)
        return ((self._LOGSNR_MAX - logsnr) / self._LOGSNR_RANGE).to(t.dtype)

    def _call_transformer(self, x, *, prepend_inputs=None, cross_attn_cond=None,
                         mask=None, prepend_mask=None, return_info=False,
                         exit_layer_ix=None, local_add_cond=None,
                         modular_local_cond=None, padding_mask=None,
                         extra_args=None, **kwargs):
        """Helper method to call transformer and handle early exit logic."""

        output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond,
                                    return_info=return_info, exit_layer_ix=exit_layer_ix,
                                    local_add_cond=local_add_cond, modular_local_cond=modular_local_cond,
                                    padding_mask=padding_mask,
                                    **(extra_args or {}), **kwargs)

        if return_info:
            output, info = output

        # Avoid postprocessing on early exit
        if exit_layer_ix is not None:
            if return_info:
                return output, info
            else:
                return output

        return (output, info) if return_info and 'info' in locals() else output

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        if local_add_cond is not None:
            local_add_cond = rearrange(local_add_cond, "b c t -> b t c")

        # Rearrange modular_local_cond tensors
        if modular_local_cond is not None:
            modular_local_cond = {
                k: rearrange(v, "b c t -> b t c")
                for k, v in modular_local_cond.items()
            }

        # Get the batch of timestep embeddings
        t_cond = self._t_to_logsnr_cond(t) if self.timestep_features_logsnr else t
        # Convert to model dtype for linear layers (t itself is kept in float32 for precision)
        # x has already been converted to model dtype in the outer forward() method
        t_cond = t_cond.to(x.dtype)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t_cond[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(2).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        result = self._call_transformer(
            x,
            prepend_inputs=prepend_inputs,
            cross_attn_cond=cross_attn_cond,
            mask=mask,
            prepend_mask=prepend_mask,
            return_info=return_info,
            exit_layer_ix=exit_layer_ix,
            local_add_cond=local_add_cond,
            modular_local_cond=modular_local_cond,
            padding_mask=padding_mask,
            extra_args=extra_args,
            **kwargs,
        )

        # Handle early exit (result contains both output and info)
        if exit_layer_ix is not None:
            return result

        output = result[0] if return_info else result
        if return_info:
            info = result[1]

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]       

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def apg_project(self, v0, v1, padding_mask=None):
        """
        Project v0 into components parallel and orthogonal to v1.

        Args:
            v0: Tensor to project (B, C, T)
            v1: Reference direction (B, C, T)
            padding_mask: Optional mask (B, T) where True = valid, False = padding.
                          If provided, only valid positions contribute to the projection.
        """
        dtype = v0.dtype
        v0, v1 = v0.float(), v1.float()

        if padding_mask is not None:
            # Expand mask to match tensor shape: (B, T) -> (B, 1, T)
            mask = padding_mask.unsqueeze(1).float()
            # Zero out padding positions for projection computation
            v0_masked = v0 * mask
            v1_masked = v1 * mask
            # Normalize only over valid positions
            v1_norm = v1_masked.norm(dim=[-1, -2], keepdim=True).clamp(min=1e-8)
            v1_normalized = v1_masked / v1_norm
            # Compute projection using masked values
            v0_parallel = (v0_masked * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized
            # Orthogonal component: subtract parallel from original (not masked) v0
            # but apply mask to ensure padding stays zero
            v0_orthogonal = (v0 - (v0 * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized) * mask
        else:
            v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
            v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
            v0_orthogonal = v0 - v0_parallel

        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        lora_interval = (0, 1),
        lora_layer_filter = "",
        lora_configs = None,
        causal=False,
        scale_phi=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        assert not causal, "Causal mode is not supported for DiffusionTransformer"

        model_dtype = next(self.parameters()).dtype

        x = x.to(model_dtype)

        # Keep t in float32: the logsnr transform log((1-t)/t) amplifies bf16
        # quantization error ~380x near t=1, causing catastrophic conditioning errors.
        # t is a 1D batch-size tensor so float32 has zero memory impact.
        t = t.float()

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if local_add_cond is not None:
            local_add_cond = local_add_cond.to(model_dtype)

        if modular_local_cond is not None:
            modular_local_cond = {k: v.to(model_dtype) for k, v in modular_local_cond.items()}

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        # LoRA interval
        if has_lora(self):
            if lora_configs is not None:
                # Multi-LoRA: per-LoRA interval and layer filter
                for lora_config in lora_configs:
                    idx = lora_config["lora_index"]
                    interval = lora_config.get("interval", (0, 1))
                    layer_filter = lora_config.get("layer_filter", "")
                    if interval[0] <= sigma[0] <= interval[1]:
                        enable_lora(self, lora_index=idx)
                        filter_lora_layers(self, layer_filter, lora_index=idx)
                    else:
                        disable_lora(self, lora_index=idx)
            else:
                # Legacy single-LoRA path
                if lora_interval[0] <= sigma[0] <= lora_interval[1]:
                    enable_lora(self)
                    filter_lora_layers(self, lora_layer_filter)
                else:
                    disable_lora(self)

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            if local_add_cond is not None:
                batch_local_add_cond = torch.cat([local_add_cond, local_add_cond], dim=0)
            else:
                batch_local_add_cond = None

            if modular_local_cond is not None:
                batch_modular_local_cond = {k: torch.cat([v, v], dim=0) for k, v in modular_local_cond.items()}
            else:
                batch_modular_local_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            if padding_mask is not None:
                batch_padding_mask = torch.cat([padding_mask, padding_mask], dim=0)
            else:
                batch_padding_mask = None

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                mask = batch_masks,
                input_concat_cond = batch_input_concat_cond,
                local_add_cond = batch_local_add_cond,
                modular_local_cond=batch_modular_local_cond,
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                padding_mask = batch_padding_mask,
                return_info = return_info,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            if self.diffusion_objective == "v":
                cond_denoised = x * alpha[:, None, None] - cond_output * sigma[:, None, None]
                uncond_denoised = x * alpha[:, None, None] - uncond_output * sigma[:, None, None]

            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                cond_denoised = x - cond_output * sigma[:, None, None]
                uncond_denoised = x - uncond_output * sigma[:, None, None]

            diff = cond_denoised - uncond_denoised
            
            if cfg_norm_threshold > 0:
                if padding_mask is not None:
                    # Only compute norm over valid positions
                    mask = padding_mask.unsqueeze(1).float()  # (B, 1, T)
                    diff_masked = diff * mask
                    diff_norm = diff_masked.norm(p=2, dim=[-1, -2], keepdim=True)
                else:
                    diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
                scale_factor = torch.minimum(torch.ones_like(diff), cfg_norm_threshold / diff_norm)
                diff *= scale_factor

            if apg_scale == 0.0:
                # Vanilla CFG: use full diff
                cfg_diff = diff
            elif apg_scale == 1.0:
                # Full APG: use only orthogonal component
                _, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
                cfg_diff = diff_orthogonal
            else:
                # Blended APG: interpolate between full diff and orthogonal
                diff_parallel, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
                cfg_diff = apg_scale * diff_orthogonal + (1 - apg_scale) * diff

            cfg_denoised = cond_denoised + (cfg_scale - 1) * cfg_diff
                    
            if self.diffusion_objective == "v":
                output = (x * alpha[:, None, None] - cfg_denoised) / sigma[:, None, None]
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                output = (x - cfg_denoised) / sigma[:, None, None]

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = output.std(dim=1, keepdim=True)
                output = scale_phi * (output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * output
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )

# ---------------------------------------------------------------------------
# vllm-omni adaptation: extend DiffusionTransformer
# ---------------------------------------------------------------------------


# Add class attributes + alias on DiffusionTransformer after it's defined.
DiffusionTransformer._repeated_blocks: ClassVar[list[str]] = ["TransformerBlock"]
DiffusionTransformer._layerwise_offload_blocks_attr: ClassVar[str] = "transformer.layers"


def _sa3_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Pattern 2 (BAGEL-style): standard loader + custom name remap.

    TODO(stable-audio-3): build the actual remap table after diffing a real
    checkpoint against dict(self.named_parameters()).keys().
    """
    params = dict(self.named_parameters())
    loaded: set[str] = set()
    for name, tensor in weights:
        mapped = _sa3_remap_weight_name(name)
        if mapped is None:
            continue
        if mapped in params:
            default_weight_loader(params[mapped], tensor)
            loaded.add(mapped)
    return loaded


def _sa3_remap_weight_name(name: str) -> str | None:
    """Map upstream weight name -> vllm-omni param name. Returns None to skip."""
    if name.startswith(("conditioner.", "pretransform.", "autoencoder.")):
        return None
    return name


DiffusionTransformer.load_weights = _sa3_load_weights
DiffusionTransformer._remap_weight_name = staticmethod(_sa3_remap_weight_name)


# Alias used by older scaffold imports
StableAudio3DiTModel = DiffusionTransformer
