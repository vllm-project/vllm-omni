# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Ported from the Apache-2.0 SeedVR2 reference implementation:
#   https://github.com/ByteDance-Seed/SeedVR  (models/dit_v2)
#   https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (src/models/dit_3b)
"""SeedVR2 ``NaDiT`` diffusion transformer with window-local attention.

The module layout deliberately mirrors the reference implementation (including
the ``MMModule`` vid/txt/shared parameter split) so the released checkpoint
loads with its original parameter names.

At SP=1, the window planner preserves the reference regular/shifted layouts.
At SP>1, sequence rows stay sharded through the MLP, while attention exchanges
QKV into head shards so each rank can process every window. See
:mod:`vllm_omni.diffusion.models.seedvr2.ulysses` for that exchange.

Reference quirk reproduced on purpose: ``vid_out_ada`` is declared with
``layers=["out"]``, whose ``(d, l, g)`` re-grouping of the 6*dim timestep
embedding is arithmetically incompatible with its 1-D parameters.  In the
reference the incompatible value is never used, because the module asks the
forward-wide cache for ``emb_repeat_0_vid`` -- a key the first block already
wrote -- and therefore applies the *block* attention modulation together with
the learned ``out_scale`` / ``out_shift``.  This port applies that effective
modulation directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.seedvr2.na_ops import (
    LocalWindowContext,
    SeedVR2WindowRuntime,
    pack_joint_windows,
    unpack_joint_windows,
)
from vllm_omni.diffusion.models.seedvr2.parallel import validate_seedvr2_parallel_config
from vllm_omni.diffusion.models.seedvr2.rope import NaMMRotaryEmbedding3d
from vllm_omni.diffusion.models.seedvr2.ulysses import SeedVR2UlyssesRuntime
from vllm_omni.diffusion.models.seedvr2.window_geometry import (
    DEFAULT_WINDOW,
    DEFAULT_WINDOW_METHODS,
)
from vllm_omni.diffusion.models.seedvr2.window_sp import global_window_mean

logger = init_logger(__name__)

# Reference checkpoint hyper-parameters for the released 3B model.
SEEDVR2_3B_CONFIG: dict = {
    "vid_in_channels": 33,
    "vid_out_channels": 16,
    "vid_dim": 2560,
    "txt_in_dim": 5120,
    "heads": 20,
    "head_dim": 128,
    "expand_ratio": 4,
    "norm_eps": 1e-5,
    "patch_size": (1, 2, 2),
    "num_layers": 32,
    "mm_layers": 10,
    "window": DEFAULT_WINDOW,
    "window_method": DEFAULT_WINDOW_METHODS,
    "rope_dim": 128,
    "vid_out_norm": True,
}


# =============================================================================
# Reference-faithful building blocks
# =============================================================================


class MMArg:
    """Pair of per-stream values (video / text) used by :class:`MMModule`."""

    __slots__ = ("vid", "txt")

    def __init__(self, vid, txt) -> None:
        self.vid = vid
        self.txt = txt


class MMModule(nn.Module):
    """Apply a module to the video and text streams, optionally sharing weights.

    ``shared_weights=True`` builds a single submodule used for both streams
    (the reference's ``all`` prefix in the checkpoint); ``vid_only=True`` drops
    the text branch entirely.
    """

    def __init__(self, factory, dims: MMArg, *, shared_weights: bool = False, vid_only: bool = False) -> None:
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if shared_weights:
            self.all = factory(dims.vid)
        else:
            self.vid = factory(dims.vid)
            self.txt = None if vid_only else factory(dims.txt)

    def forward(self, vid: torch.Tensor, txt: torch.Tensor | None, *args, **kwargs):
        vid_module = self.all if self.shared_weights else self.vid
        vid = vid_module(vid, *args, **kwargs)
        if not self.vid_only and txt is not None:
            txt_module = self.all if self.shared_weights else self.txt
            txt = txt_module(txt, *args, **kwargs)
        return vid, txt


class RMSNorm(nn.Module):
    """Reference ``CustomRMSNorm``: normalise in the input dtype, optional affine."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x


class AdaSingle(nn.Module):
    """Timestep modulation with per-layer shift / scale / gate parameters."""

    def __init__(self, dim: int, emb_dim: int, layers: Sequence[str], modes: Sequence[str] = ("in", "out")) -> None:
        super().__init__()
        if emb_dim != 6 * dim:
            raise ValueError(f"AdaSingle requires emb_dim == 6 * dim, got {emb_dim} != {6 * dim}")
        self.dim = int(dim)
        self.emb_dim = int(emb_dim)
        self.layers = list(layers)
        modes = set(modes)
        for layer in self.layers:
            if "in" in modes:
                self.register_parameter(f"{layer}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(f"{layer}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1))
            if "out" in modes:
                self.register_parameter(f"{layer}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def slice(self, emb: torch.Tensor, layer: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(shift, scale, gate)`` embedding slices for ``layer``, each ``[1, dim]``."""
        view = emb.view(emb.shape[0], self.dim, len(self.layers), 3)[..., self.layers.index(layer), :]
        shift, scale, gate = view.unbind(-1)
        return shift, scale, gate

    def forward(self, hid: torch.Tensor, emb: torch.Tensor, layer: str, mode: str) -> torch.Tensor:
        shift_a, scale_a, gate_a = self.slice(emb, layer)
        shift_a = shift_a.to(hid.dtype)
        scale_a = scale_a.to(hid.dtype)
        gate_a = gate_a.to(hid.dtype)
        if mode == "in":
            return hid * (scale_a + getattr(self, f"{layer}_scale")) + (shift_a + getattr(self, f"{layer}_shift"))
        if mode == "out":
            gate_b = getattr(self, f"{layer}_gate", None)
            return hid * (gate_a + gate_b if gate_b is not None else gate_a)
        raise NotImplementedError(mode)


class OutAda(nn.Module):
    """The released ``vid_out_ada`` parameters (see the module docstring)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_shift = nn.Parameter(torch.zeros(dim))
        self.out_scale = nn.Parameter(torch.ones(dim))


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer SiLU MLP."""

    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    @staticmethod
    def _sinusoidal(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, timestep, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        timestep = timestep.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        emb = self._sinusoidal(timestep, self.sinusoidal_dim).to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        return self.proj_out(emb)


class SwiGLUMLP(nn.Module):
    """Reference SwiGLU MLP (``multiple_of=256``, no biases)."""

    def __init__(self, dim: int, expand_ratio: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden = int(2 * dim * expand_ratio / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden, bias=False)
        self.proj_in = nn.Linear(dim, hidden, bias=False)
        self.proj_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))


# =============================================================================
# Window attention
# =============================================================================


def grouped_window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: LocalWindowContext,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Exact window-local attention without a varlen kernel.

    Windows are packed back-to-back in the joint sequence, so this groups the
    equal-length windows and runs one batched SDPA per group (ragged window
    sizes keep the group count small).  It is the portable path used when the
    selected backend has no packed-varlen entry point.
    """
    if ctx.local_windows == 0:
        return torch.empty_like(q)
    lengths = ctx.joint_lengths
    offsets = ctx.joint_cu_seqlens.to(torch.int64)
    out = torch.empty_like(q)
    for length in torch.unique(lengths).tolist():
        window_ids = torch.nonzero(lengths == length, as_tuple=False).flatten().tolist()
        rows = torch.cat(
            [torch.arange(int(offsets[i]), int(offsets[i]) + int(length), device=q.device) for i in window_ids]
        )
        num = len(window_ids)
        qq = q.index_select(0, rows).view(num, int(length), q.shape[1], q.shape[2]).transpose(1, 2)
        kk = k.index_select(0, rows).view(num, int(length), k.shape[1], k.shape[2]).transpose(1, 2)
        vv = v.index_select(0, rows).view(num, int(length), v.shape[1], v.shape[2]).transpose(1, 2)
        attended = F.scaled_dot_product_attention(qq, kk, vv, scale=softmax_scale)
        out.index_copy_(0, rows, attended.transpose(1, 2).reshape(-1, q.shape[1], q.shape[2]))
    return out


class NaSwinAttention(nn.Module):
    """Joint video+text window attention with a globally averaged text output."""

    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm_eps: float,
        rope_dim: int,
        shared_weights: bool,
        use_varlen_kernel: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.inner_dim = int(inner_dim)
        self.softmax_scale = 1.0 / math.sqrt(head_dim)
        dims = MMArg(vid_dim, txt_dim)
        self.proj_qkv = MMModule(
            lambda d: nn.Linear(int(d), 3 * inner_dim, bias=qk_bias), dims, shared_weights=shared_weights
        )
        self.proj_out = MMModule(lambda d: nn.Linear(inner_dim, int(d)), dims, shared_weights=shared_weights)
        self.norm_q = MMModule(
            lambda d: RMSNorm(int(d), eps=qk_norm_eps, elementwise_affine=True),
            MMArg(head_dim, head_dim),
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            lambda d: RMSNorm(int(d), eps=qk_norm_eps, elementwise_affine=True),
            MMArg(head_dim, head_dim),
            shared_weights=shared_weights,
        )
        self.rope = NaMMRotaryEmbedding3d(rotary_dim=rope_dim, num_axes=3)
        # The model owns window/head exchange; shared attention dispatch must
        # not apply another sequence-parallel strategy.
        self.attention = Attention(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            role="self",
            skip_sequence_parallel=True,
        )
        # ``use_varlen_kernel`` is a request, not a capability proof: a backend
        # that ignores ``cu_seqlens`` would attend across every local window
        # (and the replicated text stream), which is silently wrong and differs
        # by SP degree.  Resolve the request against the selected backend once.
        requested_varlen = bool(use_varlen_kernel)
        backend = self.attention.attn_backend
        supports_varlen = bool(backend is not None and backend.supports_multi_doc_packed_varlen())
        self.use_varlen_kernel = requested_varlen and supports_varlen
        self.attention_backend_name = backend.get_name() if backend is not None else None
        self.attention_path = "packed_varlen" if self.use_varlen_kernel else "grouped_sdpa"
        self.varlen_fallback_reason: str | None = None
        if requested_varlen and not supports_varlen:
            backend_name = self.attention_backend_name or "custom_attention"
            # Stable message without a layer id: 32 layers must warn once.
            self.varlen_fallback_reason = f"backend {backend_name} does not support multi-document packed varlen"
            logger.warning_once(
                "SeedVR2: attention backend %s does not support multi-document packed varlen; "
                "using grouped window SDPA.",
                backend_name,
            )
        self.attention_stats = {"packed_varlen_calls": 0, "grouped_sdpa_calls": 0, "no_local_windows_calls": 0}

    def _split_heads(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return qkv.view(qkv.shape[0], 3, self.heads, self.head_dim).unbind(1)

    def _apply_rope(self, vid_q, vid_k, txt_q, txt_k, ctx: LocalWindowContext):
        device = vid_q.device
        vid_freqs = self.rope.window_freqs_batch(ctx.window_shapes, ctx.text_len, device=device, dtype=torch.float32)
        txt_freqs = self.rope.text_freqs(ctx.text_len, device=device, dtype=torch.float32)
        if vid_freqs.shape[0] != vid_q.shape[0]:
            raise ValueError(f"window RoPE covers {vid_freqs.shape[0]} rows but the video shard has {vid_q.shape[0]}")
        return self.rope(vid_q, vid_k, vid_freqs, txt_q, txt_k, txt_freqs)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        ctx: LocalWindowContext,
        runtime: SeedVR2WindowRuntime | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        if isinstance(runtime, SeedVR2UlyssesRuntime):
            vid_q, vid_k, vid_v = runtime.to_heads(
                vid_qkv.view(vid.shape[0], 3, self.heads, self.head_dim), ctx
            ).unbind(1)
            txt_q, txt_k, txt_v = runtime.text_heads(txt_qkv.view(txt.shape[0], 3, self.heads, self.head_dim)).unbind(1)
        else:
            vid_q, vid_k, vid_v = self._split_heads(vid_qkv)
            txt_q, txt_k, txt_v = self._split_heads(txt_qkv)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)
        vid_q, vid_k, txt_q, txt_k = self._apply_rope(vid_q, vid_k, txt_q, txt_k, ctx)

        joint_q = pack_joint_windows(vid_q, txt_q, ctx)
        joint_k = pack_joint_windows(vid_k, txt_k, ctx)
        joint_v = pack_joint_windows(vid_v, txt_v, ctx)

        if not ctx.local_windows:
            self.attention_stats["no_local_windows_calls"] += 1
        elif self.use_varlen_kernel:
            self.attention_stats["packed_varlen_calls"] += 1
        else:
            self.attention_stats["grouped_sdpa_calls"] += 1

        if self.use_varlen_kernel and ctx.local_windows:
            metadata = AttentionMetadata(
                extra={
                    "cu_seqlens_q": ctx.joint_cu_seqlens,
                    "cu_seqlens_k": ctx.joint_cu_seqlens,
                    "max_seqlen_q": ctx.max_joint_len,
                    "max_seqlen_k": ctx.max_joint_len,
                }
            )
            joint_out = self.attention(
                joint_q.unsqueeze(0), joint_k.unsqueeze(0), joint_v.unsqueeze(0), metadata
            ).squeeze(0)
        else:
            joint_out = grouped_window_sdpa(joint_q, joint_k, joint_v, ctx, softmax_scale=self.softmax_scale)

        vid_out, txt_windows = unpack_joint_windows(joint_out, ctx)
        attention_dim = vid_out.shape[-2] * vid_out.shape[-1]
        if ctx.local_windows:
            local_text_sum = txt_windows.reshape(ctx.local_windows, ctx.text_len, attention_dim).sum(0)
        else:
            # Ranks without windows still join the text reduction; the dtype must
            # match the other ranks' contribution exactly (a mismatch changes the
            # collective's dtype and deadlocks NCCL).
            local_text_sum = torch.zeros((ctx.text_len, self.inner_dim), device=vid.device, dtype=vid.dtype)
        if runtime is not None:
            txt_out = runtime.reduce_text(local_text_sum, ctx.global_windows).to(vid.dtype)
        else:
            # Same reduction as the runtime path, without a collective.
            txt_out = global_window_mean(local_text_sum, ctx.global_windows, group=None, dtype=vid.dtype)

        if isinstance(runtime, SeedVR2UlyssesRuntime):
            vid_out = runtime.from_heads(vid_out, ctx)
        return self.proj_out(vid_out.reshape(-1, self.inner_dim), txt_out)


# =============================================================================
# Transformer block
# =============================================================================


class NaMMSRTransformerBlock(nn.Module):
    """Reference ``NaMMSRTransformerBlock``: modulated attention + SwiGLU MLP."""

    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm_eps: float,
        qk_bias: bool,
        mlp_type: str,
        shared_weights: bool,
        rope_dim: int,
        is_last_layer: bool,
        use_varlen_kernel: bool = True,
    ) -> None:
        super().__init__()
        if mlp_type != "swiglu":
            raise NotImplementedError(f"unsupported mlp_type {mlp_type!r} for SeedVR2")
        dims = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(
            lambda d: RMSNorm(int(d), eps=norm_eps, elementwise_affine=False), dims, shared_weights=shared_weights
        )
        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm_eps=norm_eps,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            use_varlen_kernel=use_varlen_kernel,
        )
        self.mlp_norm = MMModule(
            lambda d: RMSNorm(int(d), eps=norm_eps, elementwise_affine=False),
            dims,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.mlp = MMModule(
            lambda d: SwiGLUMLP(int(d), expand_ratio), dims, shared_weights=shared_weights, vid_only=is_last_layer
        )
        self.ada = MMModule(
            lambda d: AdaSingle(int(d), emb_dim, layers=["attn", "mlp"]),
            dims,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.is_last_layer = bool(is_last_layer)

    def _modulate(
        self, vid: torch.Tensor, txt: torch.Tensor | None, emb: torch.Tensor, layer: str, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        vid_ada = self.ada.all if self.ada.shared_weights else self.ada.vid
        vid = vid_ada(vid, emb, layer, mode)
        if txt is not None and not self.ada.vid_only:
            txt_ada = self.ada.all if self.ada.shared_weights else self.ada.txt
            txt = txt_ada(txt, emb, layer, mode)
        return vid, txt

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        emb: torch.Tensor,
        ctx: LocalWindowContext,
        runtime: SeedVR2WindowRuntime | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_norm, txt_norm = self.attn_norm(vid, txt)
        vid_norm, txt_norm = self._modulate(vid_norm, txt_norm, emb, "attn", "in")
        vid_attn, txt_attn = self.attn(vid_norm, txt_norm, ctx, runtime)
        vid_attn, txt_attn = self._modulate(vid_attn, txt_attn, emb, "attn", "out")
        vid_attn = vid_attn + vid
        txt_attn = txt_attn + txt

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self._modulate(vid_mlp, txt_mlp, emb, "mlp", "in")
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self._modulate(vid_mlp, txt_mlp, emb, "mlp", "out")
        return vid_mlp + vid_attn, txt_mlp + txt_attn


# =============================================================================
# NaDiT
# =============================================================================


class NaDiTOutput:
    """Container matching the reference ``NaDiTOutput``."""

    def __init__(self, vid_sample: torch.Tensor) -> None:
        self.vid_sample = vid_sample


class SeedVR2NaDiT(nn.Module):
    """SeedVR2 3B ``NaDiT`` transformer with model-owned window attention."""

    def __init__(
        self,
        *,
        vid_in_channels: int = 33,
        vid_out_channels: int = 16,
        vid_dim: int = 2560,
        txt_in_dim: int = 5120,
        emb_dim: int | None = None,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        norm_eps: float = 1e-5,
        patch_size: Sequence[int] = (1, 2, 2),
        num_layers: int = 32,
        mm_layers: int = 10,
        mlp_type: str = "swiglu",
        window: Sequence[int] = DEFAULT_WINDOW,
        window_method: Sequence[str] = DEFAULT_WINDOW_METHODS,
        rope_type: str = "mmrope3d",
        rope_dim: int = 128,
        vid_out_norm: bool = True,
        use_varlen_kernel: bool = True,
    ) -> None:
        super().__init__()
        if rope_type != "mmrope3d":
            raise NotImplementedError(f"unsupported rope_type {rope_type!r} for SeedVR2")
        txt_dim = vid_dim
        emb_dim = emb_dim or 6 * vid_dim
        self.patch_size = tuple(int(v) for v in patch_size)
        if self.patch_size[0] != 1:
            raise NotImplementedError("only the released temporal patch size 1 is supported")
        self.vid_dim = int(vid_dim)
        self.txt_dim = int(txt_dim)
        self.num_layers = int(num_layers)
        self.window = tuple(int(v) for v in window)
        self.window_method = tuple(window_method)

        self.vid_in = _NaPatchIn(in_channels=vid_in_channels, patch_size=self.patch_size, dim=vid_dim)
        self.txt_in = nn.Linear(txt_in_dim, txt_dim) if txt_in_dim != txt_dim else nn.Identity()
        self.emb_in = TimeEmbedding(sinusoidal_dim=256, hidden_dim=max(vid_dim, txt_dim), output_dim=emb_dim)

        self.blocks = nn.ModuleList(
            [
                NaMMSRTransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm_eps=norm_eps,
                    qk_bias=False,
                    mlp_type=mlp_type,
                    shared_weights=not (index < mm_layers),
                    rope_dim=rope_dim,
                    is_last_layer=(index == num_layers - 1),
                    use_varlen_kernel=use_varlen_kernel,
                )
                for index in range(num_layers)
            ]
        )

        self.vid_out_norm = RMSNorm(vid_dim, eps=norm_eps, elementwise_affine=True) if vid_out_norm else None
        self.vid_out_ada = OutAda(vid_dim)
        self.vid_out = _NaPatchOut(out_channels=vid_out_channels, patch_size=self.patch_size, dim=vid_dim)

    # -- runtime -----------------------------------------------------------
    def build_runtime(
        self,
        token_grid: tuple[int, int, int],
        *,
        text_len: int,
        group=None,
        world_size: int = 1,
        rank: int = 0,
        parallel_config: DiffusionParallelConfig | None = None,
        ulysses: bool = False,
    ) -> SeedVR2WindowRuntime:
        """Create the window-attention runtime for one request."""
        if parallel_config is not None:
            validate_seedvr2_parallel_config(parallel_config)
            sp = get_sp_group()
            if sp.world_size != parallel_config.sequence_parallel_size:
                raise ValueError("SeedVR2 SP group size does not match its parallel configuration")
            group, world_size, rank = sp.device_group, sp.world_size, sp.rank_in_group
        if ulysses:
            return SeedVR2UlyssesRuntime(
                token_grid,
                text_len=text_len,
                heads=self.blocks[0].attn.heads,
                group=group,
                world_size=world_size,
                rank=rank,
                window=self.window,
                methods=self.window_method,
                num_layers=self.num_layers,
            )
        return SeedVR2WindowRuntime(
            token_grid,
            text_len=text_len,
            group=group,
            world_size=world_size,
            rank=rank,
            window=self.window,
            methods=self.window_method,
            num_layers=self.num_layers,
        )

    def token_grid_for(self, vid_shape: torch.Tensor) -> tuple[int, int, int]:
        frames, height, width = (int(v) for v in vid_shape[0].tolist())
        t, h, w = self.patch_size
        if t > 1 and frames % t != 1:
            raise ValueError(f"frame count {frames} must satisfy frames % {t} == 1")
        return frames // t, height // h, width // w

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep,
        runtime: SeedVR2WindowRuntime | None = None,
    ) -> NaDiTOutput:
        """Run the transformer.

        ``vid`` is the flattened pre-patchify latent ``[T*H*W, C]`` with
        ``vid_shape`` holding the raw latent ``(T, H, W)``; ``txt`` is
        ``[L, txt_in_dim]``. The runtime selects local windows at SP=1 or
        sequence rows with head-sharded attention at SP>1.
        """
        weight = next(self.vid_in.parameters())
        frames, height, width = (int(v) for v in vid_shape[0].tolist())
        token_grid = self.token_grid_for(vid_shape)

        txt_tokens = self.txt_in(txt.to(weight.dtype))
        text_len = int(txt_tokens.shape[0])

        canonical_rows, _ = patchify(vid, (frames, height, width), self.patch_size)
        canonical_rows = canonical_rows.to(weight.dtype)

        if runtime is None:
            runtime = self.build_runtime(token_grid, text_len=text_len, world_size=1, rank=0)

        first_layout = runtime.layout_for_layer(0)
        local_rows = runtime.local_rows_for(canonical_rows, first_layout)
        vid_hidden = self.vid_in.proj(local_rows)
        emb = self.emb_in(timestep, device=vid_hidden.device, dtype=vid_hidden.dtype)

        current = first_layout.key
        for index, block in enumerate(self.blocks):
            layout = runtime.layout_for_layer(index)
            vid_hidden = runtime.ensure_layout(vid_hidden, current, layout.key)
            current = layout.key
            ctx = runtime.context(layout, vid_hidden.device)
            vid_hidden, txt_tokens = block(vid_hidden, txt_tokens, emb, ctx, runtime)

        vid_hidden = self._output_projection(vid_hidden, emb)
        vid_hidden = runtime.to_canonical_rows(vid_hidden, runtime.layout_for_layer(self.num_layers - 1))
        return NaDiTOutput(vid_sample=unpatchify(vid_hidden, token_grid, self.patch_size))

    # -- attention path reporting -----------------------------------------
    def attention_path_summary(self) -> dict:
        """Requested vs resolved attention path, backends and per-path call counts."""
        layers_per_path: dict[str, int] = {}
        backends: set[str] = set()
        fallbacks: set[str] = set()
        stats = {"packed_varlen_calls": 0, "grouped_sdpa_calls": 0, "no_local_windows_calls": 0}
        for block in self.blocks:
            attn = block.attn
            layers_per_path[attn.attention_path] = layers_per_path.get(attn.attention_path, 0) + 1
            if attn.attention_backend_name:
                backends.add(attn.attention_backend_name)
            if attn.varlen_fallback_reason:
                fallbacks.add(attn.varlen_fallback_reason)
            for key, value in attn.attention_stats.items():
                stats[key] += value
        return {
            "layers_per_path": layers_per_path,
            "backend_names": sorted(backends),
            "varlen_fallback_reasons": sorted(fallbacks),
            **stats,
        }

    def reset_attention_stats(self) -> None:
        for block in self.blocks:
            for key in block.attn.attention_stats:
                block.attn.attention_stats[key] = 0

    # -- internals ---------------------------------------------------------
    def _output_projection(self, vid_hidden: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.vid_out_norm is not None:
            vid_hidden = self.vid_out_norm(vid_hidden)
            block_ada = self.blocks[0].ada
            ada = block_ada.all if block_ada.shared_weights else block_ada.vid
            shift_a, scale_a, _ = ada.slice(emb, "attn")
            vid_hidden = vid_hidden * (scale_a.to(vid_hidden.dtype) + self.vid_out_ada.out_scale) + (
                shift_a.to(vid_hidden.dtype) + self.vid_out_ada.out_shift
            )
        return self.vid_out.proj(vid_hidden)


# =============================================================================
# Patch in / out
# =============================================================================


class _NaPatchIn(nn.Module):
    """Reference ``NaPatchIn``: ``(T t)(H h)(W w) c -> T H W (t h w c)`` + Linear."""

    def __init__(self, in_channels: int, patch_size: Sequence[int], dim: int) -> None:
        super().__init__()
        t, h, w = (int(v) for v in patch_size)
        self.patch_size = (t, h, w)
        self.proj = nn.Linear(in_channels * t * h * w, dim)


class _NaPatchOut(nn.Module):
    """Reference ``NaPatchOut``: Linear then ``T H W (t h w c) -> (T t)(H h)(W w) c``."""

    def __init__(self, out_channels: int, patch_size: Sequence[int], dim: int) -> None:
        super().__init__()
        t, h, w = (int(v) for v in patch_size)
        self.patch_size = (t, h, w)
        self.proj = nn.Linear(dim, out_channels * t * h * w)


def patchify(vid: torch.Tensor, shape: tuple[int, int, int], patch_size: Sequence[int]):
    """Patch embedding layout of the reference (token-major rows)."""
    frames, height, width = (int(v) for v in shape)
    t, h, w = (int(v) for v in patch_size)
    if t != 1:
        raise NotImplementedError("only temporal patch size 1 is supported")
    channels = vid.shape[-1]
    grid = vid.reshape(frames // t, t, height // h, h, width // w, w, channels)
    rows = grid.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, t * h * w * channels)
    return rows, (frames // t, height // h, width // w)


def unpatchify(rows: torch.Tensor, token_grid: tuple[int, int, int], patch_size: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`patchify`: ``[N, t*h*w*c] -> [T*H*W, c]`` (reference order)."""
    t, h, w = (int(v) for v in patch_size)
    frames, height, width = (int(v) for v in token_grid)
    channels = rows.shape[-1] // (t * h * w)
    grid = rows.reshape(frames, height, width, t, h, w, channels)
    return grid.permute(0, 3, 1, 4, 2, 5, 6).reshape(frames * t * height * h * width * w, channels)
