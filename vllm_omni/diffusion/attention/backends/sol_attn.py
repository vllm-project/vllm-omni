# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sol-Attn sparse attention backend for diffusion transformers.

This backend registers the NVlabs Sol-Attn training-free sparse attention
kernel (https://nvlabs.github.io/Sana/Sol-Attn/) as a diffusion attention
backend. Sol-Attn routes attention blocks on the fly during an online-softmax
pass instead of materializing a full proxy score map, which makes it a good
fit for packed-sequence DiTs whose attention is sequence-bound (e.g.
MiniMax-H3: ~50 layers x tens of thousands of packed tokens per denoise step).

The upstream ``sol_attn`` package is imported lazily so that selecting the
backend without it fails with a clear message at model-build time while other
backends remain unaffected. The kernel requires BF16 activations and
``head_dim=128`` (CUDA only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

logger = init_logger(__name__)

_SOL_ATTN_HEAD_DIM = 128


def _parse_layer_ranges(spec: str | int | None) -> frozenset[int]:
    """Parse a dense-layer spec such as ``"0,1"`` or ``"3-5"`` into a set."""
    if spec is None:
        return frozenset()
    if isinstance(spec, int):
        return frozenset({spec})
    layers: set[int] = set()
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return frozenset(layers)


def _resolve_kv_splits(q: torch.Tensor, kv_splits: int | str | None) -> int:
    """Resolve the ``kv_splits`` knob, auto-tuning only for known fast paths."""
    if kv_splits not in (None, "auto"):
        return int(kv_splits)
    arch = tuple(torch.cuda.get_device_capability(q.device))
    if arch == (9, 0) and q.shape[1] >= 65536:
        try:
            import cuda.bindings.driver  # noqa: F401
            import cutlass.cute  # noqa: F401

            return 4
        except ImportError:
            pass
    return 1


@dataclass(frozen=True)
class SolAttnConfig:
    """Typed runtime config for the Sol-Attn kernel, defaults match upstream."""

    tau: float = 1.0
    thresh_type: str = "diag"
    kv_splits: int | str = "auto"
    sink_tokens: int = 0
    sink_start: int | None = 0
    dense_steps: int = 10
    dense_layers: frozenset[int] = frozenset({0, 1})

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> SolAttnConfig:
        raw = (backend_kwargs or {}).get("sol_attn") or {}
        if not isinstance(raw, dict):
            raise TypeError(f"sol_attn config must be a dict, got {type(raw)!r}")
        return cls(
            tau=float(raw.get("tau", 1.0)),
            thresh_type=str(raw.get("thresh_type", "diag")),
            kv_splits=raw.get("kv_splits", "auto"),
            sink_tokens=int(raw.get("sink_tokens", 0)),
            sink_start=None if raw.get("sink_start") is None else int(raw["sink_start"]),
            dense_steps=int(raw.get("dense_steps", 10)),
            dense_layers=_parse_layer_ranges(raw.get("dense_layers", "0,1")),
        )


class SolAttnBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_prefix_kv_slicing: bool = True
    supported_platforms: tuple[str, ...] = ("cuda",)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_SOL_ATTN_HEAD_DIM]

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl


class SolAttnImpl(AttentionImpl):
    """Packed-varlen Sol-Attn kernel with an exact dense fallback.

    Dense guard steps use packed-varlen FlashAttention except on SM120, where
    the CuTe varlen kernel cannot compile MiniMax-H3's packed shape. SM120 uses
    the platform-native cuDNN dense backend while sparse steps still use Sol.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        del qkv_layout, extra_impl_args
        if head_size != _SOL_ATTN_HEAD_DIM:
            raise ValueError(f"Sol-Attn requires head_size={_SOL_ATTN_HEAD_DIM}, got {head_size}")
        if causal:
            raise ValueError("SOL_ATTN does not support causal attention; select a dense backend for causal roles")
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError(
                f"SOL_ATTN does not support GQA/MQA; num_kv_heads ({num_kv_heads}) must equal num_heads ({num_heads})"
            )
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.config = SolAttnConfig.from_backend_kwargs(backend_kwargs)
        self._validate_parallel_config()
        self._validate_kv_splits()
        self.layer_idx = self._parse_layer_idx(prefix)
        self._cudnn_dense_fallback = CuDNNAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

    @staticmethod
    def _validate_parallel_config() -> None:
        config = get_current_diffusion_config_or_none()
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if ring_degree > 1:
            raise ValueError(
                "SOL_ATTN is not compatible with ring sequence parallelism "
                f"(ring_degree={ring_degree}): the sparse kernel needs the whole key sequence. "
                "Use Ulysses SP (ring_degree=1) instead."
            )

    def _validate_kv_splits(self) -> None:
        if self.config.kv_splits not in (2, 4):
            return
        capability = tuple(torch.cuda.get_device_capability())
        if capability != (9, 0):
            raise ValueError(
                f"SOL_ATTN kv_splits={self.config.kv_splits} is supported on SM90 only; "
                f"current device is SM{capability[0]}{capability[1]}. Use kv_splits=1 or 'auto'."
            )

    @staticmethod
    def _parse_layer_idx(prefix: str) -> int | None:
        match = re.search(r"blocks\.(\d+)", prefix)
        if match is None:
            return None
        return int(match.group(1))

    def _current_denoise_step(self) -> int | None:
        if not is_forward_context_available():
            return None
        return getattr(get_forward_context(), "denoise_step_idx", None)

    def _should_use_dense(self) -> bool:
        step = self._current_denoise_step()
        if step is not None and step < self.config.dense_steps:
            return True
        if self.layer_idx is not None and self.layer_idx in self.config.dense_layers:
            return True
        return False

    @staticmethod
    def _requires_sm120_dense_fallback(device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 0)

    @staticmethod
    def _clamp_sink_range(
        used: int,
        sink_start: int | None,
        sink_tokens: int,
    ) -> tuple[int, int]:
        """Clamp the exact-KV sink range to ``[0, used]``.

        The kernel requires ``sink_start + sink_tokens <= T``. Short sequences
        (e.g. the text-refiner attention, which runs on the text rows only) can
        be shorter than the configured sink, so clamp instead of failing.
        """
        start = 0 if sink_start is None else min(int(sink_start), used)
        tokens = min(int(sink_tokens), used - start)
        return start, tokens

    @staticmethod
    def _used_length(attn_metadata: AttentionMetadata | None, seq_len: int) -> int:
        if attn_metadata is None:
            return seq_len
        used = attn_metadata.extra.get("max_seqlen_q")
        if used is None:
            used = attn_metadata.extra.get("valid_kv_length")
        if used is None:
            return seq_len
        return min(int(used), seq_len)

    def _forward_dense_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Run the exact packed dense fallback selected for this platform."""
        if self._requires_sm120_dense_fallback(query.device):
            logger.info_once("Using cuDNN for the Sol-Attn dense guard on SM120")
            return self._cudnn_dense_fallback.forward_cuda(
                query,
                key,
                value,
                attn_metadata,
            )

        from vllm_omni.diffusion.attention.backends.utils.fa import (
            flash_attn_varlen_func,
        )

        if flash_attn_varlen_func is None:
            raise ImportError("Sol-Attn dense fallback requires flash_attn_varlen_func")
        extra = attn_metadata.extra
        out = flash_attn_varlen_func(
            q=query.flatten(0, 1),
            k=key.flatten(0, 1),
            v=value.flatten(0, 1),
            cu_seqlens_q=extra["cu_seqlens_q"],
            cu_seqlens_k=extra["cu_seqlens_k"],
            max_seqlen_q=extra["max_seqlen_q"],
            max_seqlen_k=extra["max_seqlen_k"],
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.reshape_as(query)

    def _forward_dense_batched(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Dense fallback for a plain (B, S, H, D) batch with no packed metadata."""
        batch_size, seq_len = query.shape[:2]
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            device=query.device,
            dtype=torch.int32,
        )
        out = self._forward_dense_varlen(
            query,
            key,
            value,
            AttentionMetadata(
                extra={
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": seq_len,
                    "max_seqlen_k": seq_len,
                }
            ),
        )
        return out.reshape(batch_size, seq_len, *out.shape[2:])

    def _forward_dense_queries(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        start: int,
        tokens: int,
    ) -> torch.Tensor:
        """Recompute selected query rows densely against every key/value row."""
        return F.scaled_dot_product_attention(
            query[:, start : start + tokens].transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
        ).transpose(1, 2)

    def _forward_sol_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        batch_size, seq_len = query.shape[:2]
        if batch_size != 1:
            raise NotImplementedError(
                "Sol-Attn sparse attention currently requires batch size 1 "
                "(packed single-document layout, e.g. MiniMax-H3)."
            )
        # The packed layout may append an alignment-padding document. The
        # kernel has no cu_seqlens support, so slice to the real length first;
        # the padding rows are masked by the model and their outputs discarded.
        used = self._used_length(attn_metadata, seq_len)
        sink_start, sink_tokens = self._clamp_sink_range(
            used,
            self.config.sink_start,
            self.config.sink_tokens,
        )
        q = query[:, :used].contiguous()
        k = key[:, :used].contiguous()
        v = value[:, :used].contiguous()
        if q.dtype != torch.bfloat16:
            raise TypeError(
                f"Sol-Attn requires bfloat16 activations, got {q.dtype}. "
                "Select a backend that supports the model dtype instead."
            )
        out = sol_attn(
            q,
            k,
            v,
            scale=self.softmax_scale,
            tau=self.config.tau,
            thresh_type=self.config.thresh_type,
            kv_splits=_resolve_kv_splits(q, self.config.kv_splits),
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        if sink_tokens:
            out[:, sink_start : sink_start + sink_tokens] = self._forward_dense_queries(
                q,
                k,
                v,
                start=sink_start,
                tokens=sink_tokens,
            )
        if out.shape[1] < seq_len:
            padded = torch.zeros_like(query)
            padded[:, :used] = out
            out = padded
        return out

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if self._should_use_dense():
            if attn_metadata is not None and "cu_seqlens_q" in attn_metadata.extra:
                return self._forward_dense_varlen(query, key, value, attn_metadata)
            return self._forward_dense_batched(query, key, value)
        return self._forward_sol_attn(query, key, value, attn_metadata)
