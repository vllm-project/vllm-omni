# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sol-Attn sparse attention on AMD RDNA4 (gfx1201) through aiter.

aiter's ``gfx1201_sol_attention`` implements the Sol-Attn ``diag`` routing rule
(arXiv 2607.24027) on top of the ``AITER_SAGE`` kernel: per 64-row query
block it computes the selected key blocks exactly and adds a block-summary
correction for the rest. Denoise steps before ``dense_steps`` and the layers in
``dense_layers`` run the dense ``AITER_SAGE`` kernel instead. The non-video
prefix of ``AttentionMetadata.video_layout`` is the exact sink, as keys in the
kernel and as queries through a dense attention over the prefix rows.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
from vllm_omni.diffusion.attention.backends.aiter_sage import (
    AiterSageAttentionBackend,
    AiterSageAttentionImpl,
    _valid_kv_length,
)
from vllm_omni.diffusion.attention.capabilities import ExecutionContext, ExecutionPathResult, SupportStatus
from vllm_omni.diffusion.data import SolAttnSpec
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

_NAME = "AITER_SOL"


def _sol_kernel():
    # Importing aiter registers the op, with a fake implementation, so it is opaque to torch.compile.
    import aiter  # noqa: F401

    return torch.ops.aiter.gfx1201_sol_attention


class AiterSolAttentionBackend(AiterSageAttentionBackend):
    @classmethod
    def validate_available(cls) -> None:
        super().validate_available()
        import aiter

        if not hasattr(aiter, "gfx1201_sol_attention"):
            raise ValueError(
                f"{_NAME} requires an aiter build with gfx1201_sol_attention; update aiter or select AITER_SAGE."
            )

    @staticmethod
    def get_name() -> str:
        return _NAME

    @staticmethod
    def get_impl_cls() -> type[AiterSolAttentionImpl]:
        return AiterSolAttentionImpl


class AiterSolAttentionImpl(AiterSageAttentionImpl):
    _backend_name = _NAME

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        backend_kwargs = dict(backend_kwargs or {})
        config = SolAttnSpec(**backend_kwargs.pop("sol_attn", {}))
        super().__init__(
            num_heads,
            head_size,
            softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            backend_kwargs=backend_kwargs,
            **extra_impl_args,
        )
        self.config = config
        self.layer_idx = self._parse_layer_idx(prefix)

    @staticmethod
    def _parse_layer_idx(prefix: str) -> int | None:
        parts = prefix.split(".")
        for name, index in zip(parts, parts[1:]):
            if name == "blocks" and index.isdigit():
                return int(index)
        return None

    def _use_dense(self) -> bool:
        if self.layer_idx is not None and self.layer_idx in (self.config.dense_layer_indices or ()):
            return True
        if not self.config.dense_steps:
            return False
        step = get_forward_context().denoise_step_idx if is_forward_context_available() else None
        # Without a published step the guard cannot tell warmup steps apart, so it stays dense.
        return step is None or step < self.config.dense_steps

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        result = super().resolve_execution_path(context, query, key, value, attn_metadata)
        if result.support.status is not SupportStatus.SUPPORTED or self._use_dense():
            return result
        return replace(result, path="sol_sparse")

    @staticmethod
    def _prefix_len(layout: VideoTokenLayout | None) -> int | None:
        # Rows before the target video: text, conditions, references and audio.
        if layout is None:
            return None
        targets = [span.start for span in layout.video_spans if span.role == "target"]
        if targets:
            return targets[-1]
        return layout.prefix_len

    def _sink_range(self, seq_len: int) -> tuple[int, int | None]:
        # Short sequences can be shorter than the configured sink, so clamp instead of failing.
        tokens = min(self.config.sink_tokens, seq_len)
        start = self.config.sink_start
        if start is not None:
            start = min(start, seq_len - tokens)
        return tokens, start

    def _sol(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, prefix: int | None) -> torch.Tensor:
        start: int | None
        if self.config.prefix_sink and prefix is not None:
            # Sink queries also attend to every key exactly.
            tokens, start, query_sink = min(prefix, query.shape[1]), 0, True
        else:
            (tokens, start), query_sink = self._sink_range(query.shape[1]), False
        output = _sol_kernel()(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            tau=self.config.tau,
            sink_tokens=tokens,
            sink_start=start,
            softmax_scale=self.softmax_scale,
        )
        if query_sink and tokens:
            output[:, :tokens] = F.scaled_dot_product_attention(
                query[:, :tokens].transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=self.softmax_scale,
            ).transpose(1, 2)
        return output

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if self._use_dense():
            return super().forward_hip(query, key, value, attn_metadata)
        reason = self._unsupported_reason(query, key, value, attn_metadata)
        if reason:
            raise ValueError(f"{_NAME}: {reason}")
        valid = _valid_kv_length(attn_metadata)
        prefix = self._prefix_len(attn_metadata.video_layout if attn_metadata is not None else None)
        if valid is None or valid == key.shape[1]:
            return self._sol(query, key, value, prefix)
        output = torch.zeros(query.shape, dtype=query.dtype, device=query.device)
        output[:, :valid] = self._sol(query[:, :valid], key[:, :valid], value[:, :valid], prefix)
        return output
