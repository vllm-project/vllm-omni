# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental SM120 FP8 diffusion attention through FlashInfer CuTe DSL prims."""

from __future__ import annotations

import math
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_WORKSPACE_BYTES = 16 << 20
_WORKSPACES: dict[tuple[str, int | None], torch.Tensor] = {}


def _positive_optional_scale(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return scale


def _get_sm120_workspace(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACES[key] = workspace
    return workspace


def _ragged_wrapper_cls():
    try:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        from flashinfer.attention.cute_dsl.sm120_fmha import (  # noqa: F401
            SM120PrimsBatchPrefillBackend,
        )
    except Exception as exc:
        raise ImportError(
            "FLASHINFER_SM120_ATTN requires the FlashInfer SM120 CuTe DSL "
            "prims implementation from commit 4a2345906256 and its "
            "cutlass.experimental dependency"
        ) from exc
    return BatchPrefillWithRaggedKVCacheWrapper


class FlashInferSM120AttentionBackend(AttentionBackend):
    """FP8 E4M3 MHA/GQA prefill on SM120 with BF16/FP16 output."""

    accept_output_buffer: bool = True
    supports_prefix_kv_slicing: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # Explicit masks use the correctness-preserving SDPA fallback.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_SM120_ATTN"

    @staticmethod
    def get_impl_cls() -> type[FlashInferSM120AttentionImpl]:
        return FlashInferSM120AttentionImpl


class FlashInferSM120AttentionImpl(AttentionImpl):
    """Quantize diffusion Q/K/V and invoke FlashInfer ragged prefill.

    Static scales avoid device-to-host synchronization and are recommended for
    measured runs. When a scale is omitted, the first invocation calibrates it
    once with 2x headroom and caches it for the lifetime of the layer.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        del extra_impl_args
        quant = (backend_kwargs or {}).get("quant") or {}
        if quant.get("dtype_qk") != "fp8_e4m3":
            raise ValueError("FLASHINFER_SM120_ATTN requires quant.dtype_qk='fp8_e4m3'")
        if quant.get("flashinfer_backend") != "cute-dsl-prims":
            raise ValueError("FLASHINFER_SM120_ATTN requires quant.flashinfer_backend='cute-dsl-prims'")
        if head_size not in (32, 64, 128, 256):
            raise ValueError(f"unsupported SM120 FP8 head_size={head_size}")
        resolved_kv_heads = num_kv_heads or num_heads
        if resolved_kv_heads <= 0 or num_heads % resolved_kv_heads:
            raise ValueError(f"num_heads must be divisible by num_kv_heads, got {num_heads}/{resolved_kv_heads}")

        self.num_heads = num_heads
        self.num_kv_heads = resolved_kv_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.prefix = prefix or "<unnamed>"
        self._configured_scales = tuple(
            _positive_optional_scale(quant.get(name), f"quant.{name}") for name in ("q_scale", "k_scale", "v_scale")
        )
        self._resolved_scales: tuple[float, float, float] | None = None
        self._wrapper = None
        self._plan_key: tuple[Any, ...] | None = None
        self._uniform_indptr_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def _sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        return SDPAImpl(
            num_heads=query.shape[2],
            num_kv_heads=key.shape[2],
            head_size=query.shape[3],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        ).forward_cuda(query, key, value, attn_metadata)

    def _uniform_indptr(self, batch: int, length: int, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index, batch, length)
        indptr = self._uniform_indptr_cache.get(key)
        if indptr is None:
            indptr = torch.arange(batch + 1, dtype=torch.int32, device=device).mul_(length)
            self._uniform_indptr_cache[key] = indptr
        return indptr

    def _resolve_scales(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[float, float, float]:
        if self._resolved_scales is not None:
            return self._resolved_scales

        if all(scale is not None for scale in self._configured_scales):
            self._resolved_scales = self._configured_scales  # type: ignore[assignment]
            return self._resolved_scales

        # One synchronization per layer, on its first invocation only. Reserving
        # half the E4M3 range tolerates approximately 2x activation drift.
        maxima = torch.stack(tuple(t.detach().abs().amax().float() for t in (q, k, v))).cpu()
        dynamic = tuple(
            max(float(value) / (_FP8_MAX / 2.0), torch.finfo(torch.float32).tiny) for value in maxima.tolist()
        )
        self._resolved_scales = tuple(
            configured if configured is not None else calibrated
            for configured, calibrated in zip(self._configured_scales, dynamic, strict=True)
        )
        logger.info(
            "Calibrated FLASHINFER_SM120_ATTN scales for %s: q=%g k=%g v=%g",
            self.prefix,
            *self._resolved_scales,
        )
        return self._resolved_scales

    @staticmethod
    def _quantize(value: torch.Tensor, scale: float) -> torch.Tensor:
        scaled = value if scale == 1.0 else value / scale
        return scaled.to(_FP8_DTYPE).contiguous()

    def _packed_indptrs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extra = attn_metadata.extra if attn_metadata is not None else {}
        qo_indptr = extra.get("cu_seqlens_q")
        kv_indptr = extra.get("cu_seqlens_k")
        if (qo_indptr is None) != (kv_indptr is None):
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
        if qo_indptr is None:
            qo_indptr = self._uniform_indptr(query.shape[0], query.shape[1], query.device)
            kv_indptr = self._uniform_indptr(key.shape[0], key.shape[1], key.device)
        else:
            qo_indptr = qo_indptr.to(device=query.device, dtype=torch.int32).contiguous()
            kv_indptr = kv_indptr.to(device=query.device, dtype=torch.int32).contiguous()
        if qo_indptr.numel() != kv_indptr.numel():
            raise ValueError("Q and K ragged batches must contain the same number of sequences")
        return qo_indptr, kv_indptr

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            logger.debug("SM120 prims does not accept custom masks; using SDPA")
            return self._sdpa_fallback(query, key, value, attn_metadata)
        capability = torch.cuda.get_device_capability(query.device)
        if capability != (12, 0):
            raise RuntimeError(
                f"FLASHINFER_SM120_ATTN requires compute capability 12.0, got {capability[0]}.{capability[1]}"
            )
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"FLASHINFER_SM120_ATTN expects FP16/BF16 activations, got {query.dtype}")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise TypeError("Q/K/V must have the same input dtype")

        extra = attn_metadata.extra if attn_metadata is not None else {}
        valid_kv_length = extra.get("valid_kv_length")
        has_ragged_metadata = extra.get("cu_seqlens_k") is not None
        if valid_kv_length is not None and not has_ragged_metadata:
            valid_kv_length = int(valid_kv_length)
            if not 0 < valid_kv_length <= key.shape[1]:
                raise ValueError(f"valid_kv_length must be in [1, {key.shape[1]}], got {valid_kv_length}")
            key = key[:, :valid_kv_length]
            value = value[:, :valid_kv_length]

        qo_indptr, kv_indptr = self._packed_indptrs(query, key, attn_metadata)
        q = query.reshape(-1, query.shape[2], query.shape[3]).contiguous()
        k = key.reshape(-1, key.shape[2], key.shape[3]).contiguous()
        v = value.reshape(-1, value.shape[2], value.shape[3]).contiguous()

        if self._wrapper is None:
            wrapper_cls = _ragged_wrapper_cls()
            self._wrapper = wrapper_cls(
                _get_sm120_workspace(query.device),
                "NHD",
                backend="cute-dsl-prims",
            )

        plan_key = (
            qo_indptr.data_ptr(),
            getattr(qo_indptr, "_version", 0),
            kv_indptr.data_ptr(),
            getattr(kv_indptr, "_version", 0),
            tuple(q.shape),
            tuple(k.shape),
            query.dtype,
            self.causal,
        )
        if plan_key != self._plan_key:
            self._wrapper.plan(
                qo_indptr,
                kv_indptr,
                q.shape[1],
                k.shape[1],
                q.shape[2],
                causal=self.causal,
                q_data_type=_FP8_DTYPE,
                kv_data_type=_FP8_DTYPE,
                o_data_type=query.dtype,
            )
            self._plan_key = plan_key

        q_scale, k_scale, v_scale = self._resolve_scales(q, k, v)
        output = torch.empty_like(q, dtype=query.dtype)
        self._wrapper.run(
            self._quantize(q, q_scale),
            self._quantize(k, k_scale),
            self._quantize(v, v_scale),
            out=output,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return output.reshape_as(query)
