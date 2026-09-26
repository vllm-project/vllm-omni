# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from packaging.version import InvalidVersion, Version
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    PackedPaddingMetadata,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

logger = init_logger(__name__)

try:
    import flashinfer
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    logger.warning(
        "FlashInfer is unavailable; FLASHINFER_ATTN backend will not work. Reason: %s",
        e,
    )


class FlashInferAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls, attention_spec: object | None = None) -> bool:
        # fa2/fa3 batch-prefill accept boolean custom masks. cute-dsl does not:
        # automatic platform selection may fall back to SDPA, but an explicit
        # FLASHINFER_ATTN config must not be advertised as mask-capable.
        requested = "auto"
        quant = getattr(attention_spec, "quant", None) if attention_spec is not None else None
        requested_backend = getattr(quant, "flashinfer_backend", None)
        if requested_backend:
            requested = requested_backend
        backend = FlashInferAttentionImpl._select_backend(requested)
        if backend == "cute-dsl-prims":
            return False
        if backend == "cute-dsl":
            return attention_spec is None
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FlashInfer dense prefill is well-tested for these head_dims on
        # Ampere/Hopper/Blackwell. Covers the dominant diffusion DiT shapes
        # (SD3 = 64, Flux/HV/Wan = 128, joint-attn = 256).
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_ATTN"

    @staticmethod
    def get_impl_cls() -> type[FlashInferAttentionImpl]:
        return FlashInferAttentionImpl


class FlashInferSM120AttentionBackend(FlashInferAttentionBackend):
    """Capabilities of the explicitly selected SM120 PRIMS variant."""

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        return True

    @classmethod
    def supports_attention_mask(cls, attention_spec: object | None = None) -> bool:
        return False


class FlashInferAttentionImpl(AttentionImpl):
    _QK_DTYPES = {torch.float16, torch.bfloat16}
    _VO_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    @dataclass(frozen=True)
    class _WrapperPlanKey:
        batch_size: int
        qo_len: int
        kv_len: int
        num_q_heads: int
        num_kv_heads: int
        head_dim_qk: int
        head_dim_k: int
        head_dim_vo: int
        q_dtype: torch.dtype
        k_dtype: torch.dtype
        v_dtype: torch.dtype
        o_dtype: torch.dtype
        causal: bool
        softmax_scale: float
        has_custom_mask: bool

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
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        backend_kwargs = backend_kwargs or {}
        quant = backend_kwargs.get("quant") or {}
        requested_backend = quant.get("flashinfer_backend", "auto")
        qk_dtypes = {torch.float8_e4m3fn} if requested_backend == "cute-dsl-prims" else self._QK_DTYPES
        self.dtype_qk = self._check_dtype(quant.get("dtype_qk"), "dtype_qk", qk_dtypes)
        self.dtype_vo = self._check_dtype(quant.get("dtype_vo"), "dtype_vo", self._VO_DTYPES)
        self.skip_softmax_threshold = backend_kwargs.get("skip_softmax_threshold")
        if self.skip_softmax_threshold is not None:
            self.skip_softmax_threshold = float(self.skip_softmax_threshold)
            if not math.isfinite(self.skip_softmax_threshold) or self.skip_softmax_threshold < 0:
                raise ValueError("skip_softmax_threshold must be finite and >= 0")
            if requested_backend != "cute-dsl-prims":
                raise ValueError("FLASHINFER_ATTN skip_softmax_threshold requires cute-dsl-prims")
        if backend_kwargs.get("target_sparsity") is not None:
            raise ValueError("FLASHINFER_ATTN supports an absolute skip_softmax.threshold, not target_sparsity")
        self.disabled_until_timestep = float(backend_kwargs.get("disabled_until_timestep", 0.0))
        if not math.isfinite(self.disabled_until_timestep) or not 0 <= self.disabled_until_timestep <= 1:
            raise ValueError("disabled_until_timestep must be finite and in [0, 1]")
        if self.disabled_until_timestep and requested_backend != "cute-dsl-prims":
            raise ValueError("FLASHINFER_ATTN disabled_until_timestep requires cute-dsl-prims")
        # Set when AttentionConfig (or --diffusion-attention-backend) selected
        # FLASHINFER_ATTN. Cute-dsl custom-mask SDPA is automatic-only.
        self.backend_explicit = bool(extra_impl_args.get("backend_explicit", False))
        self._sdpa_fallback: SDPAImpl | None = None

        if not HAS_FLASHINFER:
            raise ImportError("FLASHINFER_ATTN backend requires flashinfer")

        self._check_flashinfer_version()

        self.flashinfer_backend = self._select_backend(requested_backend, device=self.device)
        self._qo_indptr: torch.Tensor | None = None
        self._kv_indptr: torch.Tensor | None = None
        self._plan_key: FlashInferAttentionImpl._WrapperPlanKey | None = None
        self._sm120_shape: tuple[int, int, int] | None = None
        if self.flashinfer_backend == "cute-dsl-prims":
            self._init_sm120(head_size, num_heads, num_heads if num_kv_heads is None else num_kv_heads)
            return

        workspace_size = 0 if self.flashinfer_backend == "cute-dsl" else 128 * 1024 * 1024
        self._workspace = torch.empty(
            workspace_size,
            device=self.device,
            dtype=torch.uint8,
        )
        self._wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
            backend=self.flashinfer_backend,
        )

        if self.dtype_qk is not None or self.dtype_vo is not None:
            logger.info_once(
                "FLASHINFER_ATTN dtype override: Q/K=%s, V=%s.",
                self.dtype_qk,
                self.dtype_vo,
            )

        logger.info_once(
            "FLASHINFER_ATTN initialized backend=%s on %s.",
            self.flashinfer_backend,
            self.device,
        )

    def _init_sm120(self, head_size: int, num_heads: int, num_kv_heads: int) -> None:
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("FLASHINFER_ATTN cute-dsl-prims requires an SM120 GPU (compute capability 12.0)")
        if self.dtype_qk != torch.float8_e4m3fn or self.dtype_vo != torch.float8_e4m3fn:
            raise ValueError("cute-dsl-prims requires dtype_qk=dtype_vo='fp8_e4m3'")
        if head_size not in (64, 128, 256) or num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("cute-dsl-prims requires head_size 64/128/256 and Q heads divisible by KV heads")
        try:
            from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill
        except ImportError as e:
            raise ImportError("cute-dsl-prims requires FlashInfer with SM120 support (PR #4859)") from e
        if "skip_softmax_threshold" not in inspect.signature(sm120_fmha_fp8_ragged_prefill).parameters:
            raise ImportError("Install FlashInfer including PR #4859's direct skip_softmax_threshold API")
        self._sm120_prefill = sm120_fmha_fp8_ragged_prefill
        logger.info_once(
            "FLASHINFER_ATTN initialized SM120 FP8 cute-dsl-prims, skip_softmax_threshold=%s.",
            self.skip_softmax_threshold,
        )

    @torch.compiler.disable
    def _run_sm120_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        skip_softmax_threshold: float | torch.Tensor | None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Run the direct PRIMS API; shared prefill wrappers cannot enable skipping."""
        if any(t.device != self.device for t in (query, key, value)):
            raise ValueError(f"cute-dsl-prims inputs must be on the initialization device {self.device}")
        if any(t.ndim != 4 for t in (query, key, value)):
            raise ValueError("cute-dsl-prims expects Q/K/V in (batch, sequence, heads, head_dim) layout")
        if any(t.dtype not in (torch.float16, torch.bfloat16) for t in (query, key, value)):
            raise ValueError("cute-dsl-prims expects FP16/BF16 inputs, converted to FP8 internally")
        batch, qo_len, num_heads, head_dim = query.shape
        kv_len, num_kv_heads = key.shape[1:3]
        if (
            key.shape != value.shape
            or key.shape[0] != batch
            or key.shape[3] != head_dim
            or head_dim not in (64, 128, 256)
            or num_heads == 0
            or num_kv_heads == 0
            or num_heads % num_kv_heads
        ):
            raise ValueError("cute-dsl-prims requires matching K/V, batch and head dimensions, and valid GQA heads")
        q_tokens, kv_tokens, cu_q, cu_k, max_q_len = self._sm120_layout(query, key, attn_metadata)
        q = query.reshape(batch * qo_len, num_heads, head_dim)[:q_tokens].to(torch.float8_e4m3fn).contiguous()
        k = key.reshape(batch * kv_len, num_kv_heads, head_dim)[:kv_tokens].to(torch.float8_e4m3fn).contiguous()
        v = value.reshape(batch * kv_len, num_kv_heads, head_dim)[:kv_tokens].to(torch.float8_e4m3fn).contiguous()
        # H3's alignment rows are excluded from the packed-padding launch.
        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        if q_tokens < batch * qo_len:
            out.zero_()
        if batch == 0 or qo_len == 0:
            return out
        self._sm120_prefill(
            q,
            k,
            v,
            out.view(batch * qo_len, num_heads, head_dim)[:q_tokens],
            cu_q,
            cu_k,
            max_seqlen_q=max_q_len,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            skip_softmax_threshold=skip_softmax_threshold,
        )
        return out

    def _sm120_layout(self, query: torch.Tensor, key: torch.Tensor, metadata: AttentionMetadata | None):
        """Use caller-owned packed offsets without a CUDA scalar read or copy."""
        batch, qo_len = query.shape[:2]
        kv_len = key.shape[1]
        q_tokens, kv_tokens = batch * qo_len, batch * kv_len
        extra = metadata.extra if metadata is not None else {}
        padding = metadata.packed_padding if metadata is not None else None
        packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q")
        has_packed = any(name in extra for name in packed_keys)
        if padding is not None and not has_packed:
            raise ValueError("cute-dsl-prims packed_padding requires cu_seqlens_q/k and max_seqlen_q")
        if not has_packed:
            shape = (batch, qo_len, kv_len)
            if shape != self._sm120_shape:
                self._qo_indptr = self._make_indptr(batch, qo_len)
                self._kv_indptr = self._make_indptr(batch, kv_len)
                self._sm120_shape = shape
            return q_tokens, kv_tokens, self._qo_indptr, self._kv_indptr, qo_len
        if batch != 1 or not all(name in extra for name in packed_keys):
            raise ValueError("cute-dsl-prims packed attention requires batch=1, cu_seqlens_q/k and max_seqlen_q")
        cu_q, cu_k, max_q_len = (extra[name] for name in packed_keys)
        if isinstance(max_q_len, bool) or not isinstance(max_q_len, int) or not 0 < max_q_len <= q_tokens:
            raise ValueError("cute-dsl-prims packed max_seqlen_q must be a positive Python int within Q length")
        if padding is not None:
            if not isinstance(padding, PackedPaddingMetadata):
                raise ValueError("packed_padding must be PackedPaddingMetadata")
            q_tokens, kv_tokens = padding.q_length, padding.kv_length
            if (
                isinstance(q_tokens, bool)
                or not isinstance(q_tokens, int)
                or isinstance(kv_tokens, bool)
                or not isinstance(kv_tokens, int)
                or not 0 < q_tokens <= qo_len
                or not 0 < kv_tokens <= kv_len
                or max_q_len != q_tokens
                or extra.get("valid_kv_length", kv_tokens) != kv_tokens
            ):
                raise ValueError("cute-dsl-prims packed-padding lengths must match valid Q/KV prefixes")
            cu_q, cu_k = padding.cu_seqlens_q, padding.cu_seqlens_k
        for cu in (cu_q, cu_k):
            if (
                not isinstance(cu, torch.Tensor)
                or cu.dtype != torch.int32
                or cu.device != query.device
                or cu.ndim != 1
                or cu.numel() < 2
                or not cu.is_contiguous()
            ):
                raise ValueError("cute-dsl-prims packed offsets must be contiguous int32 vectors on the Q device")
        if cu_q.shape != cu_k.shape:
            raise ValueError("cute-dsl-prims packed Q/K offsets must have the same number of sequences")
        if padding is not None and cu_q.shape != (2,):
            raise ValueError("cute-dsl-prims packed-padding offsets must contain exactly one sequence")
        return q_tokens, kv_tokens, cu_q, cu_k, max_q_len

    def _check_flashinfer_version(self) -> None:
        if self.dtype_qk == self.dtype_vo:
            return
        try:
            flashinfer_version = Version(flashinfer.__version__)
        except (AttributeError, InvalidVersion):
            return
        if flashinfer_version <= Version("0.6.15"):
            raise RuntimeError(
                f"FlashInfer {flashinfer_version} is too old for reliable mixed "
                f"QK/V dtype attention (Q/K={self.dtype_qk}, V={self.dtype_vo}); "
                "install flashinfer >= 0.6.16rc1."
            )

    @staticmethod
    def _select_backend(requested_backend: str, device: torch.device | None = None) -> str:
        if requested_backend != "auto":
            return requested_backend
        try:
            major, _minor = (
                torch.cuda.get_device_capability(device) if device is not None else torch.cuda.get_device_capability()
            )
        except Exception:
            return "fa2"
        if major >= 10:
            return "cute-dsl"
        if major >= 9:
            return "fa3"
        return "fa2"

    @classmethod
    def _check_dtype(
        cls,
        dtype: torch.dtype | str | None,
        option_name: str,
        allowed: set[torch.dtype],
    ) -> torch.dtype | None:
        if dtype is None:
            return None
        if isinstance(dtype, str):
            dtype = torch.float8_e4m3fn if dtype == "fp8_e4m3" else getattr(torch, dtype)
        if dtype not in allowed:
            choices = ", ".join(sorted(str(item) for item in allowed))
            raise ValueError(f"Unsupported {option_name}={dtype}; expected one of: {choices}")
        return dtype

    @staticmethod
    def _pack_mask_for_flashinfer(
        attn_mask: torch.Tensor, batch_size: int, qo_len: int, kv_len: int
    ) -> torch.Tensor | None:
        """Convert a diffusion-style attn_mask into the boolean form
        FlashInfer's ``custom_mask`` expects (``True`` = keep).

        Returns either ``(qo_len, kv_len)`` (shared across the batch) or
        ``(batch_size, qo_len, kv_len)`` (per-sample), or ``None`` when the
        mask is all-keep (elide). Only boolean masks are handled here;
        additive/float masks and shape mismatches raise ``ValueError`` because
        an explicitly selected backend must not silently switch to SDPA.
        """
        mask = attn_mask
        if mask.dtype != torch.bool:
            # Additive masks (0 / -inf / -1e4 / finfo.min) cannot be faithfully
            # reduced to a boolean keep-mask here; SDPA handles them correctly.
            raise ValueError(f"non-boolean attn_mask (dtype={mask.dtype}); FlashInfer custom_mask is boolean-only")
        # Diffusion masks arrive as (qo,kv), (1,1,kv), (B,1,1,kv), (B,1,qo,kv)
        # or (B,H,qo,kv). The mask is identical across heads, so collapse the
        # head dim, but keep a real batch dim — indexing mask[0] would reuse
        # sample 0's padding for every sample (wrong under CFG / mixed lengths).
        if mask.dim() == 4:
            mask = mask[:, 0]  # (B, qo|1, kv)
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]  # (qo|1, kv) — shared across the batch
        try:
            if mask.dim() >= 3:
                mask = mask.broadcast_to((batch_size, qo_len, kv_len))
            else:
                mask = mask.broadcast_to((qo_len, kv_len))
        except RuntimeError as e:
            raise ValueError(
                f"attn_mask shape {tuple(attn_mask.shape)} cannot broadcast to "
                f"(batch={batch_size}, qo_len={qo_len}, kv_len={kv_len})"
            ) from e
        if mask.all():
            return None
        # ``broadcast_to`` returns a non-contiguous view; materialize for the
        # kernel, which reads from GPU memory directly.
        return mask.contiguous()

    @torch.compiler.disable
    def _plan_wrapper(
        self,
        key: _WrapperPlanKey,
        flat_mask: torch.Tensor | None,
    ) -> None:
        self._wrapper.plan(
            self._qo_indptr,
            self._kv_indptr,
            key.num_q_heads,
            key.num_kv_heads,
            key.head_dim_qk,
            head_dim_vo=key.head_dim_vo,
            custom_mask=flat_mask,
            causal=key.causal,
            sm_scale=key.softmax_scale,
            q_data_type=key.q_dtype,
            # FlashInfer versions newer than 0.6.15 can dispatch K and V with
            # independent runtime dtypes even though plan() names this K/V.
            kv_data_type=key.k_dtype,
            o_data_type=key.o_dtype,
        )

    @torch.compiler.disable
    def _make_indptr(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        return (
            torch.arange(
                batch_size + 1,
                device=self.device,
                dtype=torch.int32,
            )
            * sequence_length
        )

    def _ensure_plan(
        self,
        key: _WrapperPlanKey,
        flat_mask: torch.Tensor | None,
    ) -> None:
        key_changed = key != self._plan_key
        if not key_changed and flat_mask is None:
            return

        if key_changed:
            self._qo_indptr = self._make_indptr(key.batch_size, key.qo_len)
            self._kv_indptr = self._make_indptr(key.batch_size, key.kv_len)

        # A custom mask's contents may change without its shape changing, so
        # copy/re-plan it for every masked invocation.
        self._plan_wrapper(key, flat_mask)
        self._plan_key = key

    def _run_batch_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        custom_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if query.device != self.device or key.device != self.device or value.device != self.device:
            raise ValueError(
                "FLASHINFER_ATTN inputs must remain on the layer initialization "
                f"device {self.device}; got Q={query.device}, K={key.device}, "
                f"V={value.device}"
            )

        batch_size, qo_len, num_q_heads, head_dim_qk = query.shape
        kv_len = key.shape[1]
        num_kv_heads = key.shape[2]
        head_dim_k = key.shape[3]
        head_dim_vo = value.shape[3]

        q = query.reshape(batch_size * qo_len, num_q_heads, head_dim_qk)
        k = key.reshape(batch_size * kv_len, num_kv_heads, head_dim_k)
        v = value.reshape(batch_size * kv_len, num_kv_heads, head_dim_vo)
        if self.dtype_qk is not None:
            q = q.to(self.dtype_qk)
            k = k.to(self.dtype_qk)
        if self.dtype_vo is not None:
            v = v.to(self.dtype_vo)

        flat_mask = None
        if custom_mask is not None:
            if custom_mask.dim() == 2:
                custom_mask = custom_mask.unsqueeze(0).expand(batch_size, -1, -1)
            flat_mask = custom_mask.contiguous().view(-1)

        self._ensure_plan(
            FlashInferAttentionImpl._WrapperPlanKey(
                batch_size=batch_size,
                qo_len=qo_len,
                kv_len=kv_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_k=head_dim_k,
                head_dim_vo=head_dim_vo,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                o_dtype=query.dtype,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
                has_custom_mask=flat_mask is not None,
            ),
            flat_mask,
        )
        out = self._run_wrapper(q, k, v)
        out = out.reshape(batch_size, qo_len, num_q_heads, head_dim_vo)
        return out.to(query.dtype) if out.dtype != query.dtype else out

    @torch.compiler.disable
    def _run_wrapper(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self._wrapper.run(query, key, value)

    def _sdpa_for_unsupported_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        reason: str,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        logger.warning_once("FLASHINFER_ATTN falling back to SDPA: %s", reason)
        fallback = self._sdpa_fallback
        if fallback is None:
            fallback = SDPAImpl(
                num_heads=query.shape[2],
                head_size=query.shape[3],
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )
            self._sdpa_fallback = fallback
        return fallback.forward_cuda(query, key, value, attn_metadata)

    def _resolve_sm120_threshold(self, attn_metadata: AttentionMetadata | None) -> float | torch.Tensor | None:
        extra = attn_metadata.extra if attn_metadata is not None else {}
        threshold = extra.get("skip_softmax_threshold", self.skip_softmax_threshold)
        if threshold is None or not self.disabled_until_timestep:
            return threshold

        # Python denoise progress is not re-evaluated during graph replay.
        # Ungated caller-owned threshold tensors remain graph-compatible.
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("SM120 disabled_until_timestep requires eager execution; set enforce_eager=True")
        from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

        timestep = get_forward_context().denoise_timestep if is_forward_context_available() else None
        if timestep is None or not math.isfinite(timestep):
            logger.warning_once(
                "FLASHINFER_ATTN skip: disabled_until_timestep=%s requires a finite denoise_timestep; staying dense.",
                self.disabled_until_timestep,
            )
            return None
        return None if timestep > self.disabled_until_timestep else threshold

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if not HAS_FLASHINFER:
            raise ImportError(
                "FLASHINFER_ATTN backend requires flashinfer. "
                "Install it or set DIFFUSION_ATTENTION_BACKEND to another backend."
            )

        if self.flashinfer_backend == "cute-dsl-prims":
            if attn_metadata is not None and attn_metadata.attn_mask is not None:
                raise ValueError("FLASHINFER_ATTN cute-dsl-prims does not support custom masks")
            # Caller-owned CUDA float32 [batch] tensors keep their address and
            # lifetime across graph replays. Forward them without copying or
            # reading their values on the host; FlashInfer validates the layout.
            threshold = self._resolve_sm120_threshold(attn_metadata)
            return self._run_sm120_prefill(query, key, value, threshold, attn_metadata)

        # Explicit FLASHINFER_ATTN must not silently switch away from the
        # requested kernel. Automatic platform selection (Blackwell cute-dsl)
        # may use SDPA for masks that this FlashInfer variant cannot run.
        batch_size = query.shape[0]
        custom_mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            try:
                custom_mask = self._pack_mask_for_flashinfer(
                    attn_metadata.attn_mask,
                    batch_size=batch_size,
                    qo_len=query.shape[1],
                    kv_len=key.shape[1],
                )
            except ValueError:
                if self.backend_explicit:
                    raise
                return self._sdpa_for_unsupported_mask(
                    query,
                    key,
                    value,
                    attn_metadata,
                    "attn_mask is not a FlashInfer boolean custom_mask",
                )
            # FlashInfer cannot combine causal masking with a custom_mask; rather
            # than silently dropping the explicit mask (diverging from SDPA),
            # require an explicit caller to select a compatible backend.
            if custom_mask is not None and self.causal:
                if self.backend_explicit:
                    raise ValueError(
                        "FLASHINFER_ATTN does not support causal=True together with an explicit custom mask."
                    )
                return self._sdpa_for_unsupported_mask(
                    query,
                    key,
                    value,
                    attn_metadata,
                    "causal=True with a custom mask",
                )

        if custom_mask is not None and self.flashinfer_backend == "cute-dsl":
            if self.backend_explicit:
                raise ValueError(
                    "FLASHINFER_ATTN cute-dsl backend does not support custom masks. "
                    "Select TORCH_SDPA or a FlashInfer backend that accepts custom_mask."
                )
            return self._sdpa_for_unsupported_mask(
                query,
                key,
                value,
                attn_metadata,
                "cute-dsl does not support custom masks",
            )

        return self._run_batch_prefill(query, key, value, custom_mask)
