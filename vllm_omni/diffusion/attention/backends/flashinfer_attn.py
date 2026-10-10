# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from packaging.version import InvalidVersion, Version
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.capabilities import (
    CapabilityResult,
    CompilationMode,
    ExecutionContext,
    ExecutionPathResult,
    MaskMode,
    PackingMode,
    ParallelStrategy,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

logger = init_logger(__name__)

_PACKED_KEYS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")

try:
    import flashinfer
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    try:
        from flashinfer.prefill import single_prefill_with_kv_cache
    except ImportError:
        single_prefill_with_kv_cache = None

    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError:
        trtllm_ragged_attention_deepseek = None

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    single_prefill_with_kv_cache = None
    trtllm_ragged_attention_deepseek = None
    logger.warning(
        "FlashInfer is unavailable; FLASHINFER_ATTN backend will not work. Reason: %s",
        e,
    )


if not hasattr(torch.ops.vllm_omni, "flashinfer_cute_dsl_attention"):
    # The current cute-dsl route accepts workspace/seq_lens to share the public
    # FlashInfer signature, but reads its indptr tensors and mutates no inputs.
    @torch.library.custom_op(
        "vllm_omni::flashinfer_cute_dsl_attention",
        mutates_args=(),
    )
    def _flashinfer_cute_dsl_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        workspace_buffer: torch.Tensor,
        seq_lens: torch.Tensor,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        softmax_scale: float,
        batch_size: int,
    ) -> torch.Tensor:
        kernel = trtllm_ragged_attention_deepseek
        if kernel is None:
            raise RuntimeError("FlashInfer cute-dsl kernel is unavailable")
        output = kernel(
            query=query,
            key=key,
            value=value,
            workspace_buffer=workspace_buffer,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=batch_size,
            window_left=-1,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            enable_pdl=False,
            is_causal=False,
            return_lse=False,
            backend="cute-dsl",
        )
        return output.contiguous()

    @_flashinfer_cute_dsl_attention_op.register_fake
    def _flashinfer_cute_dsl_attention_fake(
        query,
        key,
        value,
        workspace_buffer,
        seq_lens,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        softmax_scale,
        batch_size,
    ):
        return query.new_empty((*query.shape[:-1], value.shape[-1]))


_flashinfer_cute_dsl_attention_op = torch.ops.vllm_omni.flashinfer_cute_dsl_attention


if not hasattr(torch.ops.vllm_omni, "flashinfer_fa2_attention"):

    @torch.library.custom_op(
        "vllm_omni::flashinfer_fa2_attention",
        mutates_args=(),
    )
    def _flashinfer_fa2_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        custom_mask: torch.Tensor | None,
        softmax_scale: float,
    ) -> torch.Tensor:
        kernel = single_prefill_with_kv_cache
        if kernel is None:
            raise RuntimeError("FlashInfer single-prefill kernel is unavailable")

        outputs = []
        for batch_idx in range(query.shape[0]):
            mask = None
            if custom_mask is not None:
                mask = custom_mask if custom_mask.ndim == 2 else custom_mask[batch_idx]
            output = kernel(
                query[batch_idx].contiguous(),
                key[batch_idx].contiguous(),
                value[batch_idx].contiguous(),
                custom_mask=mask,
                causal=False,
                sm_scale=softmax_scale,
                backend="fa2",
            )
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output)
        return torch.stack(outputs, dim=0).contiguous()

    @_flashinfer_fa2_attention_op.register_fake
    def _flashinfer_fa2_attention_fake(
        query,
        key,
        value,
        custom_mask,
        softmax_scale,
    ):
        return query.new_empty((*query.shape[:-1], value.shape[-1]))


_flashinfer_fa2_attention_op = torch.ops.vllm_omni.flashinfer_fa2_attention


def _resolve_mask_mode(attn_metadata: AttentionMetadata | None) -> MaskMode:
    if attn_metadata is None or attn_metadata.attn_mask is None:
        return MaskMode.NONE

    published_mask_mode = attn_metadata.extra.get("attention_mask_mode")
    if published_mask_mode is None:
        return MaskMode.UNKNOWN
    try:
        mask_mode = MaskMode(published_mask_mode)
    except ValueError as error:
        raise ValueError(f"Unknown attention_mask_mode {published_mask_mode!r}") from error
    if mask_mode is MaskMode.UNKNOWN:
        raise ValueError("attention_mask_mode='unknown' is reserved for unpublished semantics")
    return mask_mode


def _resolve_packing_mode(attn_metadata: AttentionMetadata | None) -> PackingMode:
    if attn_metadata is None:
        return PackingMode.NONE
    present_packed_keys = [key for key in _PACKED_KEYS if key in attn_metadata.extra]
    if not present_packed_keys:
        return PackingMode.NONE
    if len(present_packed_keys) != len(_PACKED_KEYS):
        missing = sorted(set(_PACKED_KEYS) - set(present_packed_keys))
        raise ValueError(f"Incomplete packed FlashInfer metadata; missing {missing}")
    cu_seqlens_q = attn_metadata.extra["cu_seqlens_q"]
    return PackingMode.MULTI_DOCUMENT if cu_seqlens_q.shape[0] > 3 else PackingMode.PACKED_PADDING


def _is_cuda_execution_path(*tensors: torch.Tensor) -> bool:
    return all(tensor.device.type == "cuda" for tensor in tensors)


def _metadata_is_plain_dense(attn_metadata: AttentionMetadata | None) -> bool:
    return attn_metadata is None or (
        attn_metadata.attn_mask is None
        and attn_metadata.joint_attn_mask is None
        and attn_metadata.joint_query is None
        and attn_metadata.joint_key is None
        and attn_metadata.joint_value is None
        and not attn_metadata.extra
        and attn_metadata.full_attn_spans is None
        and attn_metadata.query_ranges is None
        and attn_metadata.video_layout is None
        and attn_metadata.packed_padding is None
    )


def _runtime_context_allows_custom_op(
    attn_metadata: AttentionMetadata | None,
    execution_context: ExecutionContext | None,
) -> bool:
    runtime_context = execution_context
    if runtime_context is None and attn_metadata is not None:
        runtime_context = getattr(attn_metadata, "runtime_context", None)
    if runtime_context is not None and (
        runtime_context.parallel_strategy is not ParallelStrategy.NONE
        or runtime_context.outer_boundaries
        or runtime_context.paged_kv
        or runtime_context.piecewise
        or runtime_context.kv_cache_dtype is not None
    ):
        return False

    # The layer's active strategy is derived from ForwardContext. Check it
    # again at the runtime call boundary so a stale or absent capability
    # result cannot select the opaque path in an SP/HSDP region.
    try:
        if not is_forward_context_available():
            return True
        forward_context = get_forward_context()
        config = forward_context.omni_diffusion_config
        if config is None:
            return True
        if forward_context.sp_active:
            return False
        parallel_config = getattr(config, "parallel_config", None)
        return not bool(getattr(parallel_config, "use_hsdp", False))
    except (AssertionError, ValueError):
        # Standalone backend calls and capability unit tests may not install a
        # ForwardContext. The tensor/context checks above remain authoritative.
        return True


def _flashinfer_execution_path(context: ExecutionContext) -> str:
    if context.mask_mode is MaskMode.UNKNOWN:
        return "runtime_mask_dependent"
    if (
        context.packing_mode is not PackingMode.NONE
        or context.piecewise
        or context.paged_kv
        or context.kv_cache_dtype is not None
        or context.parallel_strategy is not ParallelStrategy.NONE
        or context.outer_boundaries
    ):
        return "unverified"
    suffix = "masked" if context.mask_mode is not MaskMode.NONE else "dense"
    return f"flashinfer_{context.kernel_variant}_{suffix}"


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

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        # Backend selection happens before FlashInfer initializes its wrapper,
        # so no concrete backend path is verified at this stage.
        return ExecutionPathResult.unmigrated(
            cls.get_name(),
            replace(context, kernel_variant=None),
            path="unverified",
        )


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
        self.dtype_qk = self._check_dtype(quant.get("dtype_qk"), "dtype_qk", self._QK_DTYPES)
        self.dtype_vo = self._check_dtype(quant.get("dtype_vo"), "dtype_vo", self._VO_DTYPES)
        requested_backend = quant.get("flashinfer_backend", "auto")
        # Set when AttentionConfig (or --diffusion-attention-backend) selected
        # FLASHINFER_ATTN. Cute-dsl custom-mask SDPA is automatic-only.
        self.backend_explicit = bool(extra_impl_args.get("backend_explicit", False))
        self._sdpa_fallback: SDPAImpl | None = None

        if not HAS_FLASHINFER:
            raise ImportError("FLASHINFER_ATTN backend requires flashinfer")

        self._check_flashinfer_version()

        self.flashinfer_backend = self._select_backend(requested_backend, device=self.device)
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
        self._qo_indptr: torch.Tensor | None = None
        self._kv_indptr: torch.Tensor | None = None
        self._plan_key: FlashInferAttentionImpl._WrapperPlanKey | None = None

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

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        extra = attn_metadata.extra if attn_metadata is not None else {}
        resolved_context = replace(
            context,
            kernel_variant=self.flashinfer_backend,
            dtype=str(query.dtype).removeprefix("torch."),
            causal=self.causal,
            mask_mode=_resolve_mask_mode(attn_metadata),
            packing_mode=_resolve_packing_mode(attn_metadata),
            piecewise=(attn_metadata is not None and attn_metadata.full_attn_spans is not None) or context.piecewise,
            kv_cache_dtype=extra.get("kv_cache_dtype", context.kv_cache_dtype),
        )
        # Do not select or widen hardware variants here. The initialized
        # FlashInfer backend already owns that selection; this contract only
        # describes the concrete dense paths made opaque below.
        result = ExecutionPathResult.unmigrated(
            "FLASHINFER_ATTN",
            resolved_context,
            path=_flashinfer_execution_path(resolved_context),
        )
        if (
            resolved_context.platform == "cuda"
            and result.path in {"flashinfer_fa2_dense", "flashinfer_cute-dsl_dense"}
            and (
                self._is_fa2_custom_op_candidate(query, key, value, attn_metadata, execution_context=resolved_context)
                or self._is_cute_dsl_custom_op_candidate(
                    query, key, value, attn_metadata, execution_context=resolved_context
                )
            )
        ):
            return replace(
                result,
                support=CapabilityResult.supported(),
                compilation_mode=CompilationMode.CUSTOM_OP,
            )
        return result

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

    @torch.compiler.disable
    @staticmethod
    def _extract_scalar(scalar_tensor: torch.Tensor):
        return scalar_tensor.item()

    @staticmethod
    def _per_tensor_quantize(
        tensor: torch.Tensor,
        to_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, float]:
        if torch.finfo(to_dtype).bits == 8:
            scale_tensor = tensor.abs().amax().float().clamp_min(1e-6) / torch.finfo(to_dtype).max
            tensor = (tensor.float() * torch.reciprocal(scale_tensor)).to(to_dtype)
            return tensor, FlashInferAttentionImpl._extract_scalar(scale_tensor)
        return tensor.to(to_dtype), 1.0

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

    def _is_cute_dsl_custom_op_candidate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        execution_context: ExecutionContext | None = None,
    ) -> bool:
        if not _metadata_is_plain_dense(attn_metadata) or not _runtime_context_allows_custom_op(
            attn_metadata, execution_context
        ):
            return False
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            return False
        return (
            self.flashinfer_backend == "cute-dsl"
            and trtllm_ragged_attention_deepseek is not None
            and not self.causal
            and self.dtype_qk in (None, torch.bfloat16)
            and self.dtype_vo in (None, torch.bfloat16)
            and query.dtype == key.dtype == value.dtype == torch.bfloat16
            and _is_cuda_execution_path(query, key, value)
            and query.device == key.device == value.device
            and query.shape[0] == key.shape[0] == value.shape[0]
            and key.shape[1] == value.shape[1]
            and query.shape[2] == key.shape[2] == value.shape[2]
            and query.shape[1] <= key.shape[1]
            and query.shape[0] > 0
            and query.shape[1] > 0
            and key.shape[1] > 0
            and query.shape[2] > 0
            and query.shape[3] == key.shape[3] == value.shape[3] == 128
        )

    def _is_fa2_custom_op_candidate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        execution_context: ExecutionContext | None = None,
    ) -> bool:
        if not _metadata_is_plain_dense(attn_metadata) or not _runtime_context_allows_custom_op(
            attn_metadata, execution_context
        ):
            return False
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            return False
        return (
            self.flashinfer_backend == "fa2"
            and single_prefill_with_kv_cache is not None
            and not self.causal
            and self.dtype_qk in (None, torch.bfloat16)
            and self.dtype_vo in (None, torch.bfloat16)
            and query.dtype == key.dtype == value.dtype == torch.bfloat16
            and _is_cuda_execution_path(query, key, value)
            and query.device == key.device == value.device
            and query.shape[0] == key.shape[0] == value.shape[0]
            and key.shape[1] == value.shape[1]
            and query.shape[2] == key.shape[2] == value.shape[2]
            and query.shape[1] <= key.shape[1]
            and query.shape[0] > 0
            and query.shape[1] > 0
            and key.shape[1] > 0
            and query.shape[2] > 0
            and query.shape[3] == key.shape[3] == value.shape[3] == 128
        )

    def _run_fa2_custom_op(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return _flashinfer_fa2_attention_op(
            query,
            key,
            value,
            None,
            self.softmax_scale,
        )

    def _run_cute_dsl_custom_op(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, qo_len, num_heads, head_dim = query.shape
        kv_len = key.shape[1]
        query = query.reshape(batch_size * qo_len, num_heads, head_dim).contiguous()
        key = key.reshape(batch_size * kv_len, num_heads, head_dim).contiguous()
        value = value.reshape(batch_size * kv_len, num_heads, head_dim).contiguous()
        qo_indptr = (
            torch.arange(
                batch_size + 1,
                device=query.device,
                dtype=torch.int32,
            )
            * qo_len
        )
        kv_indptr = (
            torch.arange(
                batch_size + 1,
                device=query.device,
                dtype=torch.int32,
            )
            * kv_len
        )
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        output = _flashinfer_cute_dsl_attention_op(
            query,
            key,
            value,
            self._workspace,
            seq_lens,
            qo_indptr,
            kv_indptr,
            qo_len,
            kv_len,
            self.softmax_scale,
            batch_size,
        )
        return output.reshape(batch_size, qo_len, num_heads, head_dim)

    def _run_batch_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        custom_mask: torch.Tensor | None,
        use_cute_dsl_custom_op: bool = False,
        use_fa2_custom_op: bool = False,
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

        if use_fa2_custom_op:
            return self._run_fa2_custom_op(query, key, value)
        if use_cute_dsl_custom_op:
            return self._run_cute_dsl_custom_op(query, key, value)

        q = query.reshape(batch_size * qo_len, num_q_heads, head_dim_qk)
        k = key.reshape(batch_size * kv_len, num_kv_heads, head_dim_k)
        v = value.reshape(batch_size * kv_len, num_kv_heads, head_dim_vo)
        q, q_scale = self._per_tensor_quantize(q, self.dtype_qk or q.dtype)
        k, k_scale = self._per_tensor_quantize(k, self.dtype_qk or k.dtype)
        v, v_scale = self._per_tensor_quantize(v, self.dtype_vo or v.dtype)

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
        out = self._run_wrapper(q, k, v, q_scale, k_scale, v_scale)
        out = out.reshape(batch_size, qo_len, num_q_heads, head_dim_vo)
        return out.to(query.dtype) if out.dtype != query.dtype else out

    @torch.compiler.disable
    def _run_wrapper(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_scale: float,
        k_scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        try:
            return self._wrapper.run(
                query,
                key,
                value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        except NotImplementedError as _:
            # Older FlashInfer does not route scales for CuTeDSL backends.
            # Apply some scales manually.
            if q_scale != 1.0 or k_scale != 1.0:
                raise NotImplementedError("FlashInfer/CuTeDSL backend doesn't support quantizing QK into FP8 yet.")
            out = self._wrapper.run(query, key, value)
            if v_scale is not None and v_scale != 1.0:
                out = out * v_scale
            return out

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

        use_fa2_custom_op = self._is_fa2_custom_op_candidate(
            query,
            key,
            value,
            attn_metadata,
        )
        use_cute_dsl_custom_op = self._is_cute_dsl_custom_op_candidate(
            query,
            key,
            value,
            attn_metadata,
        )
        return self._run_batch_prefill(
            query,
            key,
            value,
            custom_mask,
            use_cute_dsl_custom_op=use_cute_dsl_custom_op,
            use_fa2_custom_op=use_fa2_custom_op,
        )
