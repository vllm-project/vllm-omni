# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from typing import Any

import torch
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
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_MISSING_AUTO_ROUND = (
    "SPARGE_ATTN requires auto-round-lib with the ARK XPU sparse attention kernel. "
    "Install it with `pip install auto-round-lib`."
)
_MISSING_SPAS_CUDA = (
    "SPARGE_ATTN on CUDA requires the SpargeAttn CUDA extension (spas_sage_attn). "
    "Build it from source with "
    "`pip install git+https://github.com/thu-ml/SpargeAttn.git` and apply the SM90 "
    "deadlock fix from thu-ml/SpargeAttn#127 (`__syncthreads()` before the V load in "
    "csrc/qattn/qk_int_sv_f8_cuda_sm90.cuh) until that fix lands upstream: builds "
    "without it can hang on H100/H200."
)
# Hopper needs the natively compiled SM90 sparse kernel; an importable SM80/89-only
# build passes a plain import check and only fails at the first sparse call.
_SM90_SPARSE_ENTRY_POINT = "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90"
_MISSING_SM90_KERNEL = (
    "SPARGE_ATTN: the installed spas_sage_attn build has no SM90 sparse attention "
    f"entry point ({_SM90_SPARSE_ENTRY_POINT}); it was most likely built without "
    "compute capability 9.0. Rebuild SpargeAttn with "
    "`TORCH_CUDA_ARCH_LIST=9.0 CUDA_HOME=<cuda-12.8+> pip install --no-build-isolation .`."
)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_HEAD_SIZES = (64, 128)
# SpargeAttn's CUDA kernels tile the query sequence and assert Q has at least
# 128 rows; K/V lengths are free (the kernel pads its own V staging buffer).
_CUDA_MIN_SEQ_LEN = 128


def _load_spas_cuda_kernel():
    try:
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    except ImportError:
        return None
    if not callable(spas_sage2_attn_meansim_topk_cuda):
        return None
    return spas_sage2_attn_meansim_topk_cuda


def _cuda_device_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(torch.accelerator.current_device_index())
    except (RuntimeError, AssertionError):
        return None


def _validate_cuda_arch_support() -> None:
    """Reject a build that cannot serve the current GPU before the first call."""
    if _cuda_device_capability() != (9, 0):
        return
    try:
        from spas_sage_attn import _qattn
    except ImportError as exc:
        raise ImportError(_MISSING_SPAS_CUDA) from exc
    if not hasattr(_qattn, _SM90_SPARSE_ENTRY_POINT):
        raise ImportError(_MISSING_SM90_KERNEL)


if not hasattr(torch.ops.vllm_omni, "sparge_attention"):

    @torch.library.custom_op("vllm_omni::sparge_attention", mutates_args=())
    def _sparge_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scale: float,
        topk: float,
        q_tile_override: int,
        sparse_q_block_tokens: int,
        sparse_k_block_tokens: int,
    ) -> torch.Tensor:
        import auto_round_kernel as ark

        output = ark.sparge_sage2_attn_meansim_topk_xpu(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=topk,
            attention_sink=False,
            tensor_layout="NHD",
            q_tile_override=q_tile_override,
            sparse_q_block_tokens=sparse_q_block_tokens,
            sparse_k_block_tokens=sparse_k_block_tokens,
        )
        return output.contiguous()

    @_sparge_attention_op.register_fake
    def _sparge_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scale: float,
        topk: float,
        q_tile_override: int,
        sparse_q_block_tokens: int,
        sparse_k_block_tokens: int,
    ) -> torch.Tensor:
        del key, is_causal, scale, topk, q_tile_override, sparse_q_block_tokens, sparse_k_block_tokens
        return query.new_empty(query.shape, dtype=value.dtype)


_sparge_attention_op = torch.ops.vllm_omni.sparge_attention


if not hasattr(torch.ops.vllm_omni, "sparge_attention_cuda"):

    @torch.library.custom_op("vllm_omni::sparge_attention_cuda", mutates_args=())
    def _sparge_attention_cuda_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scale: float,
        topk: float,
    ) -> torch.Tensor:
        kernel = _load_spas_cuda_kernel()
        if kernel is None:
            raise ImportError(_MISSING_SPAS_CUDA)
        output = kernel(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale,
            smooth_k=True,
            topk=topk,
            tensor_layout="NHD",
        )
        return output.contiguous()

    @_sparge_attention_cuda_op.register_fake
    def _sparge_attention_cuda_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scale: float,
        topk: float,
    ) -> torch.Tensor:
        del key, is_causal, scale, topk
        return query.new_empty(query.shape, dtype=value.dtype)


_sparge_attention_cuda_op = torch.ops.vllm_omni.sparge_attention_cuda


def _sparge_metadata_error(attn_metadata: AttentionMetadata | None, *, allow_packed_prefix: bool = False) -> str | None:
    if attn_metadata is None:
        return None
    if attn_metadata.attn_mask is not None:
        return "attn_mask is not supported"
    if attn_metadata.joint_attn_mask is not None:
        return "joint_attn_mask is not supported"
    if attn_metadata.cross_query_mask is not None:
        return "cross-query masks are not supported"
    if attn_metadata.packed_padding is not None and not allow_packed_prefix:
        return "packed padding metadata is not supported"
    if attn_metadata.full_attn_spans is not None or attn_metadata.query_ranges is not None:
        return "piecewise attention metadata is not supported"
    extra = attn_metadata.extra
    packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
    if not allow_packed_prefix and any(name in extra for name in packed_keys):
        return "packed variable-length attention is not supported"
    if extra.get("kv_cache_dtype") not in (None, "auto", "float"):
        return "quantized KV cache is not supported"
    return None


def _validate_sparge_metadata(attn_metadata: AttentionMetadata | None, *, allow_packed_prefix: bool = False) -> None:
    reason = _sparge_metadata_error(attn_metadata, allow_packed_prefix=allow_packed_prefix)
    if reason is not None:
        raise ValueError(f"SPARGE_ATTN: {reason}. Select a compatible backend.")


_PACKED_EXTRA_KEYS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")


def _has_raw_packed_metadata(attn_metadata: AttentionMetadata | None) -> bool:
    if attn_metadata is None or attn_metadata.packed_padding is not None:
        return False
    extra = attn_metadata.extra or {}
    return any(name in extra for name in _PACKED_EXTRA_KEYS)


def _cuda_packed_lengths(attn_metadata: AttentionMetadata | None, q_length: int, kv_length: int) -> tuple[int, int]:
    """Return the real Q/KV bounds of a packed [real, pad] call.

    The CUDA SpargeAttn kernels have no varlen or mask argument, so the padding
    tail is removed by slicing. Only the producer-validated
    ``PackedPaddingMetadata`` view is accepted: raw ``cu_seqlens``/
    ``max_seqlen_*`` describe a multi-document packing whose real rows are not a
    prefix, so slicing them as one would drop documents or keep padding.
    """
    if attn_metadata is None:
        return q_length, kv_length
    packed = attn_metadata.packed_padding
    if packed is None:
        if _has_raw_packed_metadata(attn_metadata):
            raise NotImplementedError(
                "SPARGE_ATTN on CUDA only supports packed calls that publish "
                "AttentionMetadata.packed_padding; raw cu_seqlens/max_seqlen_* "
                "describe a multi-document packing whose real rows are not a prefix. "
                "Select a backend that isolates packed documents."
            )
        return q_length, kv_length
    return min(int(packed.q_length), q_length), min(int(packed.kv_length), kv_length)


def _validate_sparse_tiling(head_size: int, q_block_tokens: int, q_tile_override: int) -> None:
    if head_size == 128:
        if q_block_tokens == 64 and q_tile_override not in (0, 64):
            raise ValueError("SPARGE_ATTN with head_dim=128 and q_block_tokens=64 requires q_tile_override 0 or 64.")
        if q_block_tokens == 256 and q_tile_override != 256:
            raise ValueError("SPARGE_ATTN with head_dim=128 and q_block_tokens=256 requires q_tile_override=256.")
        if q_block_tokens not in (64, 256):
            raise ValueError("SPARGE_ATTN with head_dim=128 requires q_block_tokens 64 or 256.")
        return
    if q_block_tokens != 64:
        raise ValueError("SPARGE_ATTN with head_dim=64 requires q_block_tokens=64.")
    if q_tile_override not in (0, 128):
        raise ValueError("SPARGE_ATTN with head_dim=64 requires q_tile_override 0 or 128.")


class SpargeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_platforms = ("xpu", "cuda")

    @classmethod
    def validate_available(cls) -> None:
        if current_omni_platform.is_cuda():
            if _load_spas_cuda_kernel() is None:
                raise ImportError(_MISSING_SPAS_CUDA)
            _validate_cuda_arch_support()
            return
        try:
            import auto_round_kernel as ark
        except ImportError as exc:
            raise ImportError(_MISSING_AUTO_ROUND) from exc
        if not callable(getattr(ark, "sparge_sage2_attn_meansim_topk_xpu", None)):
            raise ImportError(_MISSING_AUTO_ROUND)

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # The CUDA path slices the packed [real, pad] tail before the kernel,
        # so it needs no padding mask. The XPU ARK kernel rejects packed
        # metadata outright and must keep receiving the masked plan.
        return current_omni_platform.is_cuda()

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return list(_SUPPORTED_HEAD_SIZES)

    @staticmethod
    def get_name() -> str:
        return "SPARGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SpargeAttentionImpl"]:
        return SpargeAttentionImpl

    @classmethod
    def supports_cross_head_geometry(cls, q_heads: int, kv_heads: int) -> bool:
        return kv_heads > 0 and q_heads % kv_heads == 0

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        return ExecutionPathResult.unmigrated(cls.get_name(), context, path="runtime_validation_dependent")


class SpargeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        del prefix, qkv_layout
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0))

        if head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"SPARGE_ATTN supports head sizes {list(_SUPPORTED_HEAD_SIZES)}, got {head_size}.")
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError(
                "SPARGE_ATTN requires query heads to be a multiple of KV heads, "
                f"got q_heads={self.num_heads} and kv_heads={self.num_kv_heads}."
            )

        kwargs = backend_kwargs or {}
        default_q_block = 256 if head_size == 128 else 64
        self.topk = float(kwargs.get("sparge_topk", 0.5))
        self.q_block_tokens = int(kwargs.get("sparge_q_block_tokens", default_q_block))
        self.k_block_tokens = int(kwargs.get("sparge_k_block_tokens", 64))
        q_tile_override = kwargs.get("sparge_q_tile_override")
        if q_tile_override is None:
            if head_size == 64:
                q_tile_override = 128
            else:
                q_tile_override = 0 if self.q_block_tokens == 64 else 256
        self.q_tile_override = int(q_tile_override)

        if not 0.0 < self.topk <= 1.0:
            raise ValueError(f"SPARGE_ATTN top-k ratio must be in (0, 1], got {self.topk}.")
        if self.k_block_tokens != 64:
            raise ValueError(f"SPARGE_ATTN requires k_block_tokens=64, got {self.k_block_tokens}.")
        if current_omni_platform.is_cuda():
            logger.warning_once(
                "SPARGE_ATTN CUDA path uses SpargeAttn's SM-specific tiling and ignores the "
                "XPU-only q_block_tokens=%d, k_block_tokens=%d, q_tile_override=%d knobs.",
                self.q_block_tokens,
                self.k_block_tokens,
                self.q_tile_override,
            )
        else:
            _validate_sparse_tiling(head_size, self.q_block_tokens, self.q_tile_override)

        known_kwargs = {
            "sparge_topk",
            "sparge_q_block_tokens",
            "sparge_k_block_tokens",
            "sparge_q_tile_override",
        }
        unknown_kwargs = sorted(set(kwargs) - known_kwargs)
        if unknown_kwargs:
            logger.warning("SpargeAttentionImpl ignoring unknown backend kwargs: %s", unknown_kwargs)

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        is_cuda = context.platform == "cuda"
        result = ExecutionPathResult.unmigrated(
            "SPARGE_ATTN", replace(context, kernel_variant=None), path="sparge_int8"
        )
        metadata_error = _sparge_metadata_error(attn_metadata, allow_packed_prefix=is_cuda)
        if metadata_error is not None:
            return replace(result, support=CapabilityResult.unsupported(f"SPARGE_ATTN: {metadata_error}"))
        if self.dropout != 0.0:
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    f"SPARGE_ATTN: does not support dropout (dropout_p={self.dropout})."
                ),
            )
        if context.platform not in ("xpu", "cuda"):
            return replace(
                result,
                support=CapabilityResult.unsupported("SPARGE_ATTN runs on XPU or CUDA only."),
            )
        if is_cuda and self.causal:
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    "SPARGE_ATTN on CUDA does not apply a causal mask inside the selected KV blocks; "
                    "select a causal-capable backend."
                ),
            )
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            reason = "Q, K, and V must have rank 4 (batch, sequence, heads, head dimension)."
        elif any(tensor.dtype not in _SUPPORTED_DTYPES for tensor in (query, key, value)):
            reason = "Q, K, and V must be float16 or bfloat16."
        elif query.dtype != key.dtype or query.dtype != value.dtype:
            reason = "Q, K, and V dtypes must match."
        elif query.device != key.device or query.device != value.device:
            reason = "Q, K, and V devices must match."
        elif key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
            reason = "K/V shapes, Q/K batch sizes, and Q/K head dimensions must match."
        elif query.shape[-1] not in _SUPPORTED_HEAD_SIZES:
            reason = f"head dimension must be one of {list(_SUPPORTED_HEAD_SIZES)}."
        elif key.shape[2] <= 0 or query.shape[2] % key.shape[2]:
            reason = "query heads must be a multiple of KV heads."
        elif is_cuda and query.shape[2] != key.shape[2]:
            reason = "the CUDA SpargeAttn kernel requires equal query and KV head counts."
        elif any(size == 0 for tensor in (query, key, value) for size in tensor.shape):
            reason = "Q, K, and V dimensions must be nonzero."
        elif self.causal and query.shape[1] != key.shape[1]:
            reason = "causal attention requires equal query and K/V sequence lengths."
        else:
            reason = None
        if reason is not None:
            return replace(result, support=CapabilityResult.unsupported(f"SPARGE_ATTN: {reason}"))

        if is_cuda and _has_raw_packed_metadata(attn_metadata):
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    "SPARGE_ATTN on CUDA requires packed_padding for packed calls; "
                    "raw cu_seqlens describe a multi-document packing."
                ),
            )

        dtype = str(query.dtype).removeprefix("torch.")
        packed_call = attn_metadata is not None and attn_metadata.packed_padding is not None
        resolved_context = replace(
            context,
            kernel_variant=None,
            dtype=dtype,
            causal=self.causal,
            mask_mode=MaskMode.NONE,
            packing_mode=PackingMode.PACKED_PADDING if is_cuda and packed_call else PackingMode.NONE,
        )
        result = ExecutionPathResult.unmigrated("SPARGE_ATTN", resolved_context, path="sparge_int8")
        if context.parallel_strategy in (ParallelStrategy.RING, ParallelStrategy.HYBRID_ULYSSES_RING):
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    "SPARGE_ATTN does not support ring sequence parallelism: "
                    "the ring path bypasses the backend. Use Ulysses SP or disable ring SP."
                ),
            )
        allowed_mask_modes = (MaskMode.NONE, MaskMode.PADDING) if is_cuda else (MaskMode.NONE,)
        allowed_packing_modes = (PackingMode.NONE, PackingMode.PACKED_PADDING) if is_cuda else (PackingMode.NONE,)
        if (
            context.mask_mode not in allowed_mask_modes
            or context.packing_mode not in allowed_packing_modes
            or context.piecewise
            or context.paged_kv
            or context.kv_cache_dtype not in (None, "auto", "float")
            or context.parallel_strategy is not ParallelStrategy.NONE
            or context.cross_parallel_layout is not None
            or context.outer_boundaries
        ):
            return replace(
                result,
                support=CapabilityResult.unmigrated(
                    "SPARGE_ATTN supports dense XPU attention and single-document packed CUDA attention; "
                    "this execution context is not verified."
                ),
            )
        return replace(result, support=CapabilityResult.supported(), compilation_mode=CompilationMode.CUSTOM_OP)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        _validate_sparge_metadata(attn_metadata, allow_packed_prefix=True)
        if self.dropout != 0.0:
            raise ValueError(f"SPARGE_ATTN: does not support dropout (dropout_p={self.dropout}).")
        if self.causal:
            raise NotImplementedError(
                "SPARGE_ATTN: the CUDA SpargeAttn kernel selects KV blocks with is_causal but does "
                "not mask future tokens inside a block; select a causal-capable backend."
            )
        if query.shape[2] != key.shape[2]:
            raise NotImplementedError(
                "SPARGE_ATTN: the CUDA SpargeAttn kernel requires equal query and KV head counts; "
                f"got q_heads={query.shape[2]} and kv_heads={key.shape[2]}."
            )
        q_length = query.shape[1]
        q_valid, kv_valid = _cuda_packed_lengths(attn_metadata, q_length, key.shape[1])
        if q_valid <= 0 or kv_valid <= 0:
            raise ValueError(
                f"SPARGE_ATTN: packed metadata leaves an empty attention call (q_valid={q_valid}, kv_valid={kv_valid})."
            )
        query = query[:, :q_valid]
        key = key[:, :kv_valid].contiguous()
        value = value[:, :kv_valid].contiguous()
        if q_valid < _CUDA_MIN_SEQ_LEN:
            # The kernel asserts >= _CUDA_MIN_SEQ_LEN query rows, and padding the
            # short query with zeros would put pad rows into the first meansim
            # query tile, changing which KV blocks the real rows select. Keep the
            # exact result for this small call instead of approximating it.
            logger.info_once(
                "SPARGE_ATTN: %d query rows is below the SpargeAttn CUDA minimum of %d; "
                "running this call with dense attention.",
                q_valid,
                _CUDA_MIN_SEQ_LEN,
            )
            output = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=self.softmax_scale,
                is_causal=self.causal,
            ).transpose(1, 2)
        else:
            output = _sparge_attention_cuda_op(
                query.contiguous(),
                key,
                value,
                self.causal,
                self.softmax_scale,
                self.topk,
            )
        if q_valid != q_length:
            restored = output.new_zeros((output.shape[0], q_length, output.shape[2], output.shape[3]))
            restored[:, :q_valid] = output
            output = restored
        return output

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        _validate_sparge_metadata(attn_metadata)
        if self.dropout != 0.0:
            raise ValueError(f"SPARGE_ATTN: does not support dropout (dropout_p={self.dropout}).")
        return _sparge_attention_op(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.causal,
            self.softmax_scale,
            self.topk,
            self.q_tile_override,
            self.q_block_tokens,
            self.k_block_tokens,
        )
