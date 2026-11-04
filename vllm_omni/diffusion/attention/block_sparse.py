# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compose block selection with a provider-specific sparse attention adapter."""

import copy
import itertools
import math
import weakref

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    BlockSparseAdapter,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.block_selection.abstract import validate_protected_kv_prefix
from vllm_omni.diffusion.attention.block_selection.registry import get_block_selector
from vllm_omni.diffusion.attention.capabilities import (
    CapabilityResult,
    CompilationMode,
    ExecutionPathResult,
    MaskMode,
    PackingMode,
    ParallelStrategy,
)
from vllm_omni.diffusion.attention.contracts import MethodCapabilities
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.data import BlockSparseAttentionSpec

# Compiled graphs receive a CPU scalar owner handle as a runtime tensor input,
# never a per-layer Python constant or a snapshot of readiness.
# Weak references do not extend model/request lifetimes; tokens are never reused.
_request_owners: weakref.WeakValueDictionary[int, "BlockSparseAttention"] = weakref.WeakValueDictionary()
_request_owner_ids = itertools.count()


@torch.library.custom_op("vllm_omni::block_sparse_request", mutates_args=())
def _block_sparse_request(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, prefix: int, owner_handle: torch.Tensor
) -> torch.Tensor:
    owner = _request_owners.get(owner_handle.item())
    if owner is None:
        raise RuntimeError("Block-sparse request owner was released; keep the attention layer alive during execution")
    return owner._dispatch_request(query, key, value, prefix)


@_block_sparse_request.register_fake
def _block_sparse_request_fake(query, key, value, prefix, owner_handle):
    # No readiness lookup, allocation of real buffers, or trial execution while tracing.
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


def validate_block_sparse_parallel(config):
    """Admit local execution or pure strict Ulysses; kernels consume local heads."""
    parallel = getattr(config, "parallel_config", None)
    degree = getattr(parallel, "ulysses_degree", 1) or 1
    if (
        (getattr(parallel, "sequence_parallel_size", degree) or degree) != degree
        or any((getattr(parallel, name, 1) or 1) != 1 for name in ("ring_degree", "allgather_degree"))
        or getattr(parallel, "use_hsdp", False)
    ):
        raise ValueError("Block-sparse attention supports only local execution or pure strict Ulysses")
    if degree > 1:
        if getattr(parallel, "ulysses_mode", "strict") != "strict":
            raise ValueError("Block-sparse attention requires strict Ulysses mode")
        if any(
            (getattr(parallel, name, 1) or 1) != 1
            for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "cfg_parallel_size")
        ) or getattr(config, "enable_distributed_layerwise_offload", False):
            raise ValueError("Block-sparse Ulysses does not support combined parallel modes")
    return degree


class BlockSparseBackend(AttentionBackend):
    """Method-level preflight capabilities, independent of the dense provider.

    Only producer-validated single-document suffix padding is supported. Masks,
    multi-document packing, piecewise spans, prefix slicing and paged KV retain
    AttentionBackend's conservative False defaults. The bound execution adapter
    remains on BlockSparseAttention and establishes actual request support.
    """

    strategy_capabilities = MethodCapabilities(local_execution=True)

    @staticmethod
    def get_name() -> str:
        return "BLOCK_SPARSE"

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # Shared orchestration trims both Q and K/V using the typed contract.
        return True

    @staticmethod
    def get_impl_cls() -> type["BlockSparseAttention"]:
        return BlockSparseAttention

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # No duplicate provider support table: actual tensors require preparation.
        return []

    @classmethod
    def resolve_capabilities(cls, context):
        return ExecutionPathResult(
            backend=cls.get_name(),
            path="block_sparse_unprepared",
            support=CapabilityResult.unsupported(
                "Block-sparse support requires actual request preparation; "
                "query the bound Attention layer after eager warmup"
            ),
            compilation_mode=CompilationMode.EAGER_ONLY,
            platform=context.platform,
            kernel_variant=None,
            parallel_strategy=context.parallel_strategy,
        )


class BlockSparseAttention(AttentionImpl[AttentionMetadata]):
    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_size,
        scale,
        causal,
        layout,
        spec: BlockSparseAttentionSpec,
        *,
        adapter: BlockSparseAdapter,
    ):
        self.block_size = spec.block_size
        selection = spec.selection
        self.selector = get_block_selector(selection["name"])(selection["config"])
        if causal or layout not in (None, "BSHD", "BSND"):
            raise ValueError("Block-sparse attention requires noncausal BSHD attention")
        if head_size <= 0:
            raise ValueError("Block-sparse attention requires a positive Q/K head dimension")
        if min(num_heads, num_kv_heads) <= 0 or num_heads % num_kv_heads:
            raise ValueError("Block-sparse attention requires positive Q heads divisible by KV heads")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Block-sparse attention requires a finite positive scale")
        if not torch.cuda.is_available():
            raise ValueError("The initial Block-sparse attention router requires CUDA")
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.num_heads, self.num_kv_heads, self.scale = num_heads, num_kv_heads, scale
        self.head_size = head_size
        cfg = get_current_diffusion_config_or_none()
        if cfg is not None:
            validate_block_sparse_parallel(cfg)
            if getattr(cfg, "fa_deterministic", False):
                raise ValueError("Block-sparse attention does not support fa_deterministic")
            if getattr(cfg, "diffusion_kv_cache_dtype", None) not in (None, "auto", "float"):
                raise ValueError("Block-sparse attention does not support KV quantization")
            if getattr(cfg, "diffusion_kv_mode", "dense_legacy") != "dense_legacy":
                raise ValueError("Block-sparse attention does not support paged KV")
        self.selector.prepare(self.block_size, head_size, self.device)
        self.adapter = adapter
        # Geometry-only outcomes: never retain request tensors or exception tracebacks.
        # None denotes successful execution and synchronization; strings record failures.
        self._request_preparation: dict[tuple, str | None] = {}
        adapter.prepare(spec.implementation, head_size, num_heads, num_kv_heads, self.device, self.block_size)
        self._register_request_owner()
        init_logger(__name__).info(
            "Configured block-sparse attention (request preparation pending): provider=%s kernel=%s dependency=%s "
            "native heads=%s/%s selector=%s block_size=%s",
            adapter.provider,
            adapter.kernel_variant,
            adapter.dependency_version,
            num_heads,
            num_kv_heads,
            type(self.selector).__name__,
            self.block_size,
        )

    def _register_request_owner(self):
        owner_id = next(_request_owner_ids)
        # A CPU scalar is a runtime graph input, not a per-layer Python constant.
        # Read it only inside the opaque op to avoid specializing owner identity.
        self._request_owner_handle = torch.tensor(owner_id, dtype=torch.int64, device="cpu")
        _request_owners[owner_id] = self

    def __getstate__(self):
        # Handles are process-local identity, and readiness is evidence from this
        # instance's actual execution. Neither belongs in copied/serialized state.
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ("_request_owner_handle", "_request_preparation")
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._request_preparation = {}
        self._register_request_owner()

    def __deepcopy__(self, memo):
        cloned = type(self).__new__(type(self))
        memo[id(self)] = cloned
        cloned.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return cloned

    def __copy__(self):
        raise TypeError(
            "Shallow copying block-sparse attention is unsupported; use copy.deepcopy() or construct a new layer"
        )

    @staticmethod
    def _validate_real_length(name, length, storage_length):
        if isinstance(length, bool) or not isinstance(length, int):
            raise ValueError(f"{name} must be a host integer")
        if not 0 < length <= storage_length:
            raise ValueError(f"{name} must be positive and within its tensor sequence length")

    @staticmethod
    def _validate_cumulative_lengths(name, lengths, sizes):
        # Values are producer-validated. Inspect only tensor metadata: reading
        # device scalars would introduce an unnecessary host synchronization.
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.ndim != 1
            or lengths.shape[0] not in sizes
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must be a one-dimensional integer tensor with length in {sizes}")

    def _normalize_request(self, query, key, value, metadata):
        """Validate and trim a producer-validated B=1 [real, padding] request.

        Returns executed Q/K/V, protected prefix and the original query length.
        Cumulative-length values are never read; the typed producer contract is
        authoritative. Raw packed metadata alone cannot establish this contract.
        """
        if any(t.ndim != 4 for t in (query, key, value)):
            raise ValueError("Block-sparse attention requires four-dimensional BSHD Q/K/V")
        if (
            query.shape[0] != key.shape[0]
            or key.shape[:3] != value.shape[:3]
            or query.shape[-1] != self.head_size
            or key.shape[-1] != self.head_size
            or min(query.shape[0], query.shape[1], key.shape[1], value.shape[-1]) <= 0
            or query.shape[2] != self.num_heads
            or key.shape[2] != self.num_kv_heads
        ):
            raise ValueError("Incompatible Block-sparse attention Q/K/V geometry; native head mapping is required")
        if query.dtype != key.dtype or key.dtype != value.dtype:
            raise ValueError("Block-sparse attention requires matching Q/K/V dtypes")
        if any(t.device != self.device for t in (query, key, value)):
            raise ValueError("Block-sparse attention inputs must use the prepared CUDA device")
        original_q_length = query.shape[1]
        prefix = 0
        if metadata is not None:
            if any(
                getattr(metadata, name) is not None
                for name in ("attn_mask", "joint_attn_mask", "full_attn_spans", "query_ranges")
            ):
                raise ValueError("Block-sparse attention does not support masks or piecewise attention")
            extra = metadata.extra
            if not isinstance(extra, dict):
                raise ValueError("Block-sparse attention metadata.extra must be a dictionary")
            if extra.get("kv_cache_dtype") not in (None, "auto", "float"):
                raise ValueError("Block-sparse attention does not support quantized KV")
            padding = metadata.packed_padding
            packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "valid_kv_length")
            if padding is None:
                if any(extra.get(name) is not None for name in packed_keys):
                    raise ValueError("Block-sparse packed extras require typed PackedPaddingMetadata")
            else:
                if not isinstance(padding, PackedPaddingMetadata):
                    raise ValueError("packed_padding must be PackedPaddingMetadata")
                if query.shape[0] != 1:
                    raise ValueError("Block-sparse packed padding requires B=1 and a single real document")
                self._validate_real_length("q_length", padding.q_length, query.shape[1])
                self._validate_real_length("kv_length", padding.kv_length, key.shape[1])
                self._validate_cumulative_lengths("packed_padding.cu_seqlens_q", padding.cu_seqlens_q, (2,))
                self._validate_cumulative_lengths("packed_padding.cu_seqlens_k", padding.cu_seqlens_k, (2,))
                # MiniMax also supplies the original [0, real, total] tensors.
                # Accept these redundant extras only under the typed contract;
                # four or more boundaries imply unsupported multi-document data.
                for name in ("cu_seqlens_q", "cu_seqlens_k"):
                    if extra.get(name) is not None:
                        self._validate_cumulative_lengths(name, extra[name], (2, 3))
                for name, real_length in (
                    ("max_seqlen_q", padding.q_length),
                    ("max_seqlen_k", padding.kv_length),
                    ("valid_kv_length", padding.kv_length),
                ):
                    if extra.get(name) is not None:
                        self._validate_real_length(name, extra[name], real_length)
                        if extra[name] != real_length:
                            raise ValueError(f"{name} disagrees with PackedPaddingMetadata")
                # Trim Q too: padding in its final real block would otherwise
                # contaminate the pooled routing scores of valid query rows.
                query = query[:, : padding.q_length]
                key = key[:, : padding.kv_length]
                value = value[:, : padding.kv_length]
            prefix = extra.get("protected_kv_prefix", 0)
        validate_protected_kv_prefix(prefix, key.shape[1])
        self.adapter.validate_inputs(query, key, value)
        self.selector.validate_request(query, key, prefix)
        return query, key, value, prefix, original_q_length

    def resolve_execution_path(self, context, query, key, value, metadata):
        reason = None
        if (
            context.platform != "cuda"
            or context.parallel_strategy is not ParallelStrategy.NONE
            or context.outer_boundaries
            or context.paged_kv
            or context.piecewise
            or context.mask_mode is not MaskMode.NONE
            or context.packing_mode not in (PackingMode.NONE, PackingMode.PACKED_PADDING)
            or context.kv_cache_dtype not in (None, "auto", "float")
            or context.causal is True
        ):
            reason = (
                "Block-sparse attention requires noncausal, mask-free CUDA attention with at most typed suffix padding "
                "without parallel wrappers"
            )
        else:
            try:
                if context.packing_mode is PackingMode.PACKED_PADDING and (
                    metadata is None or not isinstance(metadata.packed_padding, PackedPaddingMetadata)
                ):
                    raise ValueError("PACKED_PADDING requires typed PackedPaddingMetadata")
                query, key, value, prefix, _ = self._normalize_request(query, key, value, metadata)
                signature = self._request_signature(query, key, value, prefix)
                reason = self._request_preparation.get(signature, self._unprepared_reason())
            except ValueError as exc:
                reason = str(exc)
        return ExecutionPathResult(
            backend=self.adapter.provider,
            path=f"{self.adapter.kernel_variant}_block_sparse",
            support=CapabilityResult.unsupported(reason) if reason else CapabilityResult.supported(),
            compilation_mode=CompilationMode.CUSTOM_OP,
            platform=context.platform,
            kernel_variant=self.adapter.kernel_variant,
            parallel_strategy=context.parallel_strategy,
        )

    def _request_signature(self, query, key, value, prefix):
        # Instance configuration is fixed. Include every request property that can
        # affect routing capacity or kernel specialization, including Q/K/V strides.
        return (
            self.block_size,
            self.scale,
            prefix,
            *((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in (query, key, value)),
        )

    @staticmethod
    def _unprepared_reason():
        return (
            "Block-sparse request is not prepared: run an eager forward or prepare_request() "
            "with this geometry to establish execution support"
        )

    def _prepare_and_execute(self, query, key, value, prefix, signature):
        if torch.compiler.is_compiling():
            raise RuntimeError(self._unprepared_reason())
        with torch.accelerator.device_index(self.device.index):
            try:
                output = self._execute(query, key, value, prefix)
                # A launch alone does not establish support: surface asynchronous
                # kernel errors before publishing a successful preparation outcome.
                torch.accelerator.synchronize(self.device)
            except Exception as exc:
                self._request_preparation[signature] = (
                    f"Block-sparse request preparation failed ({type(exc).__name__}): {exc}. "
                    "Correct the request or dependency issue and retry eager preparation"
                )
                raise
        self._request_preparation[signature] = None
        return output

    def prepare_request(self, query, key, value, metadata=None):
        """Explicit eager preparation to establish support for this signature.

        This is repeatable, including after a failed attempt. It establishes kernel
        execution support, not approximation quality or generation-quality parity.
        """
        query, key, value, prefix, _ = self._normalize_request(query, key, value, metadata)
        signature = self._request_signature(query, key, value, prefix)
        with torch.inference_mode():
            self._prepare_and_execute(query, key, value, prefix, signature)

    def forward(self, query, key, value, metadata=None):
        query, key, value, prefix, original_q_length = self._normalize_request(query, key, value, metadata)
        # Dispatch is opaque to Dynamo: exact readiness signatures are evaluated
        # against real tensors at runtime and cannot specialize symbolic lengths.
        output = _block_sparse_request(query, key, value, prefix, self._request_owner_handle)
        if query.shape[1] != original_q_length:
            tail = output.new_zeros((output.shape[0], original_q_length - query.shape[1], *output.shape[2:]))
            output = torch.cat((output, tail), dim=1)
        return output.contiguous()

    def _dispatch_request(self, query, key, value, prefix):
        """Runtime owner for both eager and compiled request preparation.

        This executes outside Dynamo, including for fullgraph=True. A new actual
        request is executed and synchronized before publishing support.
        """
        signature = self._request_signature(query, key, value, prefix)
        if signature not in self._request_preparation or self._request_preparation[signature] is not None:
            return self._prepare_and_execute(query, key, value, prefix, signature)
        return self._execute(query, key, value, prefix)

    def _execute(self, query, key, value, prefix):
        selection = self.selector.select(query, key, self.scale, prefix)
        return self.adapter.execute(query, key, value, selection, self.scale, self.block_size)

    def forward_cuda(self, query, key, value, attn_metadata=None):
        return self.forward(query, key, value, attn_metadata)
