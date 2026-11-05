# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pooling, routing and provider calls for explicit block-sparse attention."""

import functools
import importlib
import math
import os

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def validate_sparse_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    block_sizes: torch.Tensor | None,
    block_size: int = 64,
) -> None:
    """Check structural metadata without reading device values on the host."""
    for tensor in (query, key, value):
        if tensor.ndim != 4 or tensor.shape[-1] != 128 or min(tensor.shape) < 1:
            raise ValueError("Block-sparse attention expects nonempty BSHD inputs with head dimension 128")
        if tensor.stride(-1) != 1:
            raise ValueError("Block-sparse attention requires a contiguous head dimension")
    if key.shape != value.shape or query.shape[::2] != key.shape[::2]:
        raise ValueError("Block-sparse attention requires matching batch/head dimensions and matching K/V shapes")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("Block-sparse inputs must use BF16")
    batch, rows, heads, _ = query.shape
    blocks = (rows + block_size - 1) // block_size
    if indices.ndim != 4 or indices.shape[:3] != (batch, heads, blocks):
        raise ValueError("Sparse indices must have shape [B,H,ceil(Sq/block_size),capacity]")
    if indices.shape[-1] < 1 or counts.shape != indices.shape[:3]:
        raise ValueError("Sparse counts must match the query blocks and capacity must be positive")
    metadata: tuple[torch.Tensor, ...] = (indices, counts)
    if block_sizes is not None:
        key_blocks = (key.shape[1] + block_size - 1) // block_size
        if block_sizes.shape not in ((key_blocks,), (batch, key_blocks), (batch, heads, key_blocks)):
            raise ValueError("Block sizes must have shape [Kblocks], [B,Kblocks], or [B,H,Kblocks]")
        metadata += (block_sizes,)
    for tensor in metadata:
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise TypeError("Sparse metadata must be contiguous INT32")
    if any(t.device != query.device for t in (key, value, *metadata)):
        raise ValueError("Block-sparse inputs and sparse metadata must share one device")


def validate_softmax_scale(softmax_scale: float) -> None:
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a real scalar")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")


def split_tile128_map(block_map: torch.Tensor, block_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent a logical tile128 mask using 64-token compute blocks.

    Both query halves inherit the same selected KV tiles. Each selected KV
    tile contributes both halves to ONE attention reduction, preserving its
    softmax denominator. Pooling, selection and compression remain tile-128.
    """
    sizes64 = torch.stack((block_sizes.clamp(max=64), (block_sizes - 64).clamp(0, 64)), dim=-1).flatten()
    mask64 = block_map.repeat_interleave(2, -2).repeat_interleave(2, -1)
    # Partial tiles can have an entirely empty second half. Do not route it.
    mask64 &= sizes64.view(1, 1, 1, -1) > 0
    return mask64, sizes64


if not hasattr(torch.ops.vllm_omni, "fastvideo_block_sparse_attn_bshd"):

    @torch.library.custom_op("vllm_omni::fastvideo_block_sparse_attn_bshd", mutates_args=())
    def fastvideo_block_sparse_attn_bshd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        logical_blocks: int,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Run explicit block-sparse attention on BSHD tensors in 64/128-row blocks.

        Inputs and the block map may include transport-only blocks after
        ``logical_blocks``. Provider selection and any FlashInfer or cuDNN
        dispatch belong to the shared block_sparse_attn_bshd entry point.
        """
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Prefer the explicitly selected native provider when it supports
        # these tensors; retain the existing Triton provider otherwise.
        if block_size not in (64, 128):
            raise ValueError(f"unsupported FastVideo block size: {block_size}")
        if block_size == 128 or os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
                from fastvideo_kernel.triton_kernels.index import map_to_index

                if block_sparse_attn_sm100a.is_supported(q, variable_block_sizes):
                    q2k_idx, q2k_num = map_to_index(block_map)
                    out, _ = block_sparse_attn_sm100a.block_sparse_attn_sm100a(
                        q,
                        k,
                        v,
                        q2k_idx.to(torch.int32).contiguous(),
                        q2k_num.to(torch.int32).contiguous(),
                        variable_block_sizes.to(torch.int32).contiguous(),
                        need_lse=False,
                    )
                    return out.transpose(1, 2).contiguous()
                if block_size == 128:
                    raise RuntimeError(
                        "FastVideo tile128 is unavailable; select a supported Triton, cuDNN or FlashInfer provider"
                    )
            except torch.AcceleratorError:
                # A device fault can poison the context; do not launch another kernel.
                raise
            except (ImportError, RuntimeError) as exc:
                if block_size == 128:
                    raise RuntimeError("FastH3 Omni requires a supported Blackwell sparse attention kernel") from exc
                # warning_once caches its arguments. An exception retains traceback
                # frames, including this forward's query/key/value tensors.
                logger.warning_once(
                    "FASTVIDEO_VSA_SM100A=1 requested but the native Blackwell forward is "
                    "unavailable (%s); using the Triton block-sparse route instead.",
                    str(exc),
                )

        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

        logical_len = logical_blocks * 64
        out, _ = block_sparse_attn(
            q[:, :, :logical_len].contiguous(),
            k[:, :, :logical_len].contiguous(),
            v[:, :, :logical_len].contiguous(),
            block_map[..., :logical_blocks, :logical_blocks].contiguous(),
            variable_block_sizes[:logical_blocks].to(torch.int32).contiguous(),
        )
        out = out.transpose(1, 2).contiguous()
        if out.shape[1] != query.shape[1]:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, query.shape[1] - out.shape[1]))
        return out

    @fastvideo_block_sparse_attn_bshd.register_fake
    def _(query, key, value, block_map, variable_block_sizes, logical_blocks, block_size=64):
        del key, value, block_map, variable_block_sizes, logical_blocks, block_size
        return torch.empty_like(query)


fastvideo_block_sparse_attn_bshd = torch.ops.vllm_omni.fastvideo_block_sparse_attn_bshd


@functools.lru_cache(maxsize=32)
def resolve_block_sparse_provider(
    provider: str, precision: str, device: torch.device, dtype: torch.dtype, head_size: int, block_size: int = 64
) -> str:
    """Resolve sparse dispatch once per device/dtype/tile, without changing precision.

    Use the operand's device, not device zero: each worker may own a different
    accelerator. Explicit provider requests retain their strict validation.
    Availability checks run before any kernel launch, so execution faults are
    never treated as a reason to try another provider.
    """
    if block_size not in (64, 128):
        raise ValueError("block_size must be 64 or 128")
    if provider == "triton":
        from .triton_block_sparse import require_triton_sparse

        if precision != "bf16" or dtype != torch.bfloat16 or head_size != 128:
            raise ValueError("Triton VSA requires BF16 with head dimension 128")
        require_triton_sparse(device)
        return provider
    if provider == "cudnn":
        from .cudnn_block_sparse import require_cudnn_sparse

        if precision != "bf16" or dtype != torch.bfloat16 or head_size != 128:
            raise ValueError("cuDNN VSA requires BF16 with head dimension 128")
        require_cudnn_sparse(device)
        return provider
    if block_size == 128:
        if precision != "bf16" or dtype != torch.bfloat16 or head_size != 128:
            raise ValueError("tile128 VSA requires BF16 with head dimension 128")
        if device.type != "cuda":
            raise ValueError("tile128 VSA requires CUDA")
        capability = current_omni_platform.get_device_capability(device.index)
        if provider == "auto":
            if capability is not None and (capability[0] == 8 or capability == (10, 3)):
                return resolve_block_sparse_provider("triton", precision, device, dtype, head_size, block_size)
            provider = "flashinfer" if capability in ((12, 0), (12, 1)) else "fastvideo"
        if provider == "flashinfer":
            from .flashinfer_block_sparse import require_flashinfer_sparse

            require_flashinfer_sparse(precision, device)
        elif provider == "fastvideo":
            if capability != (10, 0):
                raise ValueError(
                    "FastVideo tile128 requires SM100; select the Triton or cuDNN provider on other supported GPUs"
                )
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
                from fastvideo_kernel.triton_kernels.index import map_to_index

                if not callable(block_sparse_attn_sm100a.block_sparse_attn_sm100a) or not callable(map_to_index):
                    raise ImportError("missing native tile128 entry point")
                # The Python wrapper can exist without a compatible compiled
                # kernel. Its predicate reads only metadata; probe once here,
                # before model loading, using the native BHSD layout.
                query = torch.empty(1, 1, 2 * block_size, head_size, dtype=dtype, device=device)
                sizes = torch.empty(2, dtype=torch.int32, device=device)
                if not block_sparse_attn_sm100a.is_supported(query, sizes):
                    raise ImportError("installed native kernel does not support tile128 on this device")
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "FastVideo tile128 requires fastvideo-kernel with a compatible sm100a sparse kernel; "
                    "select Triton or cuDNN if this build does not provide it"
                ) from exc
        else:
            raise ValueError(f"unsupported tile128 provider: {provider}")
        return provider
    if provider != "auto":
        return provider
    if precision not in ("bf16", "sage"):
        raise ValueError("VSA precision must be bf16 or sage")
    reason = "FlashInfer tile64 requires CUDA BF16 inputs with head dimension 128"
    if device.type != "cuda":
        if precision == "sage":
            raise ValueError(reason)
        # Non-CUDA dispatch belongs to the existing backend fallback; do not
        # import accelerator kernels while resolving its host-side metadata.
        return "fastvideo"
    if dtype == torch.bfloat16 and head_size == 128:
        from .flashinfer_block_sparse import require_flashinfer_sparse

        try:
            require_flashinfer_sparse(precision, device)
        except (ImportError, ValueError) as exc:
            if precision == "sage":
                raise
            reason = str(exc)
        else:
            logger.info_once("VSA auto selected FlashInfer %s on %s", precision, device)
            return "flashinfer"
    elif precision == "sage":
        raise ValueError(reason)
    try:
        kernel = importlib.import_module("fastvideo_kernel.block_sparse_attn").block_sparse_attn
        if not callable(kernel):
            raise ImportError("FastVideo block_sparse_attn is unavailable")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"VSA auto cannot use FlashInfer ({reason}) and fastvideo-kernel is unavailable. "
            "Install vllm-omni[vsa] for the FastVideo provider."
        ) from exc
    logger.info_once("VSA auto selected FastVideo on %s: %s", device, reason)
    return "fastvideo"


def block_map_to_indices(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a device-independent [B,H,Qblocks,Kblocks] mask to sparse rows.

    Selected key IDs are sorted ahead of -1 padding; counts bound each row,
    including empty rows. No device values are read back to the host.
    """
    if block_map.ndim != 4 or block_map.dtype != torch.bool or min(block_map.shape) < 1:
        raise ValueError("block_map must be a nonempty boolean [B,H,Qblocks,Kblocks] tensor")
    key_blocks = block_map.shape[-1]
    ids = torch.arange(key_blocks, device=block_map.device, dtype=torch.int32)
    indices = torch.where(block_map, ids, key_blocks).sort(dim=-1).values
    indices = torch.where(indices == key_blocks, -1, indices).contiguous()
    counts = block_map.sum(dim=-1, dtype=torch.int32).contiguous()
    return indices, counts


def block_sparse_attn_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    *,
    provider: str = "fastvideo",
    precision: str = "bf16",
    block_size: int = 64,
) -> torch.Tensor:
    """Dispatch explicit tile64/tile128 attention without model-specific metadata.

    FlashInfer accepts rectangular/ragged BSHD inputs. FastVideo accepts
    equally shaped tensors padded to full tiles, FP16/BF16 and the standard
    attention scale. Its optional paired-CTA padding belongs to this adapter,
    so callers always receive exactly their logical query rows.
    """
    if block_size not in (64, 128):
        raise ValueError("block_size must be 64 or 128")
    if provider == "triton":
        from .triton_block_sparse import triton_block_sparse_attention

        if precision != "bf16":
            raise ValueError("Triton VSA requires BF16 precision")
        return triton_block_sparse_attention(query, key, value, block_map, block_sizes, softmax_scale, block_size)
    if provider == "cudnn":
        from .cudnn_block_sparse import cudnn_block_sparse_attention

        if precision != "bf16":
            raise ValueError("cuDNN VSA currently supports BF16 precision")
        return cudnn_block_sparse_attention(query, key, value, block_map, block_sizes, softmax_scale, block_size)
    if provider == "flashinfer" and block_size == 64:
        from .flashinfer_block_sparse import flashinfer_block_sparse_attention

        return flashinfer_block_sparse_attention(
            query, key, value, block_map, block_sizes, softmax_scale, precision=precision
        )
    if provider not in ("fastvideo", "flashinfer") or precision != "bf16":
        raise ValueError("block-sparse attention requires fastvideo/bf16, cudnn/bf16 or flashinfer/bf16|sage")
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape or min(query.shape) < 1:
        raise ValueError("FastVideo block-sparse requires matching nonempty BSHD inputs")
    if query.shape[1] % block_size:
        raise ValueError("Block-sparse inputs must be padded to complete tiles")
    if not math.isclose(softmax_scale, query.shape[-1] ** -0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("FastVideo block-sparse requires head_size**-0.5 scaling")
    blocks = query.shape[1] // block_size
    if block_map.shape != (query.shape[0], query.shape[2], blocks, blocks) or block_map.dtype != torch.bool:
        raise ValueError("FastVideo block_map must be boolean [B,H,blocks,blocks]")
    if block_sizes is None or block_sizes.shape != (blocks,):
        raise ValueError("FastVideo requires one size per tile")
    if provider == "flashinfer":
        from .flashinfer_block_sparse import flashinfer_block_sparse_attention

        mask64, sizes64 = split_tile128_map(block_map, block_sizes.to(torch.int32))
        return flashinfer_block_sparse_attention(
            query, key, value, mask64, sizes64, query.shape[-1] ** -0.5, precision=precision
        )
    # The native SM100 paired-CTA provider requires an even block count.
    # Preserve the Triton provider's existing logical-block slicing contract.
    if blocks % 2:
        query, key, value = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, block_size)) for t in (query, key, value))
        block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=False)
        block_sizes = torch.nn.functional.pad(block_sizes, (0, 1), value=0)
    return fastvideo_block_sparse_attn_bshd(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        block_map.contiguous(),
        block_sizes.contiguous(),
        blocks,
        block_size,
    )[:, : blocks * block_size].contiguous()


def mean_pool_tiles(x: torch.Tensor, sizes: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool zero-padded BSHD tiles with FP32 accumulation.

    ``sizes`` gives the number of valid rows in each tile.
    """
    batch, seq_len, heads, dim = x.shape
    blocks = seq_len // block_size
    pooled = x.view(batch, blocks, block_size, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / sizes.view(1, -1, 1, 1).clamp_min(1)
    return pooled.permute(0, 2, 1, 3)


def build_prefix_dense_block_map(
    scores: torch.Tensor,
    num_prefix_blocks: int,
    num_sparse_blocks: int,
    topk: int,
) -> torch.Tensor:
    """Keep prefix queries/keys dense and select top-k non-prefix keys per row."""
    keep_sparse = min(topk, num_sparse_blocks)
    if keep_sparse == num_sparse_blocks:
        return torch.ones_like(scores, dtype=torch.bool)
    block_map = torch.zeros_like(scores, dtype=torch.bool)
    indices = scores[..., num_prefix_blocks:].topk(keep_sparse, dim=-1).indices + num_prefix_blocks
    block_map.scatter_(-1, indices, True)
    block_map[..., :num_prefix_blocks] = True
    block_map[:, :, :num_prefix_blocks, :] = True
    return block_map
