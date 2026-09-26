# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pooling, routing and provider calls for explicit block-sparse attention."""

import math
import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


if not hasattr(torch.ops.vllm_omni, "fastvideo_block_sparse_attn_bshd"):

    @torch.library.custom_op("vllm_omni::fastvideo_block_sparse_attn_bshd", mutates_args=())
    def fastvideo_block_sparse_attn_bshd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        logical_blocks: int,
    ) -> torch.Tensor:
        """Run explicit block-sparse attention on BSHD tensors in 64-row blocks.

        Inputs and the block map may include transport-only blocks after
        ``logical_blocks``. The Triton provider sees only logical blocks and
        the wrapper restores the original row count. Native provider selection
        and its supported-input checks remain owned by FastVideo.
        """
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Prefer the explicitly selected native provider when it supports
        # these tensors; retain the existing Triton provider otherwise.
        if os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1":
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
            except torch.AcceleratorError:
                # A device fault can poison the context; do not launch another kernel.
                raise
            except (ImportError, RuntimeError) as exc:
                logger.warning_once(
                    "FASTVIDEO_VSA_SM100A=1 requested but the native Blackwell forward is "
                    "unavailable (%s); using the Triton block-sparse route instead.",
                    exc,
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
    def _(query, key, value, block_map, variable_block_sizes, logical_blocks):
        del key, value, block_map, variable_block_sizes, logical_blocks
        return torch.empty_like(query)


fastvideo_block_sparse_attn_bshd = torch.ops.vllm_omni.fastvideo_block_sparse_attn_bshd


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
) -> torch.Tensor:
    """Dispatch explicit tile64 attention without model-specific metadata.

    FlashInfer accepts rectangular/ragged BSHD inputs. FastVideo accepts
    equally shaped tensors padded to full tiles, FP16/BF16 and the standard
    attention scale. Its optional paired-CTA padding belongs to this adapter,
    so callers always receive exactly their logical query rows.
    """
    if provider == "flashinfer":
        from .flashinfer_block_sparse import flashinfer_block_sparse_attention

        return flashinfer_block_sparse_attention(
            query, key, value, block_map, block_sizes, softmax_scale, precision=precision
        )
    if provider != "fastvideo" or precision != "bf16":
        raise ValueError("block-sparse attention requires fastvideo/bf16 or flashinfer/bf16|sage")
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape or min(query.shape) < 1:
        raise ValueError("FastVideo tile64 requires matching nonempty BSHD inputs")
    if query.shape[1] % 64:
        raise ValueError("FastVideo tile64 inputs must be padded to complete 64-token tiles")
    if not math.isclose(softmax_scale, query.shape[-1] ** -0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("FastVideo tile64 requires head_size**-0.5 scaling")
    blocks = query.shape[1] // 64
    if block_map.shape != (query.shape[0], query.shape[2], blocks, blocks) or block_map.dtype != torch.bool:
        raise ValueError("FastVideo block_map must be boolean [B,H,blocks,blocks]")
    if block_sizes is None or block_sizes.shape != (blocks,):
        raise ValueError("FastVideo requires one size per tile")
    # The native SM100 paired-CTA provider requires an even block count.
    # Preserve the Triton provider's existing logical-block slicing contract.
    if blocks % 2:
        query, key, value = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 64)) for t in (query, key, value))
        block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=False)
        block_sizes = torch.nn.functional.pad(block_sizes, (0, 1), value=0)
    return fastvideo_block_sparse_attn_bshd(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        block_map.contiguous(),
        block_sizes.contiguous(),
        blocks,
    )[:, : blocks * 64].contiguous()


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
