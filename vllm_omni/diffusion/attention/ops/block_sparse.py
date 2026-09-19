# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pooling, routing and provider calls for explicit block-sparse attention."""

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
