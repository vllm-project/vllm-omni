# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select model adapters and share receive buffers across sequential blocks."""

import torch


def supports_block_group(blocks):
    # Auxiliary stacks keep their existing block transport.
    return bool(blocks) and all(type(b).__name__ in {"MiniMaxH3DiTBlock", "MiniMaxH3TokenRefinerBlock"} for b in blocks)


class HeadAdapterFactory:
    def __init__(self, group, buckets=2):
        self.group, self.buckets = group, buckets
        self.stream = torch.npu.Stream()
        self.workspaces = {}

    def __call__(self, block):
        if type(block).__name__ == "MiniMaxH3TokenRefinerBlock":
            return []  # Replicated text refinement precedes sequence sharding.
        if type(block).__name__ != "MiniMaxH3DiTBlock":
            raise ValueError(f"No verified head adapter for {type(block).__name__}")
        from .models.minimax_h3.h3_bucket_adapter import H3BucketAdapter

        return [H3BucketAdapter(block.attn, self.group, self.buckets, stream=self.stream, workspaces=self.workspaces)]
