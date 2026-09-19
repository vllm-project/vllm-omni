# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Write each rank-major head bucket directly into a complete token/head output."""


def assemble_output_bucket(destination, recv, first_head):
    """Restore rank-major [W,T,Hb,D] into the complete [T,H,D] output.

    first_head is the bucket offset within each rank's head range.
    This is a strided copy into the caller's existing output buffer.
    """
    tokens, heads, dim = destination.shape
    world, recv_tokens, width, recv_dim = recv.shape
    if (tokens, dim) != (recv_tokens, recv_dim) or heads % world:
        raise ValueError("Bucket output shape does not match its destination")
    if first_head < 0 or first_head + width > heads // world:
        raise ValueError("Bucket head range exceeds each rank's head group")
    target = destination.view(tokens, world, heads // world, dim)
    target[:, :, first_head : first_head + width, :].copy_(recv.permute(1, 0, 2, 3))
