# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The paging unit for AR-Diffusion KV, importable by model code without the runner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backend import MultipleOf


def paging_block_size(tokens_per_frame: int, supported: Sequence[int | MultipleOf] | None = None) -> int:
    """Choose the paging unit, which is not the eviction unit.

    AR-Diffusion evicts by frame; the attention kernel tiles by a fixed number
    of tokens. Using the frame as the block binds every output resolution to
    that kernel constraint, so the resolutions that run today do so by
    arithmetic accident -- and the LingBot World v2 checkpoint's own default
    832x480 is not one of them:

        (480/16) x (832/16) = 30 x 52 = 1560 tokens per frame, 1560 % 16 = 8

    A frame that is already a legal block stays one, so every resolution that
    works today keeps precisely the paging it has now. Anything else pages at
    the finest legal unit, which costs a longer block table and gives back the
    rounding waste of a block one whole frame wide -- at 832x480 that block is
    over a gigabyte.

    What counts as legal is asked, not assumed. ``supported`` carries vLLM's
    own vocabulary: a plain int is that exact size, ``MultipleOf(b)`` is any
    positive multiple of ``b``. The kernels AR-Diffusion reaches agree on
    multiples of 16, but others in the same tree do not -- ``hpc_attn`` takes
    only 64, and FlashInfer offers pages of 128 and up on Blackwell alone. A
    constant here would be a silently wrong block size on the first machine
    whose kernel differs, and a wrong block size is not an error: it is a
    kernel reading tokens that were never written.
    """
    if supported is None:
        # Imported here, not at module scope: paged_attention registers a
        # custom op on import, and the tests that pop it from sys.modules to
        # re-import it depend on nothing else holding it first.
        from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import supported_kernel_block_sizes

        supported = supported_kernel_block_sizes()
    if not supported:
        raise ValueError("The attention kernel advertises no legal block size.")

    def legal(size: int) -> bool:
        return size > 0 and any(
            size == entry if isinstance(entry, int) else size % entry.base == 0 for entry in supported
        )

    if legal(tokens_per_frame):
        return tokens_per_frame
    finest = min(entry if isinstance(entry, int) else entry.base for entry in supported)
    if not legal(finest):
        raise ValueError(f"No legal paging unit among {supported!r}.")
    return finest
