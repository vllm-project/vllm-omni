# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FlashAttention backend adapter for named KV branches.

Validates the existing runner-owned pool layout and provides zero-copy
four-dimensional K/V cache views in FA's native layout:

    key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: [num_blocks, block_size, num_kv_heads, head_size]

The views are obtained via ``transpose(1, 2).split(head_size, dim=-1)``
on the logical cache ``(num_blocks, num_kv_heads, block_size, 2*head_size)``,
which is the same transformation FA applies internally.
"""

from __future__ import annotations

import torch

from vllm_omni.worker.named_kv.runtime import NamedCausalKVBranch


class UnsupportedNamedKVGraphError(Exception):
    """A known-unsupported configuration; the caller should select the old path."""


class FlashAttentionKVBranchAdapter:
    """Validate pool layout and provide 4-D K/V cache views for the custom op."""

    def __init__(self, branch: NamedCausalKVBranch) -> None:
        self.branch = branch
        self._validate()

    def _validate(self) -> None:
        # ``backend`` is ``type[AttentionBackend]``, not an instance.
        backend_cls = self.branch.backend
        if backend_cls.__name__ != "FlashAttentionBackend":
            raise UnsupportedNamedKVGraphError(
                f"Named KV FA adapter requires FlashAttentionBackend, got {backend_cls.__name__}"
            )

        spec = self.branch.kv_cache_spec
        if spec.page_size_padded is not None:
            raise UnsupportedNamedKVGraphError("Padded KV cache pages are not supported")
        if spec.dtype != torch.bfloat16:
            raise UnsupportedNamedKVGraphError(f"Named KV graph requires bfloat16, got {spec.dtype}")
        if spec.block_size % 16 != 0:
            raise UnsupportedNamedKVGraphError(f"Block size must be a multiple of 16, got {spec.block_size}")

        # Validate actual tensor shape, dtype, and device.
        expected_logical = (
            self.branch.num_blocks,
            spec.num_kv_heads,
            self.branch.block_size,
            2 * spec.head_size,
        )
        for name in self.branch.layer_names:
            cache = self.branch.kv_caches[name]
            if cache.ndim != 4:
                raise UnsupportedNamedKVGraphError(f"Named KV cache must be 4-D, got {cache.ndim}D for {name!r}")
            if tuple(cache.shape) != expected_logical:
                raise UnsupportedNamedKVGraphError(
                    f"Named KV cache shape mismatch for {name!r}: {tuple(cache.shape)} != {expected_logical}"
                )
            # Accept only the two native packed physical orders. Shape alone
            # does not establish the scatter kernel's stride contract.
            heads, tokens, width = expected_logical[1:]
            supported_strides = {
                (heads * tokens * width, tokens * width, width, 1),
                (heads * tokens * width, width, heads * width, 1),
            }
            if cache.stride() not in supported_strides:
                raise UnsupportedNamedKVGraphError(f"Unsupported named KV cache strides for {name!r}: {cache.stride()}")
            if cache.dtype != spec.dtype:
                raise UnsupportedNamedKVGraphError(
                    f"Named KV cache dtype mismatch for {name!r}: {cache.dtype} != {spec.dtype}"
                )
            if cache.device != self.branch.device:
                raise UnsupportedNamedKVGraphError(f"Named KV cache device mismatch for {name!r}")

    def get_kv_caches(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return 4-D K/V views: ``[num_blocks, block_size, Hkv, D]``.

        FA logical cache:
            ``(num_blocks, num_kv_heads, block_size, 2*head_size)``

        ``transpose(1, 2)``:
            ``(num_blocks, block_size, num_kv_heads, 2*head_size)``

        ``split(head_size, dim=-1)``:
            K and V each ``(num_blocks, block_size, num_kv_heads, head_size)``

        These are zero-copy views (non-contiguous but share storage).
        """
        spec = self.branch.kv_cache_spec
        k_caches: list[torch.Tensor] = []
        v_caches: list[torch.Tensor] = []
        for name in self.branch.layer_names:
            cache = self.branch.kv_caches[name]
            key_cache, value_cache = cache.transpose(1, 2).split(spec.head_size, dim=-1)
            k_caches.append(key_cache)
            v_caches.append(value_cache)
        return k_caches, v_caches


__all__ = [
    "FlashAttentionKVBranchAdapter",
    "UnsupportedNamedKVGraphError",
]
