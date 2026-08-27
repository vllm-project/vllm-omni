# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the xDiT authors.
#
# This module is adapted from xDiT (https://github.com/xdit-project/xdit)

from abc import ABC, abstractmethod
from functools import wraps

import torch

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import (
    PipeFusionCacheIdentity,
    get_pipefusion_runtime,
    is_pipefusion_initialized,
)


class PipeFusionRotaryEmbeddingMixin(ABC):
    """
    Mixin for rotary embedding modules that need PipeFusion patch slicing.

    The owning RoPE module computes and caches full-resolution embeddings;
    this mixin only slices the returned embedding during PipeFusion patch mode.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        forward = cls.__dict__.get("forward")
        if callable(forward) and is_pipefusion_initialized():  # if PipeFusion is initialized

            @wraps(forward)
            def wrapped_forward(self, *args, **kwargs):
                rotary_emb = forward(self, *args, **kwargs)
                if not get_pipefusion_runtime().patch_mode:
                    return rotary_emb
                return self.pipefusion_slice_rotary_emb(rotary_emb, *args, **kwargs)

            cls.forward = wrapped_forward

    @staticmethod
    def _slice_rope(
        re: torch.Tensor,
        split_dim: int,
        dim_size: int,
        num_pipeline_patch: int,
        patch_idx: torch.Tensor,
    ) -> torch.Tensor:
        base_patch_size = dim_size // num_pipeline_patch
        start = patch_idx * base_patch_size
        is_last = patch_idx == num_pipeline_patch - 1
        length = base_patch_size + is_last * (dim_size - start - base_patch_size)
        re = torch.narrow(re, split_dim, start, length)
        # Reshape back to [1, seq, 1, dim]
        return re.reshape(1, -1, 1, re.shape[-1])

    def pipefusion_slice_rotary_emb(
        self,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        num_frames: int,
        height: int,
        width: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slice RoPE embeddings to match the current PipeFusion patch.

        Only applies when PipeFusion is in patched mode; otherwise returns
        the embeddings unchanged.

        Returns:
            Sliced RoPE tuple for the current patch.
        """
        runtime = get_pipefusion_runtime()

        cache_key = (runtime.pipeline_patch_idx, num_frames, height, width)
        if not hasattr(self, "_cached_sliced_rope"):
            self._cached_sliced_rope = {}
        if cache_key in self._cached_sliced_rope:
            return self._cached_sliced_rope[cache_key]

        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        num_pipeline_patch = runtime.num_pipeline_patch
        patch_idx = runtime.patch_idx_tensor

        if runtime.split_dim == "temporal":
            split_dim = 0
            dim_size = ppf
        else:
            split_dim = 1
            dim_size = pph

        self._cached_sliced_rope[cache_key] = tuple(
            # [1, ppf*pph*ppw, 1, dim] -> [ppf, pph, ppw, dim]
            self._slice_rope(re.reshape(ppf, pph, ppw, -1), split_dim, dim_size, num_pipeline_patch, patch_idx)
            for re in rotary_emb
        )
        return self._cached_sliced_rope[cache_key]


class PipeFusionSelfAttentionMixin(ABC):
    """
    Mixin for self-attention modules in PipeFusion.

    In patch mode, maintains full K/V caches across patches so that
    each patch's query can attend to the full sequence.

    Maintains separate KV caches per request, sequence, and CFG branch
    to prevent concurrent requests or CFG negative predictions from
    contaminating another cache.
    """

    _kv_caches: dict[PipeFusionCacheIdentity, tuple[torch.Tensor, torch.Tensor]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        init = cls.__dict__.get("__init__")
        if is_pipefusion_initialized() and callable(init):

            @wraps(init)
            def wrapped_init(self, *args, **kwargs):
                init(self, *args, **kwargs)
                self._kv_caches = {}
                for module in self.modules():
                    if isinstance(module, Attention):
                        self._pipefusion_patch_attention(module)

            cls.__init__ = wrapped_init

    def _pipefusion_patch_attention(self, attention: Attention):
        original_forward = attention.forward

        def pipefusion_forward(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata=None
        ) -> torch.Tensor:
            key, value = self._pipefusion_update_kv_cache(key, value)
            return original_forward(query, key, value, attn_metadata)

        attention.forward = pipefusion_forward

    def pipefusion_reset_cache(self, request_id: str | None = None, sequence_id: int | None = None) -> None:
        if request_id is None and sequence_id is None or not hasattr(self, "_kv_caches"):
            self._kv_caches = {}
            return
        self._kv_caches = {
            cache_identity: cache
            for cache_identity, cache in self._kv_caches.items()
            if cache_identity[0] != request_id or cache_identity[1] != sequence_id
        }

    def _get_kv_cache(self, cache_key: PipeFusionCacheIdentity) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_key not in self._kv_caches:
            raise RuntimeError(
                f"PipeFusion KV cache for {cache_key!r} is missing. "
                "Run at least one warmup step before patched execution."
            )
        return self._kv_caches[cache_key]

    def _set_kv_cache(self, cache_key: PipeFusionCacheIdentity, k: torch.Tensor, v: torch.Tensor):
        """Store the KV cache for the given request/sequence/CFG identity."""
        self._kv_caches[cache_key] = (k, v)

    def _pipefusion_update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update and return full K/V for the current patch.

        In patch mode, inserts the current patch's K/V into the full cache
        and returns the full K/V. In non-patch mode, just stores K/V directly.

        Uses the PipeFusion runtime cache identity to select between
        request, sequence, and CFG-branch K/V caches.

        The token sequence is flattened as [frames, height, width], so
        height-based patches are NOT contiguous in the flat sequence.
        We reshape to 5D [B, ppf, pph, ppw, heads, dim] to write
        at the correct height positions via view-based slicing.

        Args:
            key: Current patch's key tensor [B, patch_seq, heads, dim].
            value: Current patch's value tensor [B, patch_seq, heads, dim].

        Returns:
            (full_key, full_value) for attention computation.
        """
        runtime = get_pipefusion_runtime()
        cache_identity = runtime.cache_identity
        if runtime.patch_mode:
            full_k, full_v = self._get_kv_cache(cache_identity)
            ppf = runtime.ppf
            pph = runtime.pph
            ppw = runtime.ppw
            num_pipeline_patch = runtime.num_pipeline_patch
            patch_idx = runtime.patch_idx_tensor
            B, _, heads, dim = key.shape

            if runtime.split_dim == "temporal":
                base_patch_size = ppf // num_pipeline_patch
                patch_start = patch_idx * base_patch_size
                is_last = patch_idx == num_pipeline_patch - 1
                patch_end = patch_start + base_patch_size + is_last * (ppf - patch_start - base_patch_size)

                # Temporal split: tokens are contiguous in [f, h, w] order
                # because frames are the outermost dimension.
                # Patch covers f∈[f_start, f_end), token range is contiguous.
                tok_start = patch_start * pph * ppw
                tok_end = patch_end * pph * ppw
                tok_len = tok_end - tok_start
                full_k.narrow(1, tok_start, tok_len).copy_(key)
                full_v.narrow(1, tok_start, tok_len).copy_(value)
            else:
                base_patch_size = pph // num_pipeline_patch
                patch_start = patch_idx * base_patch_size
                is_last = patch_idx == num_pipeline_patch - 1
                patch_end = patch_start + base_patch_size + is_last * (pph - patch_start - base_patch_size)

                # Height split: tokens are NON-contiguous (interleaved by frames).
                # Reshape to 5D [B, ppf, pph, ppw, heads, dim] and slice height.
                pph_patch = patch_end - patch_start
                key_5d = key.reshape(B, ppf, pph_patch, ppw, heads, dim)
                value_5d = value.reshape(B, ppf, pph_patch, ppw, heads, dim)
                full_k_5d = full_k.view(B, ppf, pph, ppw, heads, dim)
                full_v_5d = full_v.view(B, ppf, pph, ppw, heads, dim)
                patch_len = patch_end - patch_start
                full_k_5d.narrow(2, patch_start, patch_len).copy_(key_5d)
                full_v_5d.narrow(2, patch_start, patch_len).copy_(value_5d)

            return full_k, full_v
        else:
            if runtime.update_warmup_cache:
                self._set_kv_cache(cache_identity, key, value)
            return key, value


class PipeFusionTransformerMixin(ABC):
    """
    Mixin for transformer models that participate in PipeFusion.

    Provides helper methods for:
    1. Slicing RoPE embeddings to match the current patch
    2. Conditional patch embedding (first stage) and output projection (last stage)
    3. Pipeline-parallel weight loading with block index remapping

    Models using this mixin should call these helpers from their forward() method
    instead of inlining the PipeFusion logic.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        unpatchify = getattr(cls, "_unpatchify", None)
        if not callable(unpatchify):
            raise TypeError(f"{cls.__name__} must define _unpatchify() to use PipeFusionTransformerMixin")

        if is_pipefusion_initialized():

            @wraps(unpatchify)
            def wrapped_unpatchify(self, hidden_states: torch.Tensor, dims: tuple[int, int, int, int, int]):
                if not get_pipefusion_runtime().patch_mode:
                    return unpatchify(self, hidden_states, dims)
                return self.pipefusion_unpatchify(hidden_states, dims)

            cls._unpatchify = wrapped_unpatchify

    @abstractmethod
    def _unpatchify(self, hidden_states: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        """REQUIRED: Implementation that reshapes tokens back to image/video space."""
        pass

    @staticmethod
    def pipefusion_get_post_patch_height(height: int, patch_height: int) -> int:
        """
        Get the post-patch height for the current pipeline stage.

        In height-split mode, returns the height of the current patch;
        in temporal-split mode or non-patch mode, returns the full post-patch height.
        """
        runtime = get_pipefusion_runtime()
        if runtime.split_dim == "height":
            base_size = height // patch_height
            num_pipeline_patch = runtime.num_pipeline_patch
            patch_idx = runtime.patch_idx_tensor
            p_size = base_size // num_pipeline_patch
            start = patch_idx * p_size
            is_last = patch_idx == num_pipeline_patch - 1
            return p_size + is_last * (base_size - start - p_size)
        return height // patch_height

    @staticmethod
    def pipefusion_get_post_patch_num_frames(num_frames: int, patch_frames: int) -> int:
        """
        Get the post-patch frame count for the current pipeline stage.

        In temporal-split mode, returns the frame count of the current patch;
        in height-split mode or non-patch mode, returns the full post-patch frame count.
        """
        runtime = get_pipefusion_runtime()
        if runtime.split_dim == "temporal":
            base_size = num_frames // patch_frames
            num_pipeline_patch = runtime.num_pipeline_patch
            patch_idx = runtime.patch_idx_tensor
            p_size = base_size // num_pipeline_patch
            start = patch_idx * p_size
            is_last = patch_idx == num_pipeline_patch - 1
            return p_size + is_last * (base_size - start - p_size)
        return num_frames // patch_frames

    def pipefusion_unpatchify(self, hidden_states: torch.Tensor, dims: tuple[int, int, int, int, int]) -> torch.Tensor:
        batch_size, _, num_frames, height, width = dims
        p_t, p_h, p_w = self.config.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            self.pipefusion_get_post_patch_num_frames(num_frames, p_t),
            self.pipefusion_get_post_patch_height(height, p_h),
            width // p_w,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
