# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState


@dataclass
class ARDiffusionKVLayerContext:
    """Assemble exact history/media/conditioning after shared attention resharding.

    Segmented state owns reusable history staging. Ordinary offline callers may
    supply contiguous storage with the same layout. Publication stays with the
    caller's all-layer transaction; this context never commits session state.
    """

    layer_idx: int
    history_tokens: int
    media_tokens: int
    cache_conditioning: bool
    update_cache: bool
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    state: ARDiffusionSegmentedKVState | None = None

    def assemble(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
            raise ValueError("AR KV assembly requires matching single-session BSND tensors")
        if self.media_tokens <= 0 or self.media_tokens > key.shape[1]:
            raise ValueError("AR KV media length is outside current Q/K/V")
        media_key, media_value = key[0, : self.media_tokens], value[0, : self.media_tokens]
        condition_key, condition_value = key[0, self.media_tokens :], value[0, self.media_tokens :]
        if self.cache_conditioning:
            media_key, media_value = key[0], value[0]
            condition_key, condition_value = key[0, :0], value[0, :0]
        elif not self.update_cache:
            condition_key, condition_value = key[0, :0], value[0, :0]
        if self.state is not None:
            if self.history_tokens != self.state.history_tokens:
                raise ValueError("AR KV assembly history length disagrees with its segmented owner")
            full_key, full_value = self.state.assemble(
                self.layer_idx, condition_key, condition_value, media_key, media_value
            )
            return full_key[None], full_value[None]
        end = self.history_tokens + media_key.shape[0]
        if (
            self.key_cache.ndim != 4
            or self.key_cache.shape[0] != 1
            or self.history_tokens < 0
            or end > self.key_cache.shape[1]
            or self.key_cache.shape != self.value_cache.shape
        ):
            raise ValueError("AR KV write exceeds contiguous storage capacity")
        for target, source in ((self.key_cache, media_key), (self.value_cache, media_value)):
            if target.device != source.device or target.dtype != source.dtype or target.shape[2:] != source.shape[1:]:
                raise ValueError("AR KV storage must match rank-local dtype, device and heads")
        self.key_cache[:, self.history_tokens : end].copy_(media_key)
        self.value_cache[:, self.history_tokens : end].copy_(media_value)
        full_key, full_value = self.key_cache[:, :end], self.value_cache[:, :end]
        if condition_key.numel():
            full_key = torch.cat((full_key, condition_key[None]), 1)
            full_value = torch.cat((full_value, condition_value[None]), 1)
        return full_key, full_value
