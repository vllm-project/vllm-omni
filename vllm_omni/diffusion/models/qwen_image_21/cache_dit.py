# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode-only Cache-DiT adapters preserving Qwen 2.1's per-layer prefix KV."""

import weakref

import cache_dit
import torch
from cache_dit import BlockAdapter, ForwardPattern
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.cachedit import CacheDiTConfig, CacheDiTEnableResult

logger = init_logger(__name__)


class _Layer(nn.Module):
    def __init__(self, block, index):
        super().__init__()
        self._block_ref = weakref.ref(block)
        self.index = index

    def forward(self, hidden_states: torch.Tensor, *, kv_cache, **kwargs) -> torch.Tensor:
        block = self._block_ref()
        if block is None:
            raise RuntimeError("Qwen-Image 2.1 transformer was released")
        return block(hidden_states=hidden_states, kv_cache=kv_cache[self.index], **kwargs)


class _DecodeProxy(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([_Layer(block, i) for i, block in enumerate(blocks)])

    def forward(self, hidden_states, **kwargs):
        for block in self.blocks:
            hidden_states = block(hidden_states, **kwargs)
        return hidden_states


def enable_qwen_image21_cache(pipeline, cache_config):
    """Install branch-isolated caches; prefill always executes the original blocks."""
    transformer = pipeline.transformer
    if not transformer.causal_condition:
        raise ValueError("Qwen-Image 2.1 Cache-DiT requires causal prefix KV caching")
    projected = CacheDiTConfig.from_diffusion_config(cache_config)
    if projected.scm_steps_mask_policy is not None:
        raise ValueError("Qwen-Image 2.1 Cache-DiT does not yet support SCM masks")
    proxies, adapters = {}, []
    for branch in ("cond", "uncond"):
        proxy = _DecodeProxy(transformer.transformer_blocks)
        adapter = BlockAdapter(
            transformer=proxy,
            blocks=[proxy.blocks],
            forward_pattern=[ForwardPattern.Pattern_3],
            has_separate_cfg=False,
        )
        pipe_cls = type(adapter.pipe)
        if "_is_cached" in vars(pipe_cls):
            del pipe_cls._is_cached
        config = projected.to_db_cache_config()
        config.enable_separate_cfg = False
        cache_dit.enable_cache(adapter, cache_config=config, calibrator_config=projected.to_calibrator_config())
        proxies[branch] = proxy
        adapters.append(adapter)
    # Keep adapters outside the registered module tree: weights remain owned by
    # the original transformer blocks, and checkpoint names do not change.
    transformer._cache_dit_decode_proxies = proxies
    if transformer._decode_graph_manager is not None:
        transformer._decode_graph_manager.clear()
        transformer._decode_graph_manager = None
    transformer.enable_cuda_graph_decode = False
    logger.info("Qwen-Image 2.1 Cache-DiT: decode-only, isolated CFG branches; whole-decode CUDA Graph disabled")

    def refresh(_pipeline, num_inference_steps, verbose=True):
        for proxy in proxies.values():
            config = projected.to_db_cache_config()
            config.num_inference_steps = max(num_inference_steps - 1, 1)
            config.enable_separate_cfg = False
            cache_dit.refresh_context(proxy, cache_config=config, verbose=verbose)

    return CacheDiTEnableResult(refresh=refresh, targets=tuple(adapters))
