# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from MoonshotAI/Kimi-Audio (MIT), revision
# 349251e1d8f4f98d58fda59246381faecd7392e0, kimia_infer/models/detokenizer/.
# See vllm_omni/model_executor/models/kimi_audio/NOTICE for the upstream license.

from functools import lru_cache

import torch
import torch.nn as nn


@lru_cache(maxsize=1)
def get_cached_zeros(numel, device="cpu", dtype=torch.float32):
    return torch.zeros(numel, device=device, dtype=dtype)


class StreamingODEWrapperForPrefix(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.clear_all_states()

    def clear_all_states(self):
        self.x_cond = self.position_ids = None
        self.incremental_state = {}
        self.kv_cache_tokens = 0
        self.cu_seqlens = None
        self.cu_maxlen = None
        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None

    def set_conditions(self, x_cond, start_position_id, cache=None):
        if cache is None:
            cache = {}
        self.x_cond = x_cond

        seq_len = self.x_cond.shape[1]
        self.position_ids = torch.arange(
            start_position_id, start_position_id + seq_len, device=self.x_cond.device, dtype=torch.long
        ).unsqueeze(0)
        self.cu_seqlens = torch.tensor([0, seq_len], device=self.x_cond.device, dtype=torch.int32)
        self.cu_maxlen = seq_len

        if self.cu_seqlens_k is None:
            self.cu_seqlens_k = self.cu_seqlens
            self.cu_maxlen_k = self.cu_maxlen
            previous_seqlen = seq_len
        else:
            previous_seqlen_old = cache["previous_seqlen"]
            previous_seqlen = previous_seqlen_old + seq_len
            self.cu_seqlens_k = torch.tensor([0, previous_seqlen], device=self.x_cond.device, dtype=torch.int32)
            self.cu_maxlen_k = previous_seqlen
        self.previous_seqlen = previous_seqlen
        return {"previous_seqlen": previous_seqlen}

    def update_incremental_state(self, max_kv_cache_tokens=900, condition_cache=None):
        if condition_cache is None:
            condition_cache = {}

        for layer_cache in self.incremental_state.values():
            layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"]["cur_k"]
            layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"]["cur_v"]
            self.kv_cache_tokens = layer_cache["attn_kvcache"]["prev_k"].shape[1]

            if self.kv_cache_tokens > max_kv_cache_tokens:
                # The main generation path reserves no reference-voice prefix;
                # preserve the official rolling window of acoustic history.
                layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"]["prev_k"][:, -max_kv_cache_tokens:]
                layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"]["prev_v"][:, -max_kv_cache_tokens:]
                self.previous_seqlen = layer_cache["attn_kvcache"]["prev_k"].shape[1]
                condition_cache["previous_seqlen"] = self.previous_seqlen
                self.kv_cache_tokens = layer_cache["attn_kvcache"]["prev_k"].shape[1]

            layer_cache["attn_kvcache"].pop("cur_k")
            layer_cache["attn_kvcache"].pop("cur_v")

    def forward(self, t, x, args=None):
        t = get_cached_zeros(x.shape[0], device=x.device, dtype=torch.long) + (t * 1000).long()
        return self.net(
            x=x,
            condition=self.x_cond,
            t=t,
            position_ids=self.position_ids,
            cu_seqlens=self.cu_seqlens,
            cu_maxlen=self.cu_maxlen,
            cu_seqlens_k=self.cu_seqlens_k,
            cu_maxlen_k=self.cu_maxlen_k,
            incremental_state=self.incremental_state,
            nopadding=True,
            mask=None,
            seq_len=None,
        )
