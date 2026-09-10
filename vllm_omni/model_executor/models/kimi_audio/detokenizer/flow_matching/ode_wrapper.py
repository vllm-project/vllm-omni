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
        self.x_cond = self.position_ids = self.seq_len = None
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

        position_ids_cur = [i for i in range(start_position_id, self.x_cond.shape[1] + start_position_id)]
        position_ids = torch.tensor([position_ids_cur])
        self.position_ids = position_ids.to(self.x_cond.device).long()
        self.seq_len = torch.Tensor([position_ids.shape[1]]).to(self.x_cond.device).long()

        cu_seqlens = torch.cumsum(self.seq_len, dim=0)
        self.cu_seqlens = torch.cat([torch.Tensor([0]).to(cu_seqlens.device), cu_seqlens], dim=0).int()
        self.cu_maxlen = self.seq_len.cpu().max()

        if self.cu_seqlens_k is None:
            self.cu_seqlens_k = self.cu_seqlens
            self.cu_maxlen_k = self.cu_maxlen
            previous_seqlen = self.seq_len
        else:
            previous_seqlen_old = cache["previous_seqlen"]
            previous_seqlen = previous_seqlen_old + self.seq_len
            cu_seqlens_k = torch.cumsum(previous_seqlen, dim=0)
            self.cu_seqlens_k = torch.cat([torch.Tensor([0]).to(cu_seqlens_k.device), cu_seqlens_k], dim=0).int()
            self.cu_maxlen_k = previous_seqlen.cpu().max()
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
                bsz = layer_cache["attn_kvcache"]["prev_k"].shape[0]
                self.previous_seqlen = (
                    torch.Tensor([layer_cache["attn_kvcache"]["prev_k"].shape[1] for i in range(bsz)])
                    .to(layer_cache["attn_kvcache"]["prev_k"].device)
                    .long()
                )
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
