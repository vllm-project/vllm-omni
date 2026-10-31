# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed request-batched YuE2 NAR with independent varlen attention spans."""

from itertools import accumulate
from numbers import Integral

import torch

from vllm_omni.model_executor.models.yue2.yue2 import CachedNAR, _attention, _nar_kernels


class BatchedCachedNAR(CachedNAR):
    """Batch GEMMs over real tokens; cu_seqlens isolate each song's attention.

    There is no sequence padding. The two boundary tokens, rotary positions,
    condition KV and CPU noise remain request-local. Batch one is dispatched
    to the original CachedNAR by the caller.
    """

    def __init__(self, model, chunks, pool=None):
        if len(chunks) < 2:
            raise ValueError("Batched NAR requires batch >= 2")
        self.model, self.chunks, self.pool = model, chunks, pool
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        self.lengths = [len(chunk.noise) for chunk in chunks]
        self.query_offsets = list(accumulate([0, *(n + 2 for n in self.lengths)]))
        self.key_offsets = list(accumulate([0, *(len(c.ar_tokens) + len(c.noise) + 2 for c in chunks)]))
        self.max_query_length = max(n + 2 for n in self.lengths)
        self.max_key_length = max(b - a for a, b in zip(self.key_offsets, self.key_offsets[1:]))
        self.nar_length = self.query_offsets[-1]
        config = model.config
        shape = (
            int(config.num_hidden_layers),
            self.key_offsets[-1],
            int(config.num_key_value_heads),
            int(config.head_dim),
        )
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty_like(self.keys)
        self.graph = None
        self.last_use = None

    def prefill_iter(self):
        members = []
        for chunk, start, end in zip(self.chunks, self.key_offsets, self.key_offsets[1:]):
            # The packed engine owns storage through the solve/cancellation event.
            member = CachedNAR(self.model, chunk, self.pool, kv=(self.keys[:, start:end], self.values[:, start:end]))
            members.append(member)
            yield
        nar_indices, frame_indices = [], []
        for member, query_start, key_start in zip(members, self.query_offsets, self.key_offsets):
            nar_indices.extend(range(key_start + member.ar_length, key_start + member.ar_length + member.nar_length))
            frame_indices.extend(range(query_start + 1, query_start + member.nar_length - 1))
        self.nar_indices = torch.tensor(nar_indices, device=self.device)
        self.frame_indices = torch.tensor(frame_indices, device=self.device)
        self.cos = torch.cat([member.cos for member in members])
        self.sin = torch.cat([member.sin for member in members])
        self.pos_emb = torch.cat([member.pos_emb[0] for member in members])
        self.bounds_q = torch.tensor(self.query_offsets, dtype=torch.int32, device=self.device)
        self.bounds_k = torch.tensor(self.key_offsets, dtype=torch.int32, device=self.device)
        self.batch_flash = None
        self.flash_version = 2
        if self.device.type == "cuda":
            from vllm.vllm_flash_attn import flash_attn_interface, flash_attn_varlen_func

            self.flash_version = 3 if members[0].fa3 is not None else 2
            if flash_attn_interface.is_fa_version_supported(self.flash_version):
                self.batch_flash = flash_attn_varlen_func

    def _velocity(self, state, raw_t):
        model, config = self.model, self.model.config
        pre, post = _nar_kernels()
        eps = float(config.rms_norm_eps)
        heads, kv_heads = int(config.num_attention_heads), int(config.num_key_value_heads)
        x_nar = state.new_zeros((self.nar_length, state.shape[-1]))
        x_nar.index_copy_(0, self.frame_indices, state)
        t = torch.sigmoid(raw_t)
        shift = model._timestep_shift
        shifted = shift * t / (1 + (shift - 1) * t)
        x = model.vae2llm(x_nar)
        x = x + model.time_embedder(shifted.expand(self.nar_length))
        x = x + self.pos_emb
        for index, weights in enumerate(model.nar_weights()):
            w_norm, w_qkv, w_q_norm, w_k_norm, w_o, w_mlp_norm, w_gate_up, w_down = weights
            q, k, v = pre(x, w_norm, w_qkv, w_q_norm, w_k_norm, self.cos, self.sin, eps, heads, kv_heads)
            self.keys[index].index_copy_(0, self.nar_indices, k)
            self.values[index].index_copy_(0, self.nar_indices, v)
            if self.batch_flash is not None:
                h = self.batch_flash(
                    q,
                    self.keys[index],
                    self.values[index],
                    cu_seqlens_q=self.bounds_q,
                    cu_seqlens_k=self.bounds_k,
                    max_seqlen_q=self.max_query_length,
                    max_seqlen_k=self.max_key_length,
                    causal=False,
                    fa_version=self.flash_version,
                )
            else:
                # CPU/reference fallback also avoids a padded math-attention
                # batch. The large linear projections above remain batched.
                h = torch.cat(
                    [
                        _attention(q[qa:qb], self.keys[index, ka:kb], self.values[index, ka:kb], causal=False)
                        for qa, qb, ka, kb in zip(
                            self.query_offsets[:-1], self.query_offsets[1:], self.key_offsets[:-1], self.key_offsets[1:]
                        )
                    ]
                )
            x = post(x, h, w_o, w_mlp_norm, w_gate_up, w_down, eps)
        return model.llm2vae(model.model.norm(x[None]))[0].index_select(0, self.frame_indices)

    def solve_iter(self, steps=32):
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        # Draws have already happened independently on CPU in song_chunks.
        state = torch.cat([chunk.noise for chunk in self.chunks]).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            yield
            mid = state - first * (dt / 2)
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
            yield
        return state.float()

    def close(self):
        super().close()
        self.chunks = None
        self.nar_indices = self.frame_indices = None
