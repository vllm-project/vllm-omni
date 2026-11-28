# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image 2.1 dense editing-prefix layout for two Ulysses ranks."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

_ROUTE_MARKER = "_qwen21_sharded_prefix"


@dataclass
class PrefixLayout:
    group: object
    rank: int
    prefix: int  # Actual full prefix length, also meaningful in decode.
    target: int  # Local target rows, including baseline's possible SP tail padding.
    decode: bool
    layers: int = 0
    batch_size: int = 1

    @property
    def local_prefix(self):
        return 0 if self.decode else (self.prefix + 1) // 2

    def shard_prefix(self, tensor, dim):
        """SP rows laid out as [P_rank, T_rank]. A synthetic last P row stays off attention."""
        count = self.local_prefix
        start = self.rank * count
        part = tensor.narrow(dim, start, min(count, tensor.shape[dim] - start))
        if part.shape[dim] < count:
            # Copy a valid RoPE row as well. This temporary row is removed before softmax.
            part = torch.cat((part, tensor.narrow(dim, 0, 1)), dim=dim)
        return part

    def sequence_to_heads(self, tensor):
        # HCCL and Gloo both support equal-split all_to_all_single on flat buffers.
        batch, local, heads, dim = tensor.shape
        if heads % 2 or local != self.local_prefix + self.target:
            raise ValueError(f"Unexpected local Qwen attention shape {tensor.shape} for {self}")
        send = tensor.reshape(batch, local, 2, heads // 2, dim).permute(2, 0, 1, 3, 4).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received.view(-1), send.view(-1), group=self.group)
        rows = received.transpose(0, 1)
        if self.decode or self.prefix == 0:
            return rows.reshape(batch, 2 * local, heads // 2, dim)
        p = self.local_prefix
        prefix = rows[:, :, :p].reshape(batch, 2 * p, heads // 2, dim)[:, : self.prefix]
        target = rows[:, :, p:].reshape(batch, 2 * self.target, heads // 2, dim)
        return torch.cat((prefix, target), dim=1)

    def heads_to_sequence(self, tensor):
        batch, _, heads, dim = tensor.shape
        p = self.local_prefix
        if p:
            prefix, target = tensor.split((self.prefix, 2 * self.target), dim=1)
            if self.prefix % 2:
                prefix = torch.cat((prefix, torch.zeros_like(prefix[:, :1])), dim=1)
            pack = torch.cat(
                (prefix.reshape(batch, 2, p, heads, dim), target.reshape(batch, 2, self.target, heads, dim)), dim=2
            )
        else:
            pack = tensor.reshape(batch, 2, self.target, heads, dim)
        send = pack.transpose(0, 1).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received.view(-1), send.view(-1), group=self.group)
        return received.permute(1, 2, 0, 3, 4).reshape(batch, self.target + p, heads * 2, dim)

    def attention(self, layer, query, key, value, metadata, cache, branch):
        q, k, v = (self.sequence_to_heads(tensor) for tensor in (query, key, value))
        entry = cache.setdefault(branch, {})
        if self.decode:
            expected = (q.shape[0], self.prefix, q.shape[2], q.shape[3])
            if entry["key"].shape != expected or entry["value"].shape != expected:
                raise ValueError(f"Expected rank-owned Qwen prefix K/V {expected}")
            k = torch.cat((entry["key"], k), dim=1)
            v = torch.cat((entry["value"], v), dim=1)
        else:
            entry["key"] = k[:, : self.prefix].clone()
            entry["value"] = v[:, : self.prefix].clone()
        # Same backend, mask and dtype as the original strategy after sequence redistribution.
        # The generic joint strategy would select heads again from our already-sharded prefix.
        attended = layer._run_local_attention(q, k, v, layer._with_kv_cache_dtype(metadata))
        result = self.heads_to_sequence(attended)
        self.layers += 1
        return result

    def complete(self, cache, branch):
        shape = (self.batch_size, self.prefix, 16, 128)
        if (
            self.layers != 32
            or len(cache) != 32
            or any(c[branch][x].shape != shape for c in cache for x in ("key", "value"))
        ):
            raise RuntimeError(f"Incomplete Qwen head-sharded cache, expected 32 layers × {shape}")
        if not self.decode:
            # The pipeline slices every tensor along dim 0 and reassembles it
            # on regroup. This one small part carries ownership even in a fresh
            # list after scatter/merge. Never overwrite a native cache.
            cache[0][branch][_ROUTE_MARKER] = torch.ones((self.batch_size, 1), dtype=torch.bool, device="cpu")


def cache_route_key(cache):
    return bool(cache and _ROUTE_MARKER in cache[0].get("cond", {}))


def _eligible(model, hidden, prefix, target_rows, cache, attn_path):
    par = model.parallel_config
    if (
        getattr(model, "do_true_cfg", True)
        or prefix <= 0
        or target_rows <= 0
        or target_rows % 2
        or attn_path not in ("auto", "mask")
        or par.tensor_parallel_size != 1
        or getattr(par, "cfg_parallel_size", 1) != 1
        or getattr(par, "use_hsdp", False)
        or not model._enforce_eager
        or model._decode_graph_manager is not None
        or model.quant_config is not None
        or model.prefix_kv_cache_dtype is not None
        or not model.causal_condition
        or not model.causal_block
        or len(cache) != 32
        or len(model.transformer_blocks) != 32
        or hidden.dtype != torch.bfloat16
        or torch.is_grad_enabled()
    ):
        return False
    for block in model.transformer_blocks:
        op = block.attn.attn
        if (
            block.attn.num_heads != 32
            or block.attn.num_kv_heads != 32
            or block.attn.head_dim != 128
            or op.is_paged_kv_active()
            or op._has_custom_attention
            or op._schedule_configured
            or (getattr(op, "_scheduler_paged_kv", False) and op.paged_kv_cache_role is not None)
            or getattr(op, "_kv_cache_dtype", None) not in (None, "auto", "float")
            or op.attn_backend is None
            or op.attn_backend.get_name() != "SDPA"
        ):
            return False
    return True


def select_group(model, hidden, encoder, target_rows, cache, branch, attn_path, is_decode):
    """Select the prefill path jointly, then follow the layout stored on the KV."""
    if not getattr(model, "shard_edit_prefix", False):
        return None
    sharded = cache_route_key(cache)
    if sharded and (not is_decode or cache[0]["cond"][_ROUTE_MARKER].shape != (hidden.shape[0], 1)):
        raise RuntimeError("Incomplete or misaligned Qwen SP cache; restart this generation with a fresh cache")
    if not cache or (is_decode and not sharded):
        return None
    if branch != "cond":
        if sharded:
            raise RuntimeError("A conditional-only sharded cache cannot be reused for true CFG")
        return None
    par = model.parallel_config
    pure_sp2 = (
        par is not None
        and par.sequence_parallel_size == 2
        and par.ulysses_degree == 2
        and par.ring_degree == 1
        and par.allgather_degree == 1
    )
    if hidden.device.type != "npu" or not pure_sp2 or torch.compiler.is_compiling():
        if sharded:
            raise RuntimeError("Cannot consume head-sharded prefix KV outside its eager NPU Ulysses group")
        return None
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    sp = get_sp_group()
    if (sp.world_size, sp.ulysses_world_size, sp.ring_world_size, sp.allgather_world_size) != (2, 2, 1, 1):
        raise RuntimeError("Qwen SP configuration disagrees with the initialized process groups")
    if is_decode:
        if getattr(model, "do_true_cfg", True) or model._decode_graph_manager is not None:
            raise RuntimeError("Cannot change CFG/graph mode after head-sharded prefix prefill")
        return sp

    # E includes one VLM slot per four reference latent tokens. Expanding
    # those slots adds three rows each; no tensor values or NPU sync needed.
    ref_rows = hidden.shape[1] - target_rows
    prefix = -1
    if encoder is not None and len(encoder.shape) == 3 and ref_rows >= 0 and ref_rows % 4 == 0:
        prefix = encoder.shape[1] + 3 * (ref_rows // 4)
    eligible = ref_rows > 0 and _eligible(model, hidden, prefix, target_rows, cache, attn_path)
    vote = torch.tensor([eligible, hidden.shape[0], prefix, target_rows], dtype=torch.int64, device="cpu")
    votes = [torch.empty_like(vote) for _ in range(2)]
    dist.all_gather(votes, vote, group=sp.cpu_group)
    if not torch.equal(votes[0][1:], votes[1][1:]):
        raise RuntimeError("Qwen SP ranks disagree on request geometry")
    if not all(row[0].item() for row in votes):
        return None
    # An interrupted prefill must never be consumed as a native cache.
    cache[0].setdefault(branch, {})[_ROUTE_MARKER] = torch.empty((hidden.shape[0], 0), device="cpu", dtype=torch.bool)
    return sp


def make_layout(sp, target, full_target_rows, prefix, is_decode, cache, branch):
    if target.shape[1] * 2 != full_target_rows or target.dtype != torch.bfloat16:
        raise RuntimeError("The selected Qwen SP route received an unexpected local target shape or dtype")
    if is_decode:
        prefix = cache[0][branch]["key"].shape[1]
    return PrefixLayout(
        sp.ulysses_group, sp.ulysses_rank, prefix, target.shape[1], is_decode, batch_size=target.shape[0]
    )
