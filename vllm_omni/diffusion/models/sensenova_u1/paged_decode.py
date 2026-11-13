# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paged KV cache for SenseNova-U1 autoregressive decode, and a CUDA graph over it.

Autoregressive decode dominates a think request -- 80% of the wall clock, at
about 20 ms per token for 42 layers -- and almost all of the GPU idle in that
window is Python/ATen dispatch rather than any CUDA call. A CUDA graph retires
those dispatches, but capture needs static shapes, and the default cache grows
K/V with ``torch.cat`` on every step.

Padding the cache to a bucket and masking the tail does give static shapes, but
the mask is what costs: measured on an A800 at real decode shapes, a masked
bucket attention runs 7.96-11.55 ms per step against 1.03 ms for an exact-length
unmasked one, which is more than the graph saves.

The way out is a paged cache. ``flash_attn_varlen_func`` takes the used length
as a *tensor* (``seqused_k``) alongside a ``block_table``, so the buffers stay
bucket-sized and capturable while the kernel reads only the valid prefix. One
captured graph then serves every length in the bucket.

Scope, and when to delete this. The cache is model-local on purpose:
``DiffusionKVCacheManager`` reserves once per scheduler request, and
``ARDiffusionModelRunner`` lives under ``vllm_omni/experimental``. SenseNova runs
its whole autoregressive loop inside one pipeline ``forward``, so the decode
steps never surface to the scheduler and it cannot admit, pool, evict, reuse a
prefix for, or continuously batch them. What is here is therefore one set of
buffers behind an identity block table, reused by whichever request fits them,
not general paged-KV support: it holds while the pipeline serves one sequence
per forward, and is released when sleep discards the memory it captured
against. Once the decode loop is
driven by a scheduler-visible runner, or the pipeline handles more than one
sequence per forward, delete this file and use the manager instead.
"""

from __future__ import annotations

import inspect

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

BLOCK_SIZE = 16
# Growth schedule. Each new bucket costs one capture, so keep the list short and
# let the last entry cover the tail rather than doubling forever.
BUCKETS = (512, 1024, 2048, 4096, 8192)
# Past the last bucket the schedule grows by this much at a time. A request
# decodes at most `max_think_tokens` (1024) or `max_tokens` (512) steps, so a
# step this size costs at most one reallocation per request. Doubling the last
# bucket instead would also cost at most one, but a 9,100-token request would
# then reserve 766 MiB it never uses, against 118 MiB here.
TAIL_STEP = 2048


# Every argument to the kernel is passed by keyword, so the probe covers all of
# them. An older wheel can export `flash_attn_varlen_func` without the paged
# ones, or under different names, and a version string would not say which;
# probe the signature and fall back to SDPA when any is missing.
_REQUIRED_KWARGS = (
    "max_seqlen_q",
    "cu_seqlens_q",
    "max_seqlen_k",
    "seqused_k",
    "block_table",
    "softmax_scale",
    "causal",
)


def _flash_varlen():
    """Import lazily: the bundled kernel is CUDA-only and optional."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func


def _accepts_paged_kwargs(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - no introspectable signature
        return False
    return all(name in params for name in _REQUIRED_KWARGS)


def paged_decode_supported(device: torch.device, head_dim: int) -> bool:
    if device.type != "cuda":
        return False
    if head_dim % 8 or head_dim > 256:
        return False
    try:
        fn = _flash_varlen()
    except Exception as exc:  # pragma: no cover - depends on the wheel
        logger.debug("Paged decode unavailable: %s", exc)
        return False
    if not _accepts_paged_kwargs(fn):
        logger.debug("Paged decode unavailable: flash_attn_varlen_func lacks %s", _REQUIRED_KWARGS)
        return False
    return True


def dynamic_lora_wrappers_present(module) -> bool:
    """True once ``DiffusionLoRAManager`` has wrapped layers under ``module``.

    The manager replaces linear layers with ``BaseLayerWithLoRA`` and then
    binds, rescales or resets slot 0 per request; the wrappers stay in the tree
    afterwards. Capture records the module tree and the branches that ran during
    it, so a graph taken before an adapter was bound would replay without the
    adapter's matmuls and one taken with it bound would survive its removal --
    both silently, and neither visible in the shape, dtype and bucket that decide
    reuse. Nothing captured may outlive a request once these exist.

    The distilled LoRA is not affected: it is fused one-way into the weights and
    leaves no wrapper behind.
    """
    try:
        from vllm.lora.layers import BaseLayerWithLoRA
    except ImportError:  # pragma: no cover - depends on the wheel
        return False
    return any(isinstance(m, BaseLayerWithLoRA) for m in module.modules())


def _bucket_for(length: int) -> int:
    for b in BUCKETS:
        if length <= b:
            return b
    # Past the schedule, grow in TAIL_STEP-sized steps. Rounding to a whole
    # number of blocks here would reallocate and re-capture every BLOCK_SIZE
    # tokens: a think edit with two 2048x2048 inputs runs to about 9,100 tokens
    # and paid 41 reallocations and 42 captures in one request.
    over = length - BUCKETS[-1]
    return BUCKETS[-1] + ((over + TAIL_STEP - 1) // TAIL_STEP) * TAIL_STEP


class PagedDecodeCache:
    """Per-layer paged K/V for single-token decode.

    Layout follows what ``flash_attn_varlen_func`` accepts with a block table:
    ``(num_blocks, BLOCK_SIZE, kv_heads, head_dim)``. ``seqused`` holds the
    number of valid tokens and is the only thing that changes between decode
    steps, which is what lets a captured graph be replayed unchanged.
    """

    def __init__(self, num_layers, kv_heads, head_dim, length, device, dtype):
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.bucket = _bucket_for(length)
        nblocks = self.bucket // BLOCK_SIZE
        shape = (nblocks, BLOCK_SIZE, kv_heads, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.block_table = torch.arange(nblocks, device=device, dtype=torch.int32).unsqueeze(0)
        self.cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.seqused = torch.zeros(1, device=device, dtype=torch.int32)
        # The write slot has to live in a tensor, not a Python int. Under graph
        # capture a Python index is baked in at capture time, so every replay
        # would overwrite the same slot -- correct eagerly, silently wrong once
        # captured.
        self.pos = torch.zeros(1, device=device, dtype=torch.int64)
        self._length = 0
        # Bumped whenever the buffers are reallocated, so a graph runner can
        # tell that its capture no longer points at live memory.
        self.generation = 0

    @property
    def length(self) -> int:
        return self._length

    def set_length(self, n: int) -> None:
        self._length = n
        self.seqused.fill_(n)
        self.pos.fill_(n - 1)

    def reusable_for(self, num_layers, kv_heads, head_dim, dtype, length) -> bool:
        """Can this cache serve another request without reallocating?

        Reusing the buffers is what lets captured graphs survive across
        requests: a capture binds the addresses it recorded, so a fresh cache
        forces a fresh capture.
        """
        return (
            len(self.k) == num_layers
            and self.kv_heads == kv_heads
            and self.head_dim == head_dim
            and self.dtype == dtype
            and _bucket_for(length) <= self.bucket
        )

    def load_prefix(self, cache) -> bool:
        """Copy a prefill cache into the paged buffers.

        Only the first ``seqused`` rows are ever read, so the tail is left as
        whatever the previous request wrote. Returns False for an empty cache.
        """
        layer0 = cache.layers[0]
        if layer0.keys is None:
            return False
        prefix = layer0.keys.shape[2]
        for i in range(len(self.k)):
            keys, values = cache.layers[i].keys, cache.layers[i].values
            # [B, H, S, D] -> [S, H, D], then into the paged blocks
            self._write_prefix(i, keys[0].transpose(0, 1), values[0].transpose(0, 1))
        self.set_length(prefix)
        return True

    @classmethod
    def from_dynamic_cache(cls, cache, num_layers, device, dtype, min_length=0):
        """Allocate buffers for a prefill cache and copy it in."""
        layer0 = cache.layers[0]
        if layer0.keys is None:
            return None
        _, kv_heads, prefix, head_dim = layer0.keys.shape
        obj = cls(num_layers, kv_heads, head_dim, max(prefix + 1, min_length), device, dtype)
        obj.load_prefix(cache)
        return obj

    def _write_prefix(self, layer_idx, keys_shd, values_shd):
        n = keys_shd.shape[0]
        flat_k = self.k[layer_idx].view(-1, self.kv_heads, self.head_dim)
        flat_v = self.v[layer_idx].view(-1, self.kv_heads, self.head_dim)
        flat_k[:n].copy_(keys_shd)
        flat_v[:n].copy_(values_shd)

    def grow(self, length: int) -> bool:
        """Move to the next bucket, keeping the tokens already stored.

        Everything is built into locals first and published in one step. A
        resize that failed part way -- an allocation for a later layer, say --
        would otherwise leave new K beside old V under the ``generation`` a
        captured graph is keyed on, so the next request would keep selecting
        that graph and replay it over storage that had just been freed.
        """
        new_bucket = _bucket_for(length)
        if new_bucket == self.bucket:
            return False
        nblocks = new_bucket // BLOCK_SIZE
        shape = (nblocks, BLOCK_SIZE, self.kv_heads, self.head_dim)
        old_len = self._length
        new_k, new_v = [], []
        for old_k, old_v in zip(self.k, self.v):
            k = torch.zeros(shape, device=self.device, dtype=self.dtype)
            v = torch.zeros(shape, device=self.device, dtype=self.dtype)
            k.view(-1, self.kv_heads, self.head_dim)[:old_len].copy_(
                old_k.view(-1, self.kv_heads, self.head_dim)[:old_len]
            )
            v.view(-1, self.kv_heads, self.head_dim)[:old_len].copy_(
                old_v.view(-1, self.kv_heads, self.head_dim)[:old_len]
            )
            new_k.append(k)
            new_v.append(v)
        block_table = torch.arange(nblocks, device=self.device, dtype=torch.int32).unsqueeze(0)

        # Publish. New buffers mean any graph captured against the old ones is
        # stale, so the generation moves in the same step that hands them over.
        self.k, self.v, self.block_table, self.bucket = new_k, new_v, block_table, new_bucket
        self.generation += 1
        self.set_length(old_len)
        logger.debug("Paged decode grew to bucket %d at length %d (generation %d)", new_bucket, length, self.generation)
        return True

    def attend(self, layer_idx, query_bhsd, key_bhsd, value_bhsd, softmax_scale):
        """Append this step's K/V and attend over the whole valid prefix.

        Writes at ``length - 1`` because the caller has already advanced the
        length for this step. Both the slot and the attended length come from
        device tensors, so a captured graph reads whatever they hold at replay
        rather than whatever they held at capture.
        """
        flat_k = self.k[layer_idx].view(-1, self.kv_heads, self.head_dim)
        flat_v = self.v[layer_idx].view(-1, self.kv_heads, self.head_dim)
        flat_k.index_copy_(0, self.pos, key_bhsd[0, :, 0].unsqueeze(0))
        flat_v.index_copy_(0, self.pos, value_bhsd[0, :, 0].unsqueeze(0))
        heads, head_dim = query_bhsd.shape[1], query_bhsd.shape[3]
        q = query_bhsd[0].transpose(0, 1).reshape(1, heads, head_dim)
        out = _flash_varlen()(
            q,
            self.k[layer_idx],
            self.v[layer_idx],
            max_seqlen_q=1,
            cu_seqlens_q=self.cu_seqlens_q,
            max_seqlen_k=self.bucket,
            seqused_k=self.seqused,
            block_table=self.block_table,
            softmax_scale=softmax_scale,
            causal=False,
        )
        return out.reshape(1, 1, heads, head_dim)

    def to_dynamic_cache(self, cache) -> None:
        """Write the decoded K/V back so the generation stage sees one cache.

        Only decode is paged; the DiT stage that follows reads the ordinary
        cache, so the two are reconciled once at the hand-off rather than kept
        in sync on every step.
        """
        n = self._length
        for i, layer in enumerate(cache.layers):
            keys = self.k[i].view(-1, self.kv_heads, self.head_dim)[:n]
            values = self.v[i].view(-1, self.kv_heads, self.head_dim)[:n]
            layer.keys = keys.transpose(0, 1).unsqueeze(0).contiguous()
            layer.values = values.transpose(0, 1).unsqueeze(0).contiguous()


class DecodeGraphRunner:
    """One captured decode step per KV bucket, replayed for every token.

    The whole point of the paged cache is that everything the step reads which
    varies -- the token, its position, and how much of the cache is live -- sits
    in device tensors. So a single capture serves every step in a bucket: fill
    the tensors, replay, read the logits out of the static output.

    A capture is invalidated when the cache reallocates (tracked by
    ``PagedDecodeCache.generation``), which happens once per bucket boundary.
    """

    def __init__(self, language_model, cache, device):
        self.lm = language_model
        self.cache = cache
        self.device = device
        self.input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.indexes = torch.zeros(3, 1, dtype=torch.long, device=device)
        self._graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.captures = 0

    def _forward(self):
        return self.lm(
            input_ids=self.input_ids,
            indexes=self.indexes,
            past_key_values=None,
            use_cache=False,
            paged_cache=self.cache,
        ).logits

    def _capture(self, key):
        # Warm up on a side stream first so cuBLAS workspaces and any lazy
        # allocation happen outside the capture. The warm-up recomputes this
        # same step, writing the same K/V to the same slot, so it is idempotent.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        # A private pool would hand every runner its own arena and never give it
        # back, and a runner is rebuilt whenever the cache reallocates or a LoRA
        # wrapper appears: measured, that grew device memory by about 40 MiB per
        # request and never levelled off. The platform pool is shared, which is
        # what every other captured path in the tree uses.
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            logits = self._forward()
        self.captures += 1
        logger.debug("Captured decode graph for bucket=%d generation=%d", self.cache.bucket, self.cache.generation)
        self._graphs[key] = (graph, logits)
        return self._graphs[key]

    def step(self, token, t_index):
        """Run one decode step. Returns the static logits tensor."""
        self.input_ids[0, 0] = token
        self.indexes[0, 0] = t_index
        key = (self.cache.bucket, self.cache.generation)
        entry = self._graphs.get(key) or self._capture(key)
        entry[0].replay()
        return entry[1]
