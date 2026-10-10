# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Wan attention contexts backed by the per-block hybrid ring."""

import sys
from contextlib import contextmanager

import torch
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyLayerContext

from .block_plan import BlockPlan, KVKey
from .hybrid_transport import LayerMajorPages


class HybridNoisyKVState:
    """Single-request executor adapter; owns only the hybrid ring, no native pool."""

    def __init__(self, spec, model, group, device, dtype):
        self.spec, self.model, self.group = spec, model, group
        self.device, self.dtype = device, dtype
        self._chunk_tokens = {}
        self._rank = 0
        self.bytes_sent = self.bytes_received = 0
        self.capacity = self.reserved_bytes = 0
        self.pages = None
        self.prepared = {}
        self.current = None

    def bind_rank(self, rank, pp_group):
        self._rank = rank

    def set_inflight(self, inflight):
        # Admission is single-request; all transfers come from its native plan.
        pass

    def begin_request(self, req, plan, *, chunk_tokens, t0=0):
        if self.pages is not None or t0 != 0:
            raise ValueError("hybrid benchmark supports one request at a time")
        spec = self.spec
        if chunk_tokens <= 0 or chunk_tokens % spec.block_size or chunk_tokens > spec.max_chunk_tokens:
            raise ValueError("chunk tokens do not fit the KV specification")
        self._chunk_tokens[req] = chunk_tokens
        self.plan = BlockPlan(plan, self.model.num_layers)
        groups, stages = plan.schedule.layer_groups, plan.schedule.stages
        self.pages = LayerMajorPages(self.plan, spec, chunk_tokens, self.device, self.dtype, self.group)
        push = self.pages
        self.capacity = push.capacity * stages
        self.reserved_bytes = (
            push.buffers.numel() * push.buffers.element_size() + push.flags.numel() * push.flags.element_size()
        )
        device = push.device
        for chunk in range(plan.schedule.chunks):
            step = self._rank // groups
            task = self.plan.task(chunk, step, self.model.start_layer)
            keys = self.plan.reads[task]
            ids = [k.chunk % push.capacity * stages + k.step for k in keys]
            write = chunk % push.capacity * stages + step
            ids.append(write)
            width = spec.max_history_chunks + 1

            def gpu(values, dtype):
                return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)

            table = gpu([ids + [0] * (width - len(ids))], torch.int32)
            seq_len = len(ids) * chunk_tokens
            seq_lens = gpu([seq_len], torch.int32)
            query_locs = gpu([0, chunk_tokens], torch.int32)
            slots = torch.arange(chunk_tokens, device=device, dtype=torch.long) + write * chunk_tokens
            contexts = []
            for local in range(self.model.local_num_layers):
                pools = [push.buffers[local, field].view(-1, spec.num_kv_heads, spec.head_size) for field in range(2)]
                assert all(value.is_contiguous() for value in pools)
                contexts.append(
                    NoisyLayerContext(
                        local,
                        pools[0],
                        pools[1],
                        chunk_tokens,
                        slots,
                        table,
                        query_locs,
                        seq_lens,
                        chunk_tokens,
                        width * chunk_tokens,
                        seq_len,
                    )
                )
            self.prepared[chunk] = contexts

    def prepare(self, tasks):
        if len(tasks) != 1:
            raise ValueError("hybrid benchmark requires one chunk per slot")
        _, self.current = tasks[0]
        return [self.prepared[self.current[0]]]

    def attend(self, original, inputs, query, key, value, *args, **kwargs):
        chunk, step = self.current
        block = self.model.start_layer + int(inputs.layer_idx)
        task = self.plan.task(chunk, step, block)
        push = self.pages
        push.prepare_round(self.plan.by_tick[task.tick])
        push.publish(task.key, key.unsqueeze(0), value.unsqueeze(0))
        reads = self.plan.reads[task]
        own = KVKey(chunk, step, block)
        waits = reads + ((own,) if self._rank in push.all_destinations(own) else ())
        push.await_pages(waits)
        output = original(inputs, query, key, value, *args, **kwargs)
        for previous in reads:
            # Releases are on the attention stream, after FA3 has read the page.
            push.release(previous, chunk, step)
        push.commit_round()
        return output

    def publish(self, tasks):
        pass

    def exchange(self, slot):
        if self.pages is not None:
            self.bytes_sent = self.pages.sent_bytes
            self.bytes_received = self.pages.received_bytes
        return []

    def evict(self, slot):
        pass

    def await_ready(self, slot):
        return 0

    def drain(self):
        pass

    def end_request(self, req):
        if self.pages is not None:
            self.bytes_sent = self.pages.sent_bytes
            self.bytes_received = self.pages.received_bytes
            self.pages.close(abort=sys.exc_info()[0] is not None)
            self.pages = None
        self.prepared.clear()
        self.current = None
        self._chunk_tokens.pop(req, None)

    def reset_all(self):
        if self.pages is not None:
            self.pages.close(abort=True)
            self.pages = None
        self.prepared.clear()
        self.current = None
        self._chunk_tokens.clear()


@contextmanager
def native_hybrid_attention(state):
    from vllm_omni.diffusion.models.waveserve_wan import transformer

    original = transformer.paged_write_attn
    transformer.paged_write_attn = lambda *a, **k: state.attend(original, *a, **k)
    try:
        yield
    finally:
        transformer.paged_write_attn = original
        if state.pages is not None:
            state.reset_all()
