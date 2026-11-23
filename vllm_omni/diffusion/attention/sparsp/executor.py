# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sparse Ring attention with full or demand-directed NCCL K/V exchange."""

from typing import NamedTuple

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec
from vllm_omni.diffusion.attention.sparsp.kernels import kv_block_bytes, merge_partials, pack_kv, place_kv
from vllm_omni.diffusion.attention.sparsp.provider import SpargeProvider
from vllm_omni.diffusion.attention.sparsp.selection import cdf_block_mask
from vllm_omni.diffusion.distributed.comm import RingComm
from vllm_omni.platforms import current_omni_platform


class PeerHop(NamedTuple):
    source: int
    destination: int
    slot: int


def peer_hops(rank: int, world: int, slots: int) -> tuple[PeerHop, ...]:
    return tuple(PeerHop((rank - hop) % world, (rank + hop) % world, (hop - 1) % slots) for hop in range(1, world))


class SparseRingExecutor:
    """Own one geometry's scratch and peer slots within a Ring group."""

    def __init__(self, config: SparSPSpec, group: dist.ProcessGroup, heads: int, dim: int, device: torch.device):
        self.config, self.group, self.device = config, group, device
        self.heads, self.dim = heads, dim
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.provider = SpargeProvider(config, heads, device)
        self.compute = torch.cuda.Stream(device=device)
        self.communication = torch.cuda.Stream(device=device)
        # The manifest and per-hop planners post every hop before local attention, so each hop needs its own slot.
        self.slots = self.world - 1 if config.demand_planner in ("manifest", "per_hop") else min(2, self.world - 1)
        self.hops = peer_hops(self.rank, self.world, self.slots)
        self.tokens = 0
        self.wire_dtype: torch.dtype | None = None
        self._closed = self._poisoned = False
        self.payload_bytes = 0
        self.dense_payload_bytes = 0

    def _buffers(self, tokens: int, wire_dtype: torch.dtype) -> None:
        if tokens == self.tokens and wire_dtype == self.wire_dtype:
            return
        current_omni_platform.synchronize()
        shape = (1, self.heads, tokens, self.dim)
        self.partials = torch.empty((self.world, *shape), device=self.device, dtype=torch.float16)
        self.lses = torch.empty((self.world, 1, self.heads, tokens), device=self.device, dtype=torch.float32)
        if self.config.routing != "full_ring":
            self.key = torch.empty(shape, device=self.device, dtype=torch.int8)
            self.key_scale = torch.empty((1, self.heads, tokens // 64), device=self.device, dtype=torch.float32)
            self.value = torch.empty(shape, device=self.device, dtype=torch.float16)
            capacity = self.heads * (tokens // 64) * kv_block_bytes(self.dim)
            self.send = [torch.empty(capacity, device=self.device, dtype=torch.uint8) for _ in range(self.slots)]
            self.receive = [torch.empty_like(buffer) for buffer in self.send]
            self._sent: list[torch.cuda.Event | None] = [None] * self.slots
        else:
            self.rotations = [
                tuple(torch.empty(shape, device=self.device, dtype=wire_dtype) for _ in range(2))
                for _ in range(min(2, self.world - 1))
            ]
        self.tokens = tokens
        self.wire_dtype = wire_dtype

    def _select(self, q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
        blocks = self.tokens // 64
        if self.config.cdf == 1:
            return torch.ones(
                (1, self.heads, (self.tokens + 127) // 128, blocks * self.world), device=self.device, dtype=torch.bool
            )
        pooled_q, similar_q = self.provider.pool(q, 128)
        pooled_k, similar_k = self.provider.pool(k, 64)
        keys = [torch.empty_like(pooled_k) for _ in range(self.world)]
        similarities = [torch.empty_like(similar_k) for _ in range(self.world)]
        dist.all_gather(keys, pooled_k, group=self.group)
        dist.all_gather(similarities, similar_k, group=self.group)
        return cdf_block_mask(
            pooled_q, torch.cat(keys, dim=2), similar_q, torch.cat(similarities, dim=2), self.config.cdf, scale
        )

    def _wants(self, selected: torch.Tensor) -> torch.Tensor:
        blocks = self.tokens // 64
        return selected[0].any(dim=1).reshape(self.heads, self.world, blocks).transpose(0, 1).contiguous()

    def _demands(self, selected: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Exchange only per-head block demand, in the same order on both peers."""
        wants = self._wants(selected)
        gives = torch.empty_like(wants)
        dist.all_to_all_single(gives, wants, group=self.group)
        peer_ranks = [(self.rank + offset) % self.world for offset in range(1, self.world)]
        masks = torch.stack([gives[peer] for peer in peer_ranks] + [wants[peer] for peer in reversed(peer_ranks)])
        indices = masks.nonzero()[:, 1:].T.contiguous()
        counts = masks.sum(dim=(1, 2)).tolist()
        plans = indices.split(counts, dim=1)
        return plans[: self.world - 1], plans[self.world - 1 :]

    def _attend(self, step, owner, q, qs, k, ks, v, selected, scale) -> None:
        blocks = self.tokens // 64
        mask = selected[..., owner * blocks : (owner + 1) * blocks].contiguous()
        self.provider.attend(q, qs, k, ks, v, mask, self.partials[step], self.lses[step], scale)

    @staticmethod
    def _compute_input(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.float16, memory_format=torch.contiguous_format).contiguous()

    def _prepare_packet(self, slot: int) -> torch.Tensor:
        event = self._sent[slot]
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
        return self.send[slot]

    def _post_direct_packet(self, index, outgoing_count, incoming_count) -> torch.cuda.Event:
        source, destination, slot = self.hops[index]
        block_bytes = kv_block_bytes(self.dim)
        send_bytes = outgoing_count * block_bytes
        receive_bytes = incoming_count * block_bytes
        self.payload_bytes += send_bytes
        ops = []
        if send_bytes:
            ops.append(dist.P2POp(dist.isend, self.send[slot][:send_bytes], group=self.group, group_peer=destination))
        if receive_bytes:
            ops.append(dist.P2POp(dist.irecv, self.receive[slot][:receive_bytes], group=self.group, group_peer=source))
        # Wait for packing and any earlier read of a reused receive slot.
        self.communication.wait_stream(self.compute)
        with torch.cuda.stream(self.communication):
            if ops:
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
            done = torch.cuda.Event()
            done.record(self.communication)
        self._sent[slot] = done
        return done

    def _attend_received(self, index, q, qs, receive, selected, scale, wire_dtype) -> None:
        heads, blocks = receive
        if heads.numel():
            source, _, slot = self.hops[index]
            place_kv(self.receive[slot], heads, blocks, self.key, self.key_scale, self.value, wire_dtype)
            self._attend(index + 1, source, q, qs, self.key, self.key_scale, self.value, selected, scale)

    def _direct(self, q, qs, k, v, local_k, selected, scale) -> None:
        """Exchange all demand up front, then keep one hop in flight ahead of compute."""
        send, receive = self._demands(selected)

        def issue(index: int) -> torch.cuda.Event:
            heads, blocks = send[index]
            pack_kv(k, v, heads, blocks, self._prepare_packet(self.hops[index].slot))
            return self._post_direct_packet(index, heads.numel(), receive[index].shape[1])

        event = issue(0)
        ki, ks = self.provider.quantize(local_k, 64)
        self._attend(0, self.rank, q, qs, ki, ks, self._compute_input(v), selected, scale)
        for index in range(len(receive)):
            self.compute.wait_event(event)
            if index + 1 < len(receive):
                event = issue(index + 1)
            self._attend_received(index, q, qs, receive[index], selected, scale, k.dtype)

    def _direct_posted(self, q, qs, k, v, local_k, selected, scale) -> None:
        """Post every hop before local attention (sparse_video's NCCL put schedule).

        ``manifest`` (sparse_video without RING_PUT_REQ): one demand exchange for
        all hops, and every packed index is derived before any payload is posted.
        ``per_hop`` (sparse_video RING_PUT_REQ=hop): receive indices are known up
        front, but each hop exchanges its own demand and reads back its give
        indices immediately before packing and posting that hop, so each host
        round trip queues behind the previous hop's payload.
        """
        manifest = self.config.demand_planner == "manifest"
        if manifest:
            send, receive = self._demands(selected)
        else:
            wants = self._wants(selected)
            receive = tuple(wants[source].nonzero().T.contiguous() for source, _, _ in self.hops)
        events = []
        for index, (source, destination, slot) in enumerate(self.hops):
            if manifest:
                heads, blocks = send[index]
            else:
                give = torch.empty_like(wants[0])
                requests = [
                    dist.P2POp(dist.isend, wants[source], group=self.group, group_peer=source),
                    dist.P2POp(dist.irecv, give, group=self.group, group_peer=destination),
                ]
                for work in dist.batch_isend_irecv(requests):
                    work.wait()
                heads, blocks = give.nonzero(as_tuple=True)
            pack_kv(k, v, heads, blocks, self._prepare_packet(slot))
            events.append(self._post_direct_packet(index, heads.numel(), receive[index].shape[1]))
        ki, ks = self.provider.quantize(local_k, 64)
        self._attend(0, self.rank, q, qs, ki, ks, self._compute_input(v), selected, scale)
        for index in range(len(self.hops)):
            self.compute.wait_event(events[index])
            self._attend_received(index, q, qs, receive[index], selected, scale, k.dtype)

    def _relay(self, q, qs, k, v, local_k, selected, scale) -> None:
        comm = RingComm(self.group)
        payload_bytes = 2 * k.numel() * k.element_size()
        for step in range(self.world):
            if step + 1 < self.world:
                next_k, next_v = self.rotations[step % len(self.rotations)]
                self.communication.wait_stream(self.compute)
                with torch.cuda.stream(self.communication):
                    comm.send_recv(k, next_k)
                    comm.send_recv(v, next_v)
                    comm.commit()
                    comm.wait()
                    event = torch.cuda.Event()
                    event.record(self.communication)
                self.payload_bytes += payload_bytes
            ki, ks = self.provider.quantize(local_k if step == 0 else self._compute_input(k), 64)
            self._attend(step, (self.rank - step) % self.world, q, qs, ki, ks, self._compute_input(v), selected, scale)
            if step + 1 < self.world:
                self.compute.wait_event(event)
                k, v = next_k, next_v

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
        if self._closed or self._poisoned:
            raise RuntimeError("SparSP executor is closed or poisoned")
        current = torch.cuda.current_stream(self.device)
        self.compute.wait_stream(current)
        for tensor in (query, key, value):
            tensor.record_stream(self.compute)
        try:
            with torch.cuda.stream(self.compute):
                self._buffers(query.shape[1], key.dtype)
                q = self._compute_input(query.transpose(1, 2))
                k, v = key.transpose(1, 2), value.transpose(1, 2)
                local_k = self._compute_input(k)
                selected = self._select(q, local_k, scale)
                self.payload_bytes = 0
                self.dense_payload_bytes = (self.world - 1) * 2 * k.numel() * k.element_size()
                self.lses.fill_(-torch.inf)
                qi, qs = self.provider.quantize(q, 128)
                if self.config.routing == "full_ring":
                    self._relay(qi, qs, k.contiguous(), v.contiguous(), local_k, selected, scale)
                elif self.config.demand_planner == "torch":
                    self._direct(qi, qs, k, v, local_k, selected, scale)
                else:
                    self._direct_posted(qi, qs, k, v, local_k, selected, scale)
                result = merge_partials(self.partials, self.lses).transpose(1, 2).to(query.dtype).contiguous()
            current.wait_stream(self.compute)
            result.record_stream(current)
            return result
        except BaseException:
            self._poisoned = True
            raise

    def close(self) -> None:
        """Drain the compute/transport streams before the coordinator teardown."""
        if self._closed:
            return
        self.compute.synchronize()
        self.communication.synchronize()
        self._closed = True
