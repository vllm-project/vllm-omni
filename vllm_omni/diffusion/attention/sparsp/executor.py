# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sparse Ring attention with full or demand-directed NCCL K/V exchange."""

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec
from vllm_omni.diffusion.attention.sparsp.kernels import kv_block_bytes, merge_partials, pack_kv, place_kv
from vllm_omni.diffusion.attention.sparsp.manifest import DemandManifest, PairRelayManifest, peer_hops
from vllm_omni.diffusion.attention.sparsp.provider import SpargeProvider
from vllm_omni.diffusion.attention.sparsp.selection import cdf_block_mask
from vllm_omni.diffusion.distributed.comm import RingComm
from vllm_omni.platforms import current_omni_platform


class SparseRingExecutor:
    """Own one geometry's scratch and peer slots within a Ring group."""

    def __init__(self, config: SparSPSpec, group: dist.ProcessGroup, heads: int, dim: int, device: torch.device):
        self.config, self.group, self.device = config, group, device
        self.heads, self.dim = heads, dim
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if config.routing == "selective_pair_relay" and self.world != 4:
            raise ValueError("SparSP pair relay requires four Ring ranks")
        self.provider = SpargeProvider(config, heads, device)
        self.compute = torch.cuda.Stream(device=device)
        self.communication = torch.cuda.Stream(device=device)
        self.planning = torch.cuda.Stream(device=device) if config.demand_planner == "manifest" else None
        self.manifest: DemandManifest | None = None
        self.pair_manifest: PairRelayManifest | None = None
        self.hops = peer_hops(self.rank, self.world, min(2, self.world - 1))
        self.tokens = 0
        self.wire_dtype: torch.dtype | None = None
        self._closed = self._poisoned = False
        self.payload_bytes = 0
        self.cross_pair_payload_bytes = 0
        self.forwarded_payload_bytes = 0
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
            if self.config.routing == "selective_pair_relay":
                self.pair_manifest = PairRelayManifest(self.rank, self.heads, tokens // 64, self.device)
            elif self.config.demand_planner == "manifest":
                self.manifest = DemandManifest(self.rank, self.world, self.heads, tokens // 64, self.device)
            capacity = self.heads * (tokens // 64) * kv_block_bytes(self.dim)
            slots = min(2, self.world - 1)
            self.send = [torch.empty(capacity, device=self.device, dtype=torch.uint8) for _ in range(slots)]
            self.receive = [torch.empty_like(buffer) for buffer in self.send]
            self._sent: list[torch.cuda.Event | None] = [None] * slots
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

    def _prepare_manifest(
        self, q: torch.Tensor, local_k: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
    ) -> tuple[torch.Tensor, torch.cuda.Event, torch.cuda.Event]:
        planning = self.planning
        assert planning is not None
        for tensor in (q, local_k, k, v):
            tensor.record_stream(planning)
        planning.wait_stream(self.compute)
        with torch.cuda.stream(planning):
            selected = self._select(q, local_k, scale)
            # Local attention needs the selection, but not the exchanged demand
            # or the compact packet lengths. Let it run while we build those.
            selection_ready = planning.record_event()
            wants = self._wants(selected)
            pair_manifest = self.pair_manifest
            if pair_manifest is not None:
                dist.all_gather_into_tensor(pair_manifest.wants.flatten(), wants.flatten(), group=self.group)
                pair_manifest.compact()
                # Rows 0 and 2 are the two outgoing packets.
                plans, counts = pair_manifest.indices[:3:2], pair_manifest.counts[:3:2]
            else:
                assert self.manifest is not None
                gives = torch.empty_like(wants)
                dist.all_to_all_single(gives, wants, group=self.group)
                self.manifest.compact(gives, wants)
                plans, counts = self.manifest.indices[0], self.manifest.counts[0]
            lengths_ready = planning.record_event()
            for slot in range(len(self.send)):
                heads, blocks = plans[slot].unbind()
                pack_kv(k, v, heads, blocks, self._prepare_packet(slot), counts[slot])
        return selected, selection_ready, lengths_ready

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

    def _packet_ops(
        self,
        outbox: torch.Tensor,
        inbox: torch.Tensor,
        destination: int,
        source: int,
        send_bytes: int,
        receive_bytes: int,
    ) -> list[dist.P2POp]:
        ops = []
        if send_bytes:
            ops.append(dist.P2POp(dist.isend, outbox[:send_bytes], group=self.group, group_peer=destination))
        if receive_bytes:
            ops.append(dist.P2POp(dist.irecv, inbox[:receive_bytes], group=self.group, group_peer=source))
        return ops

    def _post_packets(self, ops: list[dist.P2POp], ready_stream: torch.cuda.Stream | None) -> torch.cuda.Event:
        if ready_stream is not None:
            self.communication.wait_stream(ready_stream)
        with torch.cuda.stream(self.communication):
            if ops:
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
            done = torch.cuda.Event()
            done.record(self.communication)
        return done

    def _post_direct_packet(self, index, outgoing_count, incoming_count, ready_stream) -> torch.cuda.Event:
        source, destination, slot = self.hops[index]
        block_bytes = kv_block_bytes(self.dim)
        send_bytes = outgoing_count * block_bytes
        self.payload_bytes += send_bytes
        if self.world == 4 and self.rank // 2 != destination // 2:
            self.cross_pair_payload_bytes += send_bytes
        ops = self._packet_ops(
            self.send[slot], self.receive[slot], destination, source, send_bytes, incoming_count * block_bytes
        )
        # Wait for packing and any earlier read of a reused receive slot.
        done = self._post_packets(ops, ready_stream)
        self._sent[slot] = done
        return done

    def _direct(self, q, qs, k, v, local_k, selected, scale) -> None:
        send, receive = self._demands(selected)

        def issue(index: int) -> torch.cuda.Event:
            heads, blocks = send[index]
            pack_kv(k, v, heads, blocks, self._prepare_packet(self.hops[index].slot))
            return self._post_direct_packet(index, heads.numel(), receive[index].shape[1], self.compute)

        event = issue(0)
        ki, ks = self.provider.quantize(local_k, 64)
        self._attend(0, self.rank, q, qs, ki, ks, self._compute_input(v), selected, scale)
        for index, (heads, blocks) in enumerate(receive):
            self.compute.wait_event(event)
            if index + 1 < len(receive):
                event = issue(index + 1)
            if heads.numel():
                source, _, slot = self.hops[index]
                place_kv(self.receive[slot], heads, blocks, self.key, self.key_scale, self.value, k.dtype)
                self._attend(
                    index + 1,
                    source,
                    q,
                    qs,
                    self.key,
                    self.key_scale,
                    self.value,
                    selected,
                    scale,
                )

    def _direct_manifest(self, q, qs, k, v, selected, scale) -> None:
        planning, manifest = self.planning, self.manifest
        assert planning is not None and manifest is not None
        send, receive = manifest.plan()
        slots = len(self.send)

        def issue(index: int) -> torch.cuda.Event:
            with torch.cuda.stream(planning):
                heads, blocks = send[index]
                if index >= slots:
                    pack_kv(k, v, heads, blocks, self._prepare_packet(self.hops[index].slot))
                return self._post_direct_packet(index, heads.numel(), receive[index].shape[1], planning)

        events = [issue(index) for index in range(slots)]
        for index, (heads, blocks) in enumerate(receive):
            self.compute.wait_event(events[index])
            source, _, slot = self.hops[index]
            if heads.numel():
                place_kv(self.receive[slot], heads, blocks, self.key, self.key_scale, self.value, k.dtype)
            if index + slots < len(receive):
                # Placement has consumed the old receive buffer. Its next
                # transfer can overlap attention on the placed K/V.
                planning.wait_stream(self.compute)
                events.append(issue(index + slots))
            if heads.numel():
                self._attend(index + 1, source, q, qs, self.key, self.key_scale, self.value, selected, scale)

    def _pair_relay(self, q, qs, wire_dtype, selected, scale) -> None:
        manifest = self.pair_manifest
        planning = self.planning
        assert manifest is not None and planning is not None
        plan = manifest.plan()
        within = self.rank ^ 1
        forwarded = plan.cross_source ^ 1
        block_bytes = kv_block_bytes(self.dim)

        def attend_packet(owner: int, inbox: torch.Tensor, indices: torch.Tensor) -> None:
            heads, blocks = indices
            if heads.numel():
                place_kv(inbox, heads, blocks, self.key, self.key_scale, self.value, wire_dtype)
                # Keep the merge order identical to full Ring attention.
                step = (self.rank - owner) % self.world
                self._attend(step, owner, q, qs, self.key, self.key_scale, self.value, selected, scale)

        local_bytes = plan.local_send.shape[1] * block_bytes
        cross_bytes = plan.cross_send.shape[1] * block_bytes
        local_ops = self._packet_ops(
            self.send[0], self.receive[0], within, within, local_bytes, plan.local_receive.shape[1] * block_bytes
        )
        cross_ops = self._packet_ops(
            self.send[1],
            self.receive[1],
            plan.cross_destination,
            plan.cross_source,
            cross_bytes,
            plan.cross_receive.shape[1] * block_bytes,
        )
        first_event = self._post_packets(local_ops + cross_ops, planning)
        forward_bytes = plan.forward_count * block_bytes
        # The first exchange has finished reading send[0] and writing receive[1]
        # before the communication stream reuses both for the forwarding hop.
        forward_ops = self._packet_ops(
            self.receive[1], self.send[0], within, within, forward_bytes, plan.forward_receive.shape[1] * block_bytes
        )
        forward_event = self._post_packets(forward_ops, None)
        self._sent[0], self._sent[1] = forward_event, first_event
        self.compute.wait_event(first_event)
        attend_packet(within, self.receive[0], plan.local_receive)
        # A prefix may include blocks wanted only by the next receiver. The
        # local attention mask still selects exactly the original subset.
        attend_packet(plan.cross_source, self.receive[1], plan.cross_receive)
        self.compute.wait_event(forward_event)
        attend_packet(forwarded, self.send[0], plan.forward_receive)
        self.payload_bytes = local_bytes + cross_bytes + forward_bytes
        self.cross_pair_payload_bytes = cross_bytes
        self.forwarded_payload_bytes = forward_bytes

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
                if self.world == 4 and self.rank // 2 != ((self.rank + 1) % self.world) // 2:
                    self.cross_pair_payload_bytes += payload_bytes
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
                planning = self.planning
                if planning is None:
                    selected = self._select(q, local_k, scale)
                else:
                    selected, selection_ready, lengths_ready = self._prepare_manifest(q, local_k, k, v, scale)
                self.payload_bytes = 0
                self.cross_pair_payload_bytes = 0
                self.dense_payload_bytes = (self.world - 1) * 2 * k.numel() * k.element_size()
                self.lses.fill_(-torch.inf)
                qi, qs = self.provider.quantize(q, 128)
                if planning is not None:
                    ki, ks = self.provider.quantize(local_k, 64)
                    local = (ki, ks, self._compute_input(v))
                    self.compute.wait_event(selection_ready)
                    selected.record_stream(self.compute)
                    self._attend(0, self.rank, qi, qs, *local, selected, scale)
                    # Host-count NCCL APIs still need the packet lengths.
                    # The GPU can prepare and attend to local K/V during this wait.
                    lengths_ready.synchronize()
                    if self.pair_manifest is None:
                        self._direct_manifest(qi, qs, k, v, selected, scale)
                    else:
                        self._pair_relay(qi, qs, k.dtype, selected, scale)
                elif self.config.routing == "selective_direct":
                    self._direct(qi, qs, k, v, local_k, selected, scale)
                else:
                    self._relay(qi, qs, k.contiguous(), v.contiguous(), local_k, selected, scale)
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
        if self.planning is not None:
            self.planning.synchronize()
        self._closed = True
