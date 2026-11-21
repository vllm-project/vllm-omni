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

    def _prepare_manifest(self, q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
        planning = self.planning
        assert planning is not None
        q.record_stream(planning)
        k.record_stream(planning)
        planning.wait_stream(self.compute)
        with torch.cuda.stream(planning):
            selected = self._select(q, k, scale)
            wants = self._wants(selected)
            if self.pair_manifest is not None:
                dist.all_gather_into_tensor(self.pair_manifest.wants.flatten(), wants.flatten(), group=self.group)
                self.pair_manifest.compact()
            else:
                assert self.manifest is not None
                gives = torch.empty_like(wants)
                dist.all_to_all_single(gives, wants, group=self.group)
                self.manifest.compact(gives, wants)
        return selected

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

    def _exchange_packet(
        self,
        slot: int,
        destination: int,
        source: int,
        send_bytes: int,
        receive_bytes: int,
        *,
        receive_into: torch.Tensor | None = None,
        ready_stream: torch.cuda.Stream | None = None,
    ) -> torch.cuda.Event:
        inbox = self.receive[slot] if receive_into is None else receive_into
        ops = []
        if send_bytes:
            ops.append(dist.P2POp(dist.isend, self.send[slot][:send_bytes], group=self.group, group_peer=destination))
        if receive_bytes:
            ops.append(dist.P2POp(dist.irecv, inbox[:receive_bytes], group=self.group, group_peer=source))
        # Wait for packing and any earlier read of a reused receive slot.
        self.communication.wait_stream(self.compute if ready_stream is None else ready_stream)
        with torch.cuda.stream(self.communication):
            if ops:
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
            done = torch.cuda.Event()
            done.record(self.communication)
        self._sent[slot] = done
        return done

    def _post_direct_packet(self, index, k, v, outgoing, incoming_count, ready_stream=None) -> torch.cuda.Event:
        source, destination, slot = self.hops[index]
        heads, blocks = outgoing
        pack_kv(k, v, heads, blocks, self._prepare_packet(slot))
        block_bytes = kv_block_bytes(self.dim)
        send_bytes = heads.numel() * block_bytes
        self.payload_bytes += send_bytes
        if self.world == 4 and self.rank // 2 != destination // 2:
            self.cross_pair_payload_bytes += send_bytes
        return self._exchange_packet(
            slot, destination, source, send_bytes, incoming_count * block_bytes, ready_stream=ready_stream
        )

    def _direct(self, q, qs, k, v, local_k, selected, scale) -> None:
        send, receive = self._demands(selected)

        def issue(index: int) -> torch.cuda.Event:
            return self._post_direct_packet(index, k, v, send[index], receive[index].shape[1])

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

    def _direct_manifest(self, q, qs, k, v, local, selected, scale) -> None:
        planning, manifest = self.planning, self.manifest
        assert planning is not None and manifest is not None
        send, receive = manifest.plan()

        def issue(index: int) -> torch.cuda.Event:
            with torch.cuda.stream(planning):
                return self._post_direct_packet(index, k, v, send[index], receive[index].shape[1], planning)

        slots = len(self.send)
        events = [issue(index) for index in range(slots)]
        self._attend(0, self.rank, q, qs, *local, selected, scale)
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

    def _pair_relay(self, q, qs, k, v, local, selected, scale) -> None:
        manifest = self.pair_manifest
        planning = self.planning
        assert manifest is not None and planning is not None
        plan = manifest.plan()
        within, across, forwarded = self.rank ^ 1, self.rank ^ 2, self.rank ^ 3
        block_bytes = kv_block_bytes(self.dim)

        def attend_packet(owner: int, inbox: torch.Tensor, indices: torch.Tensor) -> None:
            heads, blocks = indices
            if heads.numel():
                place_kv(inbox, heads, blocks, self.key, self.key_scale, self.value, k.dtype)
                # Keep the merge order identical to full Ring attention.
                step = (self.rank - owner) % self.world
                self._attend(step, owner, q, qs, self.key, self.key_scale, self.value, selected, scale)

        with torch.cuda.stream(planning):
            pack_kv(k, v, *plan.local_send.unbind(), self._prepare_packet(0))
            local_bytes = plan.local_send.shape[1] * block_bytes
            local_event = self._exchange_packet(
                0, within, within, local_bytes, plan.local_receive.shape[1] * block_bytes, ready_stream=planning
            )
            pack_kv(k, v, *plan.cross_send.unbind(), self._prepare_packet(1))
            cross_bytes = plan.cross_send.shape[1] * block_bytes
            # The NCCL stream reads send[0] for the local hop before writing
            # the cross-socket packet into it for forwarding.
            relay_slot = self.send[0]
            cross_event = self._exchange_packet(
                1,
                across,
                across,
                cross_bytes,
                plan.cross_receive.shape[1] * block_bytes,
                receive_into=relay_slot,
                ready_stream=planning,
            )
        self._attend(0, self.rank, q, qs, *local, selected, scale)
        self.compute.wait_event(local_event)
        attend_packet(within, self.receive[0], plan.local_receive)
        self.compute.wait_event(cross_event)
        forward_bytes = plan.forward_count * block_bytes
        forwarded_event = self._exchange_packet(
            0,
            within,
            within,
            forward_bytes,
            plan.forward_receive.shape[1] * block_bytes,
        )
        # A prefix may include blocks wanted only by the next receiver. The
        # local attention mask still selects exactly the original subset.
        attend_packet(across, relay_slot, plan.cross_receive)
        self.compute.wait_event(forwarded_event)
        attend_packet(forwarded, self.receive[0], plan.forward_receive)
        self.payload_bytes = local_bytes + cross_bytes + forward_bytes
        self.cross_pair_payload_bytes = cross_bytes

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
                selected = (
                    self._select(q, local_k, scale) if planning is None else self._prepare_manifest(q, local_k, scale)
                )
                self.payload_bytes = 0
                self.cross_pair_payload_bytes = 0
                self.dense_payload_bytes = (self.world - 1) * 2 * k.numel() * k.element_size()
                self.lses.fill_(-torch.inf)
                qi, qs = self.provider.quantize(q, 128)
                if planning is not None:
                    ki, ks = self.provider.quantize(local_k, 64)
                    local = (ki, ks, self._compute_input(v))
                    # NCCL needs host-visible lengths. Wait only for the
                    # planning stream while local preparation is in flight.
                    planning.synchronize()
                    selected.record_stream(self.compute)
                    k.record_stream(planning)
                    v.record_stream(planning)
                    if self.pair_manifest is None:
                        self._direct_manifest(qi, qs, k, v, local, selected, scale)
                    else:
                        self._pair_relay(qi, qs, k, v, local, selected, scale)
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
