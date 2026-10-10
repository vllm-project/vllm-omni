# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""Layer-major KV ring with local IPC tickets and static remote NCCL rounds."""

import ctypes
import socket

import torch
import torch.distributed as dist

from .block_plan import KVKey

# Keep failed requests' exported storage alive until torchrun kills the job:
# another rank may still have a mapped pointer or an asynchronous copy.
_ABORTED = []


class LayerMajorPages:
    def __init__(self, plan, spec, chunk_tokens, device, dtype, group):
        from cuda.bindings import driver

        self.driver, self.plan, self.group = driver, plan, group
        self.device = torch.device(device)
        self.rank, self.world = dist.get_rank(group), plan.world
        self.stages, self.storage_blocks = plan.stages, plan.blocks_per_rank
        self.capacity = min(plan.chunks, plan.history + 1)
        if plan.chunks >= 2**31:
            raise ValueError("IPC chunk tickets require fewer than 2**31 chunks")
        self.page_shape = (1, chunk_tokens, spec.num_kv_heads, spec.head_size)
        self.bytes_per_tensor = (
            chunk_tokens * spec.num_kv_heads * spec.head_size * torch.empty((), dtype=dtype).element_size()
        )
        self.sent_bytes = self.received_bytes = 0
        self.streams, self.opened, self.peers = {}, {}, {}
        self.staged, self.remote_ready = {}, {}
        self.round_tasks = ()
        self.round_submitted = self.closed = False
        self.buffers = self.flags = None
        error = info = None
        try:
            self.buffers = torch.empty(
                (self.storage_blocks, 2, self.capacity, self.stages, *self.page_shape), dtype=dtype, device=self.device
            )
            self.flags = torch.zeros(
                (2, self.capacity, self.stages, self.storage_blocks), dtype=torch.int32, device=self.device
            )
            info = (
                socket.gethostname(),
                str(torch.cuda.get_device_properties(self.device).uuid),
                (self.capacity, self.stages, self.storage_blocks, self.page_shape, dtype),
                self.export(self.buffers),
                self.export(self.flags),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        entries = self.gather((info, error))
        if any(error for _, error in entries):
            raise RuntimeError("IPC page allocation failed: " + str([error for _, error in entries]))
        infos = [entry for entry, _ in entries]
        self.hosts = tuple(entry[0] for entry in infos)
        if any(entry[2] != info[2] for entry in infos):
            raise ValueError("IPC pages require matching shapes")
        if any(
            len({infos[rank][1] for rank, host in enumerate(self.hosts) if host == name}) != self.hosts.count(name)
            for name in set(self.hosts)
        ):
            raise ValueError("hybrid IPC peers require distinct GPUs on each host")
        self.local_only = plan.local_only(self.hosts)
        self.round_stream = None if self.local_only else torch.cuda.Stream(device=self.device)
        error = None
        try:
            for rank, (_, _, _, data, flags) in enumerate(infos):
                if not self.local_peer(rank):
                    continue
                self.peers[rank] = (
                    (self.buffers.data_ptr(), self.flags.data_ptr())
                    if rank == self.rank
                    else (self.open(data), self.open(flags))
                )
                # One stream per destination prevents a slow reader from
                # blocking publications needed to advance another reader.
                self.streams[rank] = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self.gather(error)
        if any(errors):
            for pointer in self.opened.values():
                self.check(driver.cuIpcCloseMemHandle(pointer))
            self.opened.clear()
            raise RuntimeError("IPC page mapping failed: " + str(errors))
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group, device_ids=[self.device.index])

    def gather(self, value):
        values = [None] * self.world
        dist.all_gather_object(values, value, group=self.group)
        return values

    @staticmethod
    def check(result):
        error, *values = result
        if int(error):
            raise RuntimeError(f"CUDA driver error: {error}")
        return values[0] if len(values) == 1 else values

    def export(self, tensor):
        pointer = tensor.data_ptr()
        handle = self.check(self.driver.cuIpcGetMemHandle(pointer))
        base, _ = self.check(self.driver.cuMemGetAddressRange(pointer))
        raw = bytes(handle.reserved) if hasattr(handle, "reserved") else ctypes.string_at(handle.getPtr(), 64)
        return raw, pointer - int(base)

    def open(self, descriptor):
        raw, offset = descriptor
        if raw not in self.opened:
            handle = self.driver.CUipcMemHandle()
            if hasattr(handle, "reserved"):
                handle.reserved = raw
            else:
                if len(raw) != 64:
                    raise ValueError("invalid CUDA IPC handle")
                ctypes.memmove(handle.getPtr(), raw, 64)
            flag = int(self.driver.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS.value)
            self.opened[raw] = int(self.check(self.driver.cuIpcOpenMemHandle(handle, flag)))
        return self.opened[raw] + offset

    def copy(self, dst, src, size, stream):
        self.check(self.driver.cuMemcpyDtoDAsync(dst, src, size, stream))

    def write(self, stream, address, value):
        self.check(self.driver.cuStreamWriteValue32(stream, address, value, 0))

    def wait(self, stream, address, value):
        flag = int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ.value)
        self.check(self.driver.cuStreamWaitValue32(stream, address, value, flag))

    def storage_index(self, key):
        return key.block % self.storage_blocks

    def flag(self, base, kind, key):
        slot = key.chunk % self.capacity * self.stages + key.step
        return base + ((kind * self.capacity * self.stages + slot) * self.storage_blocks + self.storage_index(key)) * 4

    def page(self, key):
        block = self.storage_index(key)
        return tuple(self.buffers[block, field, key.chunk % self.capacity, key.step] for field in range(2))

    def address(self, base, key, field):
        block = self.storage_index(key)
        slot = key.chunk % self.capacity * self.stages + key.step
        return base + ((block * 2 + field) * self.capacity * self.stages + slot) * self.bytes_per_tensor

    def local_peer(self, rank):
        return self.hosts[rank] == self.hosts[self.rank]

    def global_rank(self, rank):
        return dist.get_global_rank(self.group, rank) if self.group is not None else rank

    def all_destinations(self, key):
        return self.plan.destinations.get(key, {})

    def previous(self, key, destination):
        old = key.chunk - self.capacity
        while old >= 0:
            previous = KVKey(old, key.step, key.block)
            if destination in self.all_destinations(previous):
                return old
            old -= self.capacity
        return None

    def prepare_round(self, tasks):
        if not self.local_only:
            self.round_tasks = tuple(
                sorted(tasks, key=lambda task: (task.key.chunk, task.key.step, task.key.block, task.rank))
            )
            self.round_submitted = False

    def publish(self, key, k, v):
        if (
            tuple(k.shape) != self.page_shape
            or v.shape != k.shape
            or k.dtype != self.buffers.dtype
            or v.dtype != k.dtype
        ):
            raise ValueError("hybrid KV shape/dtype changed")
        packed = (k.contiguous(), v.contiguous())
        if not self.local_only:
            self.staged[key] = packed
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        for destination in self.all_destinations(key):
            if not self.local_peer(destination):
                continue
            stream = self.streams[destination]
            stream.wait_event(ready)
            data, flags = self.peers[destination]
            old = self.previous(key, destination)
            if old is not None:
                self.wait(stream.cuda_stream, self.flag(flags, 1, key), old + 1)
            for field, value in enumerate(packed):
                self.copy(self.address(data, key, field), value.data_ptr(), self.bytes_per_tensor, stream.cuda_stream)
                value.record_stream(stream)
            self.write(stream.cuda_stream, self.flag(flags, 0, key), key.chunk + 1)
            if destination != self.rank:
                self.sent_bytes += 2 * self.bytes_per_tensor
        if not self.local_only:
            self.commit_round()

    def commit_round(self):
        if self.local_only or self.round_submitted:
            return
        self.round_submitted = True
        operations, received = [], []
        for task in self.round_tasks:
            key = task.key
            for destination in sorted(self.all_destinations(key)):
                if self.hosts[task.rank] == self.hosts[destination]:
                    continue
                if self.rank == task.rank:
                    tensors = self.staged[key]
                    op, peer = dist.isend, destination
                    self.sent_bytes += 2 * self.bytes_per_tensor
                elif self.rank == destination:
                    tensors = self.page(key)
                    op, peer = dist.irecv, task.rank
                    received.append(key)
                    self.received_bytes += 2 * self.bytes_per_tensor
                else:
                    continue
                operations.extend(dist.P2POp(op, tensor, self.global_rank(peer), self.group) for tensor in tensors)
        if operations:
            # Capture producer readiness BEFORE this tick's history waits;
            # waiting on compute again would create a dependency cycle.
            self.round_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.round_stream):
                for work in dist.batch_isend_irecv(operations):
                    work.wait()
                done = torch.cuda.Event()
                done.record(self.round_stream)
                for tensors in self.staged.values():
                    for tensor in tensors:
                        tensor.record_stream(self.round_stream)
            for key in received:
                self.remote_ready[key] = done
        self.staged.clear()

    def await_pages(self, keys):
        current = torch.cuda.current_stream(self.device)
        for key in keys:
            if self.local_peer(self.plan.owner(key)):
                self.wait(current.cuda_stream, self.flag(self.flags.data_ptr(), 0, key), key.chunk + 1)
            else:
                if key not in self.remote_ready:
                    raise RuntimeError(f"remote page missing from producer round: {key}")
                current.wait_event(self.remote_ready[key])

    def release(self, key, chunk, step):
        if self.all_destinations(key)[self.rank] != (chunk, step):
            return
        if not self.local_peer(self.plan.owner(key)):
            self.remote_ready.pop(key)
        elif self.plan.owner(key) != self.rank:
            self.received_bytes += 2 * self.bytes_per_tensor
        # FA3 finished reading this slot. A later local IPC producer can reuse
        # it even when the previous page arrived through NCCL.
        self.write(
            torch.cuda.current_stream(self.device).cuda_stream, self.flag(self.flags.data_ptr(), 1, key), key.chunk + 1
        )

    def close(self, abort=False):
        if self.closed:
            return
        if abort:
            _ABORTED.append(self)
            self.closed = True
            return
        if self.round_stream is not None:
            self.round_stream.synchronize()
        self.remote_ready.clear()
        current = torch.cuda.current_stream(self.device)
        for stream in self.streams.values():
            done = torch.cuda.Event()
            done.record(stream)
            current.wait_event(done)
        current.synchronize()
        dist.barrier(group=self.group, device_ids=[self.device.index])
        for pointer in self.opened.values():
            self.check(self.driver.cuIpcCloseMemHandle(pointer))
        self.opened.clear()
        self.peers.clear()
        # Every importer must close before exporters free their storage.
        dist.barrier(group=self.group, device_ids=[self.device.index])
        self.buffers = self.flags = None
        self.closed = True
