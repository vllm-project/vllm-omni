# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Expand native chunk tasks and KV sources to per-block transport rounds."""

from dataclasses import dataclass

from vllm_omni.experimental.ar_diffusion.chunk_schedule import Ordering


@dataclass(frozen=True)
class KVKey:
    chunk: int
    step: int
    block: int


@dataclass(frozen=True)
class BlockTask:
    tick: int
    rank: int
    key: KVKey


class BlockPlan:
    def __init__(self, native, blocks):
        schedule = native.schedule
        groups = schedule.layer_groups
        if blocks % groups or schedule.stages != schedule.num_denoise_steps + 1:
            raise ValueError("block transport requires equal partitions and one rank per stage/group")
        if schedule.ordering is not Ordering.INTERLEAVED or schedule.kv_history_chunks < 1:
            raise ValueError("block transport requires interleaved execution with positive history")
        self.blocks, self.stages, self.chunks = blocks, schedule.stages, schedule.chunks
        self.world, self.history = native.world, schedule.kv_history_chunks
        self.blocks_per_rank = blocks // groups
        self.by_key, self.by_tick, self.reads, self.destinations = {}, {}, {}, {}
        for slot in range(native.num_slots):
            for local in range(self.blocks_per_rank):
                tick = slot * self.blocks_per_rank + local
                tasks = []
                for rank in range(native.world):
                    version = native.task(slot, rank)
                    if version is None:
                        continue
                    chunk, step = version
                    block = rank % groups * self.blocks_per_rank + local
                    task = BlockTask(tick, rank, KVKey(chunk, step, block))
                    self.by_key[task.key] = task
                    tasks.append(task)
                    selected = tuple(KVKey(*source.version, block) for source in native.sources(version, rank))
                    self.reads[task] = selected
                    for key in selected:
                        # Iteration in native execution order records the last
                        # reader on each rank, for device-side release tickets.
                        self.destinations.setdefault(key, {})[rank] = version
                self.by_tick[tick] = tuple(tasks)
        for key, destinations in self.destinations.items():
            producer = self.by_key[key]
            slot = producer.tick // self.blocks_per_rank
            if any(native.task(slot, rank) is None for rank in destinations):
                raise ValueError(f"idle native rank would have to receive a producer round: {key}")

    def task(self, chunk, step, block):
        return self.by_key[KVKey(chunk, step, block)]

    def owner(self, key):
        return self.by_key[key].rank

    def local_only(self, hosts):
        return all(
            hosts[self.owner(key)] == hosts[rank]
            for key, destinations in self.destinations.items()
            for rank in destinations
        )
