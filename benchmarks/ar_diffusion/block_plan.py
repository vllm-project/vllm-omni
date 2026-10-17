# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Index native cells and KV sources for hybrid transport rounds."""

from dataclasses import dataclass

from vllm_omni.experimental.ar_diffusion.cell_schedule import Ordering


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
    def __init__(self, native):
        schedule = native.schedule
        groups = schedule.layer_groups
        blocks = schedule.num_blocks
        if blocks % groups or schedule.stages != schedule.num_denoise_steps + 1:
            raise ValueError("block transport requires equal partitions and one rank per stage/group")
        if schedule.ordering is not Ordering.LATEST_KV or schedule.kv_history_chunks < 1:
            raise ValueError("block transport requires Latest-KV execution with positive history")
        self.blocks, self.stages, self.chunks = blocks, schedule.stages, schedule.chunks
        self.world, self.history = native.world, schedule.kv_history_chunks
        self.blocks_per_rank = blocks // groups
        self.by_key, self.by_tick, self.reads, self.destinations = {}, {}, {}, {}
        for tick in range(native.num_slots):
            tasks = []
            for rank in range(native.world):
                cell = native.task(tick, rank)
                if cell is None:
                    continue
                task = BlockTask(tick, rank, KVKey(*cell))
                self.by_key[task.key] = task
                tasks.append(task)
                selected = tuple(KVKey(*source.version) for source in native.sources(cell, rank))
                self.reads[task] = selected
                for key in selected:
                    # Iteration in native execution order records the last
                    # reader on each rank, for device-side release tickets.
                    self.destinations.setdefault(key, {})[rank] = cell
            self.by_tick[tick] = tuple(tasks)
        for key, destinations in self.destinations.items():
            producer = self.by_key[key]
            if any(native.task(producer.tick, rank) is None for rank in destinations):
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
