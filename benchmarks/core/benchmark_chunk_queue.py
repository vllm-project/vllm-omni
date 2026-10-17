# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU microbenchmark for async-chunk queue partitioning.

Run from an installed checkout:
    python benchmarks/core/benchmark_chunk_queue.py --sizes 4 128 1024 8192

This measures the coordinator method only, excluding request construction.
It does not measure model inference or end-to-end serving throughput.
"""

import argparse
import json
import statistics
import time
from collections import deque

from vllm import SamplingParams
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.request import OmniRequest


def benchmark(size: int, move_half: bool, repeats: int, warmup: int) -> dict:
    samples = []
    for iteration in range(warmup + repeats):
        requests = [
            OmniRequest(
                request_id=str(index),
                external_req_id=str(index),
                prompt_token_ids=[1, 2],
                sampling_params=SamplingParams(max_tokens=4),
                pooling_params=None,
            )
            for index in range(size)
        ]
        for request in requests:
            request.status = RequestStatus.RUNNING
        ready = {str(index) for index in range(size) if not move_half or index % 2 == 0}
        coordinator = OmniSchedulingCoordinator(scheduler_max_num_seqs=size, stage_id=1, async_chunk=True)
        parked = deque()
        queue = list(requests)
        started = time.perf_counter_ns()
        coordinator._process_chunk_queue(queue, parked, RequestStatus.RUNNING, ready)
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        assert [request.request_id for request in queue] == [
            request.request_id for request in requests if request.request_id in ready
        ]
        assert [request.request_id for request in parked] == [
            request.request_id for request in requests if request.request_id not in ready
        ]
        if iteration >= warmup:
            samples.append(elapsed_us)
    return {
        "size": size,
        "scenario": "half_move" if move_half else "no_move",
        "warmup": warmup,
        "repeats": repeats,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 128, 1024, 8192])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("sizes and repeats must be positive; warmup must be nonnegative")
    for size in args.sizes:
        for move_half in (False, True):
            print(json.dumps(benchmark(size, move_half, args.repeats, args.warmup)), flush=True)


if __name__ == "__main__":
    main()
