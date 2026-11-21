# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in rank-zero CPU profiles and frozen conditions for Realization latency diagnosis."""

import cProfile
import json
import os
import time
from dataclasses import replace
from functools import wraps
from itertools import count
from pathlib import Path

import torch

_DIRECTORY = os.environ.get("DUO_CPU_PROFILE")
_COUNTER = count()


def profile_phase(function):
    @wraps(function)
    def measured(*args, **kwargs):
        if _DIRECTORY is None or torch.distributed.get_rank() != 0:
            return function(*args, **kwargs)
        root = Path(_DIRECTORY)
        root.mkdir(parents=True, exist_ok=True)
        index = next(_COUNTER)
        stem = f"{os.getpid()}.{index:04d}.{function.__name__}"
        profiler = cProfile.Profile()
        started = time.perf_counter_ns()
        result = profiler.runcall(function, *args, **kwargs)
        finished = time.perf_counter_ns()
        profiler.dump_stats(str(root / f"{stem}.pstats"))
        event = dict(
            name=function.__qualname__,
            object_id=id(args[0]),
            started_ns=started,
            finished_ns=finished,
            wall_ms=(finished - started) / 1e6,
            profile=f"{stem}.pstats",
        )
        with (root / "phases.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")
        return result

    return measured


def capture_condition(condition):
    if _DIRECTORY is None or torch.distributed.get_rank() != 0:
        return
    root = Path(_DIRECTORY)
    root.mkdir(parents=True, exist_ok=True)
    condition = replace(
        condition,
        sequence=condition.sequence.to("cpu"),
        prefix=None if condition.prefix is None else condition.prefix.to("cpu"),
    )
    torch.save(condition, root / f"condition_{condition.block:03d}.pt")
