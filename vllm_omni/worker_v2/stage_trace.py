# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional asynchronous stage timing for the speech pipeline."""

import json
import os
from collections import deque
from functools import wraps
from time import monotonic_ns

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
TRACE = os.getenv("MINICPMO_C8_TRACE_FILE")


def enabled():
    return TRACE is not None and os.path.isfile(TRACE)


def boundary(kind, **fields):
    logger.info("[c8-boundary] %s", json.dumps({"kind": kind, "ns": monotonic_ns(), **fields}))


def emit_ready(queue):
    # The trace is observational. Query completed stream work and leave
    # unfinished events until the following host step.
    while queue and queue[0][2].query():
        row, start, end = queue.popleft()
        row["stream_ms"] = start.elapsed_time(end)
        logger.info("[c8-runner] %s", json.dumps(row))


def timed(phase):
    def decorate(fn):
        @wraps(fn)
        def wrapped(runner, *args, **kwargs):
            if not enabled():
                return fn(runner, *args, **kwargs)
            queue = getattr(runner, "_c8_trace_queue", None)
            if queue is None:
                queue = runner._c8_trace_queue = deque()
            emit_ready(queue)
            if phase == "execute":
                scheduler = args[0] if args else kwargs["scheduler_output"]
                scheduled = scheduler.num_scheduled_tokens
                if not scheduled or kwargs.get("dummy_run"):
                    return fn(runner, *args, **kwargs)
                request_ids = list(scheduled)
                tokens = sum(scheduled.values())
                prefill = None
            else:
                batch = (
                    (args[1] if len(args) > 1 else kwargs.get("input_batch"))
                    if phase == "forward"
                    else (runner.execute_model_state.input_batch if runner.execute_model_state is not None else None)
                )
                if batch is None or getattr(batch, "is_dummy", False):
                    return fn(runner, *args, **kwargs)
                request_ids = list(batch.req_ids)
                tokens = batch.num_tokens
                prefill = batch.has_prefill if hasattr(batch, "has_prefill") else None
            if not request_ids:
                return fn(runner, *args, **kwargs)
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            started = monotonic_ns()
            start.record()
            out = fn(runner, *args, **kwargs)
            end.record()
            row = {
                "stage": runner.model_config.model_stage,
                "phase": phase,
                "requests": request_ids,
                "tokens": int(tokens),
                "prefill": None if prefill is None else bool(prefill),
                "start_ns": started,
                "returned_ns": monotonic_ns(),
            }
            queue.append((row, start, end))
            return out

        return wrapped

    return decorate
