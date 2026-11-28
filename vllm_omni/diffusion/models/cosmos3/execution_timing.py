# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local worker stages. Host wall durations, optionally synchronized.

Flat float fields travel through the existing VideoAction.policy_timing schema.
No per-step prints and no changes to tensors, sampler or stream ordering.
"""

import functools
import itertools
import json
import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_TRACE_SEQUENCE = itertools.count(1)

_CURRENT = ContextVar("cosmos3_execution_timing", default=None)


class WorkerTiming:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.synchronized = os.environ.get("COSMOS3_TIMING_SYNC", "0") == "1"
        self.rows = defaultdict(lambda: [0, 0.0, 0.0])
        self.stack = []
        self.trace_id = next(_TRACE_SEQUENCE)
        self.events = []
        self.sequence = 0
        self.request_id = None

    def synchronize(self):
        if not self.synchronized:
            return
        import torch

        device = torch.device(self.pipeline.device)
        if device.type == "npu":
            torch.npu.synchronize(device)
        elif device.type == "cuda":
            torch.accelerator.synchronize(device)

    @contextmanager
    def span(self, name):
        # The trailing sync belongs to this stage. Pending prior work at entry
        # belongs to its parent rather than being mislabelled as this stage.
        self.synchronize()
        self.sequence += 1
        identifier = self.sequence
        parent_id = self.stack[-1][2] if self.stack else None
        frame = [time.perf_counter(), 0.0, identifier]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.rows[name]
            row[0] += 1
            row[1] += elapsed * 1000
            row[2] += max(0, elapsed - frame[1]) * 1000
            self.events.append(
                dict(
                    span_id=identifier,
                    parent_id=parent_id,
                    stage=name,
                    start_s=frame[0],
                    end_s=frame[0] + elapsed,
                    inclusive_s=elapsed,
                    exclusive_s=max(0, elapsed - frame[1]),
                )
            )

    def flattened(self):
        result = {
            "exec.synchronized": float(self.synchronized),
            "exec.trace_id": float(self.trace_id),
            "exec.pid": float(os.getpid()),
        }
        for name, (calls, inclusive, exclusive) in self.rows.items():
            result.update(
                {
                    f"exec.{name}.calls": float(calls),
                    f"exec.{name}.inclusive_ms": inclusive,
                    f"exec.{name}.exclusive_ms": exclusive,
                }
            )
        return result

    def emit(self):
        destination = os.environ.get("COSMOS3_WORKER_TIMELINE")
        if not destination:
            return
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(
            schema="cosmos3_worker_timeline_v1",
            pid=os.getpid(),
            trace_id=self.trace_id,
            request_id=self.request_id,
            host=socket.gethostname(),
            clock="CLOCK_MONOTONIC perf_counter",
            clock_domain=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            synchronized=self.synchronized,
            events=self.events,
        )
        # One append per completed inference, no per-denoise-step filesystem calls.
        with path.open("a") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def timed_startup(name):
    def decorate(function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            destination = os.environ.get("COSMOS3_WORKER_STARTUP_TIMELINE")
            if not destination:
                return function(*args, **kwargs)
            started = time.perf_counter()
            outcome = "complete"
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                outcome = type(exc).__name__
                raise
            finally:
                ended = time.perf_counter()
                path = Path(destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                row = dict(
                    stage=name,
                    pid=os.getpid(),
                    start_s=started,
                    end_s=ended,
                    elapsed_s=ended - started,
                    outcome=outcome,
                    host="Huawei",
                    included_in_closed_loop=False,
                )
                fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                try:
                    os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())
                finally:
                    os.close(fd)

        return measured

    return decorate


def timed_worker(name):
    def decorate(function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            timing = _CURRENT.get()
            if timing is None:
                return function(*args, **kwargs)
            with timing.span(name):
                return function(*args, **kwargs)

        return measured

    return decorate


def instrument_pipeline(cls):
    """Apply timing to the native methods while preserving the class identity."""
    stages = {
        "_build_robolab_policy_inputs": "request_inputs",
        "_format_and_tokenize_prompts": "text_tokenize",
        "_prepare_action_latents": "action_latents",
        "_prepare_latents_action_video": "video_condition_latents",
        "_prepare_latents_i2v": "image_condition_latents",
        "_prepare_latents": "noise_latents",
        "_encode_video_tensor": "vae.encode_video",
        "_encode_conditioning_image_latent": "vae.encode_image",
        "diffuse": "denoise",
        "_decode_latents": "vae.decode_video",
    }
    for name, label in stages.items():
        if hasattr(cls, name):
            setattr(cls, name, timed_worker(label)(getattr(cls, name)))
    original_init = cls.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # __call__ still goes through PyTorch hooks; only the forward body is
        # observed. Text UND/KV setup stays in the first transformer call.
        self.transformer.forward = timed_worker("transformer")(self.transformer.forward)

    cls.__init__ = timed_startup("pipeline.initialize")(initialize)
    if hasattr(cls, "load_weights"):
        cls.load_weights = timed_startup("pipeline.load_weights")(cls.load_weights)
    original_forward = cls.forward

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if os.environ.get("COSMOS3_EXECUTION_TIMING", "1") == "0":
            return original_forward(self, *args, **kwargs)
        timing = WorkerTiming(self)
        request = args[0] if args else kwargs.get("req")
        timing.request_id = str(getattr(request, "request_id", ""))
        token = _CURRENT.set(timing)
        try:
            with timing.span("forward"):
                result = original_forward(self, *args, **kwargs)
            output = getattr(result, "output", None)
            if isinstance(output, dict):
                metadata = output.get("metadata", {}).get("actions")
                if isinstance(metadata, dict) and metadata.get("policy_profile") in {
                    "libero10_native_nonstream_v1",
                    "libero10_native_constant_v1",
                }:
                    metadata.setdefault("policy_timing", {}).update(timing.flattened())
            return result
        finally:
            timing.emit()
            _CURRENT.reset(token)

    cls.forward = forward
    return cls
