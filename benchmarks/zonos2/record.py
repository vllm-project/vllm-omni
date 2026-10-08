# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark-only timing/profiler observers; no sampling or logit substitution."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def install():
    import torch

    from vllm_omni.model_executor.models.zonos2.zonos2_dac_decoder import Zonos2Code2WavForConditionalGeneration as Dac
    from vllm_omni.model_executor.models.zonos2.zonos2_talker import Zonos2TalkerForConditionalGeneration as Talker

    if getattr(Talker, "_p6_recording", False):
        return
    Talker._p6_recording = True
    root = Path(os.environ["ZONOS2_BENCH_RECORD_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    cuda = torch.get_device_module("cuda")
    profiling = os.environ.get("ZONOS2_BENCH_PROFILE") == "1"

    def write(row):
        with (root / f"record-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")

    def maybe_profile(model, role):
        if not profiling or getattr(model, "_p6_profile_done", False):
            return
        if not hasattr(model, "_p6_profiler"):
            model._p6_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            )
            model._p6_profiler.start()
            model._p6_profile_count = 0

    def profile_step(model, role, limit):
        if not profiling or getattr(model, "_p6_profile_done", False) or not hasattr(model, "_p6_profiler"):
            return
        model._p6_profiler.step()
        model._p6_profile_count += 1
        if model._p6_profile_count >= limit:
            model._p6_profiler.stop()
            model._p6_profiler.export_chrome_trace(str(root / f"{role}.trace.json"))
            rows = [
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cpu_us": event.self_cpu_time_total,
                    "self_device_us": event.self_device_time_total,
                }
                for event in model._p6_profiler.key_averages()
            ]
            (root / f"{role}.profile.json").write_text(json.dumps(rows, indent=2))
            model._p6_profile_done = True

    forward = Talker.forward

    def ar_forward(self, *args, **kwargs):
        infos = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        if not any(info.get("_omni_req_id") for info in infos):
            return forward(self, *args, **kwargs)
        if not all(str(info.get("_omni_req_id", "")).startswith("warmup") for info in infos):
            maybe_profile(self, "ar")
        a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        a.record()
        start = time.perf_counter()
        with torch.profiler.record_function("zonos2::AR_forward"):
            result = forward(self, *args, **kwargs)
        b.record()
        if not hasattr(self, "_p6_events"):
            self._p6_events = []
        self._p6_events.append((a, b, (time.perf_counter() - start) * 1000, len(self._sampling_plan)))
        profile_step(self, "ar", 32)
        return result

    Talker.forward = ar_forward
    sample = Talker.sample

    def sampled(self, logits, metadata):
        start = time.perf_counter()
        result = sample(self, logits, metadata)
        if not self._sampling_plan:
            return result
        for index, (key, eligible, info) in enumerate(self._sampling_plan):
            if eligible and len(self._request_states[key].history) == 1:
                # Timestamp an actually available CPU frame, once per request.
                self._request_states[key].history.detach().cpu()
                write({"kind": "first_code", "request": key, "time": time.perf_counter()})
        if not hasattr(self, "_p6_sample_ms"):
            self._p6_sample_ms = []
        self._p6_sample_ms.append((time.perf_counter() - start) * 1000)
        return result

    Talker.sample = sampled
    cleanup = Talker.on_requests_finished

    def finished(self, keys):
        keys = list(keys)
        for key in keys:
            state = self._request_states.get(str(key))
            if state is not None:
                codes = state.history.detach().cpu()
                torch.save(codes, root / f"{key}.pt")
                write(
                    {
                        "kind": "finish",
                        "request": str(key),
                        "frames": len(codes),
                        "eos_frame": int(state.eos_frame),
                        "time": time.perf_counter(),
                    }
                )
        if getattr(self, "_p6_events", []):
            self._p6_events[-1][1].synchronize()
            write(
                {
                    "kind": "ar_timings",
                    "requests": [str(key) for key in keys],
                    "gpu_ms": [a.elapsed_time(b) for a, b, wall, n in self._p6_events],
                    "host_ms": [wall for a, b, wall, n in self._p6_events],
                    "batch_sizes": [n for a, b, wall, n in self._p6_events],
                    "sample_host_ms": self._p6_sample_ms,
                }
            )
            self._p6_events.clear()
            self._p6_sample_ms.clear()
        cleanup(self, keys)

    Talker.on_requests_finished = finished
    decode = Dac.forward

    def dac_forward(self, *args, **kwargs):
        if not kwargs.get("request_ids"):
            return decode(self, *args, **kwargs)
        if not all(str(key).startswith("warmup") for key in kwargs["request_ids"]):
            maybe_profile(self, "dac")
        a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        a.record()
        start = time.perf_counter()
        with torch.profiler.record_function("zonos2::DAC_forward"):
            result = decode(self, *args, **kwargs)
        b.record()
        b.synchronize()
        for index, key in enumerate(kwargs.get("request_ids", [])):
            wav = result.multimodal_outputs["model_outputs"][index]
            write(
                {
                    "kind": "dac",
                    "request": key,
                    "time": time.perf_counter(),
                    "samples": wav.numel(),
                    "gpu_ms": a.elapsed_time(b),
                    "host_ms": (time.perf_counter() - start) * 1000,
                }
            )
        profile_step(self, "dac", 4)
        return result

    Dac.forward = dac_forward
