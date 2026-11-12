# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure healthy Thinker requests during real Mooncake lookup/KV-put stalls.

Run with a running Store and two GPUs, from the repository root::

    VLLM_BATCH_INVARIANT=1 python -m benchmarks.qwen3_omni.mooncake_faults \
      --model /path/to/Qwen3-Omni --store-config /path/to/store.json \
      --output-dir results/mooncake-faults

This uses the normal MRv2 deployment and connector, requesting text outputs
to isolate Thinker scheduling. Delays precede actual Store calls; no Store
results or model outputs are mocked. Tensor dumps and save barriers are absent.
Full text/audio parity is covered separately by the turn-consistency test.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from vllm_omni.worker.gpu_ar_worker import GPUARWorker

REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


class _Probe:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.kind = ""
        self.delay = 0.0
        self.started_path: Path | None = None
        self.lock = threading.Lock()

    def record(self, event: str, **fields) -> None:
        self.events.append({"event": event, "time_ns": time.monotonic_ns(), **fields})

    def stall(self, kind: str) -> Path | None:
        with self.lock:
            if self.kind != kind:
                return
            self.kind = ""
            delay, path = self.delay, self.started_path
        self.record("fault_start", kind=kind, delay_seconds=delay)
        assert path is not None
        path.write_text("started\n")
        time.sleep(delay)
        self.record("fault_end", kind=kind)
        return path


class _StoreProxy:
    def __init__(self, store, probe: _Probe):
        self.store, self.probe = store, probe

    def __getattr__(self, name):
        return getattr(self.store, name)

    def batch_put_from_multi_buffers(self, *args, **kwargs):
        path = self.probe.stall("kv_put")
        result = self.store.batch_put_from_multi_buffers(*args, **kwargs)
        self.probe.record("native_kv_put", keys=len(args[0]))
        if path is not None:
            path.with_suffix(".completed").write_text("completed\n")
        return result


class FaultProbeWorker(GPUARWorker):
    """Keep small timing records in memory and expose them after each trial."""

    def init_device(self):
        from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
        from vllm_omni.distributed.kv_transfer.mooncake.connector import (
            JointStoreWorker,
            OmniMooncakeStoreConnector,
        )

        self.probe = probe = _Probe()
        initialize = JointStoreWorker.__init__
        lookup = JointStoreWorker.lookup
        schedule = OmniARScheduler.schedule
        capture = OmniMooncakeStoreConnector.capture_step
        read = OmniMooncakeStoreConnector._read_hs

        def initialize_store(worker, *args, **kwargs):
            initialize(worker, *args, **kwargs)
            worker.store = _StoreProxy(worker.store, probe)

        def lookup_store(worker, *args, **kwargs):
            path = probe.stall("lookup")
            result = lookup(worker, *args, **kwargs)
            probe.record("native_lookup", hit_tokens=result.hit_length)
            if path is not None:
                path.with_suffix(".completed").write_text("completed\n")
            return result

        def scheduled(scheduler, *args, **kwargs):
            started = time.monotonic_ns()
            result = schedule(scheduler, *args, **kwargs)
            finished = time.monotonic_ns()
            for req_id, count in result.num_scheduled_tokens.items():
                probe.record("scheduled", request_id=req_id, tokens=count, duration_ns=finished - started)
            return result

        def captured(connector, req_id, attempt_id, start_token, prompt_len, tensors):
            result = capture(connector, req_id, attempt_id, start_token, prompt_len, tensors)
            if result is not None:
                probe.record(
                    "capture",
                    request_id=req_id,
                    restored_rows=result["embedding"].shape[0] - tensors["embedding"].shape[0],
                )
            return result

        def read_hs(connector, attempt, tokens):
            result = read(connector, attempt, tokens)
            probe.record("hs_read", request_id=attempt.req_id, tokens=tokens)
            return result

        JointStoreWorker.__init__ = initialize_store
        JointStoreWorker.lookup = lookup_store
        OmniARScheduler.schedule = scheduled
        OmniMooncakeStoreConnector.capture_step = captured
        OmniMooncakeStoreConnector._read_hs = read_hs
        super().init_device()

    def fault_probe_arm(self, kind: str, delay: float, started_path: str) -> bool:
        self.probe.events.clear()
        self.probe.kind = kind
        self.probe.delay = delay
        self.probe.started_path = Path(started_path)
        return True

    def fault_probe_snapshot(self) -> dict:
        return {"events": list(self.probe.events)}


def _snapshot(result) -> list[dict]:
    if isinstance(result, dict) and "events" in result:
        return result["events"]
    if isinstance(result, list):
        return [event for item in result for event in _snapshot(item)]
    return []


async def _request(engine, label: str, prompt: dict, tokens: int, first: asyncio.Event | None = None) -> dict:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0,
        seed=42,
        max_tokens=tokens,
        min_tokens=tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    started = time.monotonic_ns()
    times, token_ids = [], []
    async for output in engine.generate(
        prompt,
        request_id=label,
        sampling_params_list=[params, *engine.default_sampling_params_list[1:]],
        output_modalities=["text"],
    ):
        if output.error:
            raise RuntimeError(output.error)
        if output.final_output_type == "text" and output.outputs and output.outputs[0].token_ids:
            times.append(time.monotonic_ns())
            token_ids.extend(output.outputs[0].token_ids)
            if first is not None:
                first.set()
    if not times:
        raise AssertionError(f"No generated text for {label}")
    assert len(token_ids) == tokens, f"Expected {tokens} delta token IDs for {label}, got {len(token_ids)}"
    return {
        "label": label,
        "started_ns": started,
        "token_times_ns": times,
        "token_ids": token_ids,
        "ttft_ms": (times[0] - started) / 1e6,
        "e2e_ms": (time.monotonic_ns() - started) / 1e6,
    }


async def _wait_file(path: Path, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Fault was not reached: {path}")
        await asyncio.sleep(0.01)


def _analyze(trial: dict, events: list[dict]) -> None:
    starts = [event for event in events if event["event"] == "fault_start"]
    ends = [event for event in events if event["event"] == "fault_end"]
    assert len(starts) == len(ends) == 1, "Expected one actual injected call"
    begin, end = starts[0]["time_ns"], ends[0]["time_ns"]
    trial["fault_start_ns"], trial["fault_end_ns"] = begin, end
    trial["events"] = events
    for request in trial["requests"]:
        scheduled = [e for e in events if e["event"] == "scheduled" and request["label"] in e["request_id"]]
        assert scheduled, f"No scheduler observations for {request['label']}"
        request["first_observed_schedule_ms"] = (scheduled[0]["time_ns"] - request["started_ns"]) / 1e6
        request["schedule_call_max_ms"] = max(e["duration_ns"] for e in scheduled) / 1e6
        request["scheduled_during_stall"] = sum(begin <= e["time_ns"] < end for e in scheduled)
        request["restored_rows"] = sum(
            e["restored_rows"] for e in events if e["event"] == "capture" and request["label"] in e["request_id"]
        )
    if trial["delay_seconds"]:
        ongoing, _, healthy = trial["requests"]
        assert healthy["token_times_ns"][0] < end, "Healthy request waited for the delayed Store call"
        if trial["kind"] == "lookup":
            assert ongoing["scheduled_during_stall"] > 0, "Existing decode did not progress during lookup"
        else:
            assert healthy["restored_rows"] > 0, "Healthy request did not exercise the independent HS read path"
            assert any(
                e["event"] == "hs_read" and healthy["label"] in e["request_id"] and begin < e["time_ns"] < end
                for e in events
            ), "Healthy HS read did not complete while native KV put was delayed"


async def _run(args) -> None:
    from tests.engine.kv_offload_helpers import prompts, stage_overrides
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MOONCAKE_CONFIG_PATH"] = str(args.store_config.resolve())
    overrides = stage_overrides(
        lmcache=False,
        thinker_extra={
            "worker_cls": "benchmarks.qwen3_omni.mooncake_faults.FaultProbeWorker",
            "use_v2_model_runner": True,
            "tensor_parallel_size": 1,
            "max_num_seqs": 4,
            "max_model_len": 2048,
            "max_num_batched_tokens": 256,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "block_size": 16,
            "gpu_memory_utilization": 0.85,
            "revision": REVISION,
            "omni_kv_config": {
                "kv_store_config": {
                    "mooncake_config": {
                        "config_file": str(args.store_config.resolve()),
                        "namespace": "fault-probe-" + uuid4().hex,
                        "model_revision": REVISION,
                        "load_timeout_seconds": args.lookup_timeout,
                    }
                }
            },
        },
        downstream_extra={
            "1": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.3, "max_num_seqs": 4},
            "2": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.2, "max_num_seqs": 4},
        },
    )
    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": REVISION,
        "stage_overrides": overrides,
        "trials": [],
        "passed": False,
        "scope": "Real Store + Qwen3-Omni Thinker text output; audio parity validated separately",
        "packages": {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
    }
    engine = AsyncOmni(
        model=args.model,
        revision=REVISION,
        deploy_config="qwen3_omni_moe_mrv2.yaml",
        stage_overrides=overrides,
        trust_remote_code=True,
        stage_init_timeout=1800,
        init_timeout=1800,
    )
    try:
        healthy_prompt = prompts(1)[0]
        for warmup in range(2):
            await _request(engine, f"warmup-{warmup}", healthy_prompt, 48)
        for kind in ("lookup", "kv_put"):
            for repeat in range(args.repeats):
                for delay in (0.0, args.delay):
                    case = f"{kind}-{repeat}-{'delay' if delay else 'control'}"
                    first = asyncio.Event()
                    ongoing = asyncio.create_task(_request(engine, case + "-ongoing", healthy_prompt, 256, first))
                    await asyncio.wait_for(first.wait(), timeout=60)
                    started_path = output / f"{case}.started"
                    await engine.collective_rpc(
                        "fault_probe_arm", args=(kind, delay, str(started_path)), stage_ids=[0], timeout=30
                    )
                    writer_prompt = {"prompt": healthy_prompt["prompt"].replace("user\n", f"user\nTrial {case}: ", 1)}
                    writer = asyncio.create_task(_request(engine, case + "-writer", writer_prompt, 48))
                    await _wait_file(started_path)
                    if kind == "lookup":
                        # Enter after the first lookup's deadline: this request
                        # must take the bypass while the old RPC still drains.
                        await asyncio.sleep(args.lookup_timeout + 0.1)
                    healthy = asyncio.create_task(_request(engine, case + "-healthy", healthy_prompt, 48))
                    requests = await asyncio.wait_for(asyncio.gather(ongoing, writer, healthy), timeout=180)
                    # Only snapshot after the injected operation completes.
                    await _wait_file(started_path.with_suffix(".completed"), timeout=delay + 30)
                    events = _snapshot(await engine.collective_rpc("fault_probe_snapshot", stage_ids=[0], timeout=30))
                    trial = {
                        "kind": kind,
                        "repeat": repeat,
                        "delay_seconds": delay,
                        "requests": requests,
                        "events": events,
                    }
                    report["trials"].append(trial)
                    _analyze(trial, events)
                    if delay:
                        control = report["trials"][-2]
                        for index in (0, 2):
                            assert requests[index]["token_ids"] == control["requests"][index]["token_ids"], (
                                f"Fault changed healthy output: {case}, request {index}"
                            )
                    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(case, "healthy TTFT ms", requests[2]["ttft_ms"], flush=True)
        report["summary"] = []
        for kind in ("lookup", "kv_put"):
            for delay in (0.0, args.delay):
                trials = [t for t in report["trials"] if t["kind"] == kind and t["delay_seconds"] == delay]
                values = [t["requests"][2]["ttft_ms"] for t in trials]
                scheduled = [t["requests"][2]["first_observed_schedule_ms"] for t in trials]
                report["summary"].append(
                    {
                        "kind": kind,
                        "delay_seconds": delay,
                        "healthy_ttft_mean_ms": statistics.mean(values),
                        "healthy_ttft_min_ms": min(values),
                        "healthy_ttft_max_ms": max(values),
                        "repeats": len(values),
                        "healthy_first_schedule_mean_ms": statistics.mean(scheduled),
                        "healthy_first_schedule_min_ms": min(scheduled),
                        "healthy_first_schedule_max_ms": max(scheduled),
                    }
                )
        report["passed"] = True
    finally:
        engine.shutdown()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--store-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--lookup-timeout", type=float, default=1.0)
    args = parser.parse_args()
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        parser.error("set VLLM_BATCH_INVARIANT=1 before startup for healthy-output comparisons")
    if args.delay <= args.lookup_timeout + 0.1 or args.repeats < 1:
        parser.error("require repeats >= 1 and delay > lookup-timeout + 0.1")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
