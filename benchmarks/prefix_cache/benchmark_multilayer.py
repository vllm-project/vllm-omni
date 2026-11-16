# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matched local/external prefix-cache serving experiments.

Launch one mode per invocation with the same arguments and idle GPUs. The model
and deployment are explicit: this harness measures the full audio pipeline and
requires a deployment supported by the Mooncake KV/HS connector for external
modes. Run independent repetitions in alternating mode order. JSON results,
server logs, configuration, metrics and provenance remain in --output-dir.

Example (with a compatible environment and Mooncake installed)::

    python benchmarks/prefix_cache/benchmark_multilayer.py \
        --model /path/to/checkpoint --model-revision CHECKPOINT_SHA \
        --deploy-config /path/to/deploy.yaml --mode two_level \
        --output-dir /tmp/cache-experiment --num-gpu-blocks 512

The hot workload fits the local cache. The overflow workload cycles through
more blocks than local capacity, while retaining them in Mooncake. Configure
--store-size to repeat with a working set larger than external capacity.
The mixed workload alternates one frequently used document with each overflow
document, using identical questions after populating the Store and priming the
hot document. Choose local capacity that retains the hot document but cannot
retain the complete overflow working set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import yaml
from transformers import AutoTokenizer


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def fetch(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def stop(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    # Every server owns a fresh process group, including its stage workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def documents(tokenizer, count: int, length: int, first: int) -> list[str]:
    result = []
    for index in range(first, first + count):
        rng = random.Random(index)
        parts = [f"Document {index} with catalogue identifier {rng.randrange(10**9)}."]
        # Different early content prevents cross-document full-block reuse.
        while len(tokenizer.encode(" ".join(parts))) < length + 32:
            parts.extend(
                f"Entry {rng.randrange(100000)}: {rng.choice(('Mercury', 'Venus', 'Earth', 'Mars'))} "
                f"has catalogue value {rng.randrange(10000)} and observation number {rng.randrange(10000)}."
                for _ in range(32)
            )
        text = tokenizer.decode(tokenizer.encode(" ".join(parts))[:length])
        result.append(text)
    return result


def write_prompts(path: Path, docs: list[str], round_id: int, output_tokens: int) -> int:
    rows = [
        {
            "prompt": f"{doc}\nQuestion {round_id}: briefly describe the first entry in this document.",
            "output_tokens": output_tokens,
        }
        for doc in docs
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--source-revision", help="Revision of an archived source tree without .git")
    parser.add_argument("--mode", choices=("off", "local", "mooncake", "two_level"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpu-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--hot-documents", type=int, default=4)
    parser.add_argument("--overflow-documents", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--audio-output-tokens", type=int, default=48)
    parser.add_argument("--store-size", default="8GB")
    parser.add_argument("--max-hs-bytes", type=int, default=268435456)
    parser.add_argument(
        "--batch-invariant", action="store_true", help="Use batch-invariant kernels in every compared mode"
    )
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--workloads", nargs="+", choices=("hot", "overflow", "mixed"), default=["hot", "overflow"])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env.pop("VLLM_BATCH_INVARIANT", None)
    if args.batch_invariant:
        env["VLLM_BATCH_INVARIANT"] = "1"
    port, master_port, metadata_port, metrics_port = free_ports(4)
    external = args.mode in ("mooncake", "two_level")
    stage = {
        "stage_id": 0,
        "enable_prefix_caching": args.mode in ("local", "two_level"),
        "max_model_len": max(2048, args.prefix_tokens + args.output_tokens + 512),
        "block_size": args.block_size,
        "num_gpu_blocks_override": args.num_gpu_blocks,
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 2048,
    }
    if external:
        store = {
            "metadata_server": f"http://127.0.0.1:{metadata_port}/metadata",
            "master_server_address": f"127.0.0.1:{master_port}",
            "protocol": "tcp",
            "device_name": "",
            "mode": "embedded",
            "global_segment_size": args.store_size,
            "local_buffer_size": "1GB",
            "enable_offload": False,
        }
        (out / "store.json").write_text(json.dumps(store, indent=2))
        stage["omni_kv_config"] = {
            "kv_store_config": {
                "mooncake_config": {
                    "config_file": str(out / "store.json"),
                    "namespace": f"multilayer-{uuid4().hex}",
                    "model_revision": args.model_revision,
                    "max_hs_bytes": args.max_hs_bytes,
                }
            }
        }
    deploy = {
        "base_config": str(args.deploy_config.resolve()),
        "stages": [
            stage,
            {
                "stage_id": 1,
                "enable_prefix_caching": False,
                "default_sampling_params": {
                    "temperature": 0.0,
                    "top_k": -1,
                    "top_p": 1.0,
                    "seed": 42,
                    "max_tokens": args.audio_output_tokens,
                },
                "subtalker_sampling_params": {"do_sample": False},
            },
        ],
    }
    (out / "deploy.yaml").write_text(yaml.safe_dump(deploy))
    provenance = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance.update(host=socket.gethostname(), started_unix=time.time(), python=sys.version)
    for key, command in (
        ("revision", ["git", "rev-parse", "HEAD"]),
        ("diff", ["git", "diff", "--stat"]),
        ("gpus", ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"]),
    ):
        result = subprocess.run(command, cwd=source, text=True, capture_output=True)
        provenance[key] = result.stdout if result.returncode == 0 else result.stderr
    if args.source_revision:
        provenance["revision"] = args.source_revision
    model_directory = Path(args.model)
    if model_directory.is_dir():
        provenance["checkpoint_metadata_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in model_directory.glob("*.json")
        }
        provenance["weight_files"] = {
            path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in model_directory.glob("*.safetensors")
        }
    provenance["packages"] = sorted(
        f"{package.metadata['Name']}=={package.version}" for package in importlib.metadata.distributions()
    )
    provenance["environment"] = {
        key: env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_USE_FLASHINFER_SAMPLER")
    }
    patch = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=source, text=True, capture_output=True)
    if patch.returncode == 0:
        (out / "source.diff").write_text(patch.stdout)
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    workloads = {
        name: documents(tokenizer, count, args.prefix_tokens, first)
        for name, count, first in (("hot", args.hot_documents, 100), ("overflow", args.overflow_documents, 1000))
        if name in args.workloads
    }
    if "mixed" in args.workloads:
        hot = documents(tokenizer, 1, args.prefix_tokens, 100)
        cold = documents(tokenizer, args.overflow_documents, args.prefix_tokens, 1000)
        workloads["mixed"] = [document for other in cold for document in (hot[0], other)]
    master = server = None
    handles = []
    try:
        if external:
            handle = (out / "master.log").open("w")
            handles.append(handle)
            master = subprocess.Popen(
                [
                    "mooncake_master",
                    f"--rpc_port={master_port}",
                    "--enable_http_metadata_server=true",
                    "--http_metadata_server_host=127.0.0.1",
                    f"--http_metadata_server_port={metadata_port}",
                    f"--metrics_port={metrics_port}",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            time.sleep(2)
            if master.poll() is not None:
                raise RuntimeError("Mooncake master failed; see master.log")
        handle = (out / "server.log").open("w")
        handles.append(handle)
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "serve",
                args.model,
                "--omni",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--deploy-config",
                str(out / "deploy.yaml"),
                "--revision",
                args.model_revision,
                "--init-timeout",
                str(int(args.startup_timeout)),
                "--stage-init-timeout",
                str(int(args.startup_timeout)),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=source,
            env=env,
            start_new_session=True,
        )
        started = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError("Server exited; see server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                    break
            except OSError:
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError("Server startup timed out")
                time.sleep(3)
        print(f"{args.mode}: ready after {time.monotonic() - started:.1f}s", flush=True)

        def bench(name: str, docs: list[str], round_id: int, concurrency: int) -> None:
            data = out / f"{name}.jsonl"
            count = write_prompts(data, docs, round_id, args.output_tokens)
            fetch(f"http://127.0.0.1:{port}/metrics", out / f"{name}-metrics-before.txt")
            command = [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "bench",
                "serve",
                "--omni",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                args.model,
                "--backend",
                "openai-chat-omni",
                "--endpoint",
                "/v1/chat/completions",
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(data),
                "--disable-shuffle",
                "--skip-chat-template",
                "--custom-output-len",
                str(args.output_tokens),
                "--num-prompts",
                str(count),
                "--max-concurrency",
                str(concurrency),
                "--request-rate",
                "inf",
                "--ignore-eos",
                "--ready-check-timeout-sec",
                "0",
                "--seed",
                "0",
                "--percentile-metrics",
                "ttft,e2el,audio_ttfp,audio_rtf",
                "--metric-percentiles",
                "50,95",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(out),
                "--result-filename",
                f"{name}.json",
            ]
            with (out / f"{name}.log").open("w") as log:
                subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((out / f"{name}.json").read_text())
            if result["completed"] != count or any(result.get("errors", [])):
                raise RuntimeError(f"{name}: benchmark requests failed")
            # Scheduler telemetry is throttled to one emission per second.
            # Drive a fresh step after that interval, outside the timed run,
            # so the completed workload's buffered records can be exported.
            # A private salt guarantees this probe adds no local/external hits.
            time.sleep(1.1)
            probe = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": args.model,
                        "messages": [{"role": "user", "content": "Say yes."}],
                        "max_tokens": 1,
                        "temperature": 0,
                        "modalities": ["text"],
                        "cache_salt": uuid4().hex,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(probe, timeout=60) as response:
                (out / f"{name}-drain.json").write_bytes(response.read())
            fetch(f"http://127.0.0.1:{port}/metrics", out / f"{name}-metrics.txt")
            if external:
                fetch(f"http://127.0.0.1:{metrics_port}/metrics", out / f"{name}-store-metrics.txt")
            print(f"{args.mode}: {name} completed={count} ttft={result.get('median_ttft_ms')}", flush=True)

        bench("warmup", ["Name a colour of the sky."] * 2, 0, 1)
        # Warm the measured prefill and concurrent sampling shapes with
        # separate documents. First-use JIT work must not be charged only to
        # whichever cache mode runs first on the node.
        bench(
            "warmup-concurrent",
            documents(tokenizer, max(args.concurrency), args.prefix_tokens, 100000),
            0,
            max(args.concurrency),
        )
        for name, docs in workloads.items():
            if name == "mixed":
                # Populate all distinct documents, then make the frequent
                # document most recently used. Fixed questions preserve full
                # prefix identities so measured hits need no new block writes.
                bench("mixed-fill", list(dict.fromkeys(docs)), 0, 1)
                bench("mixed-prime", docs[:1], 0, 1)
                for concurrency in args.concurrency:
                    for round_id in range(1, args.rounds + 1):
                        bench(f"mixed-c{concurrency}-r{round_id}", docs, 0, concurrency)
                # Separate source checks after timing make each class's hit
                # counters directly observable without perturbing the trace.
                bench("mixed-hot-check", docs[:1], 0, 1)
                bench("mixed-cold-check", docs[1:2], 0, 1)
                continue
            bench(f"{name}-fill", docs, 0, 1)
            for concurrency in args.concurrency:
                for round_id in range(1, args.rounds + 1):
                    bench(f"{name}-c{concurrency}-r{round_id}", docs, round_id, concurrency)
            if name == "hot":
                # Repeat the same questions at concurrency one after all
                # variants are resident. This isolates the complete local-hit
                # fast path from the changed-question suffix in the first pass.
                for round_id in range(1, args.rounds + 1):
                    bench(f"hot-full-c1-r{round_id}", docs, round_id, 1)
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv"], text=True)
        (out / "gpu-memory.csv").write_text(memory)
    finally:
        stop(server)
        stop(master)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
