# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Measure publication, claim and native READ costs with ready CUDA tensors."""

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import statistics
import time
from contextlib import closing
from pathlib import Path


def resources():
    fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return {"threads": int(fields["Threads"]), "rss_kib": int(fields["VmRSS"].split()[0])}


def complete(source, key):
    import msgspec

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _GET_META_MSG, _XFER_DONE_MSG

    metadata = msgspec.msgpack.decode(
        source._handle_handshake_message(
            _GET_META_MSG + msgspec.msgpack.encode({"key": key, "claim_id": "benchmark-reader"})
        )
    )
    source._handle_handshake_message(
        _XFER_DONE_MSG
        + msgspec.msgpack.encode({"key": key, "generation": metadata["generation"], "claim_id": "benchmark-reader"})
    )


def main(args):
    import torch

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    current_omni_platform.set_device(0)
    config = {"role": "sender", "host": "127.0.0.1", "backends": ["UCX"]}
    if args.num_threads is not None:
        config["num_threads"] = args.num_threads
    with (
        closing(NixlConnector(config)) as source,
        closing(
            NixlConnector({"role": "receiver", "receive_device": "cuda:0", "metadata_query_timeout_ms": 5000})
        ) as receiver,
    ):
        tensors = [torch.full((8192,), float(i), device="cuda") for i in range(args.prefixes)]
        torch.accelerator.synchronize()
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()
        put_kwargs = {"ready_event": ready} if args.ready_event else {}
        assert source.put("0", "2", "warm_0_0", tensors[0])[0]
        assert receiver.get("0", "2", "warm_0_0", source._published["warm_0_0"]) is not None
        baseline = resources()
        durations = []
        for index, tensor in enumerate(tensors):
            start = time.perf_counter()
            assert source.put("0", "2", f"burst{index}_0_0", tensor, **put_kwargs)[0]
            durations.append((time.perf_counter() - start) * 1000)
        peak = resources()
        contexts = len({id(payload.agent) for payload in source._pending.values() if payload.agent is not None})
        reads = []
        for index in range(args.prefixes):
            key = f"burst{index}_0_0"
            start = time.perf_counter()
            received = receiver.get("0", "2", key, source._published[key])
            reads.append((time.perf_counter() - start) * 1000)
            assert received is not None
            torch.testing.assert_close(received[0].cpu(), tensors[index].cpu(), rtol=0, atol=0)
        assert not source._pending and not source._registered_descs
        unrelated = torch.cuda.Stream()
        coupled = []
        interference = []
        for repeat in range(7):
            with torch.cuda.stream(unrelated):
                torch.cuda._sleep(100_000_000)
            start = time.perf_counter()
            key = f"interference{repeat}_0_0"
            assert source.put("0", "2", key, tensors[0], **put_kwargs)[0]
            interference.append((time.perf_counter() - start) * 1000)
            coupled.append(unrelated.query())
            unrelated.synchronize()
            complete(source, key)
        assert not source._pending and not source._registered_descs
        result = {
            "prefixes": args.prefixes,
            "num_threads": args.num_threads,
            "ready_event": args.ready_event,
            "publication_p95_ms": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
            "publication_samples_ms": durations,
            "claim_read_ack_p95_ms": sorted(reads)[math.ceil(0.95 * len(reads)) - 1],
            "claim_read_ack_samples_ms": reads,
            "put_read_ack_p95_ms": sorted(a + b for a, b in zip(durations, reads))[math.ceil(0.95 * len(reads)) - 1],
            "producer_contexts_peak": contexts,
            "resources_warm": baseline,
            "resources_peak": peak,
            "added_threads": peak["threads"] - baseline["threads"],
            "rss_increase_mib": (peak["rss_kib"] - baseline["rss_kib"]) / 1024,
            "unrelated_work_put_median_ms": statistics.median(interference),
            "unrelated_stream_complete_at_put_return": sum(coupled),
            "unrelated_work_put_samples_ms": interference,
            "registrations_after": 0,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ucx_tls": os.environ.get("UCX_TLS"),
            "ucx_net_devices": os.environ.get("UCX_NET_DEVICES"),
            "gpu": current_omni_platform.get_device_name(0),
            "scope": "Ready 32 KiB payloads: put, claim, exact native READ and ACK; no model inference",
        }
    source_path = Path(inspect.getfile(NixlConnector))
    result["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with Path(args.output).open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", type=int, required=True)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--ready-event", action="store_true", help="Use the producer's recorded readiness event")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.prefixes < 1:
        parser.error("--prefixes must be positive")
    if args.num_threads is not None and args.num_threads < 0:
        parser.error("--num-threads must be non-negative")
    main(args)
