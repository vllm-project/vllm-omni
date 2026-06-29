# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Micro-benchmark MooncakeTransferEngineConnector put/get latency.

Example:
    VLLM_OMNI_CONNECTOR_PROFILE=1 python benchmarks/connectors/benchmark_mooncake_te.py \
        --protocols tcp rdma --payload-types tensor tensor_dict --sizes 1MB 10MB 100MB \
        --output-json mooncake_te_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
import uuid
from typing import Any

import torch


def _parse_size(value: str) -> int:
    text = value.strip().upper()
    units = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "B": 1,
    }
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def _format_size(size: int) -> str:
    for suffix, divisor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if size >= divisor and size % divisor == 0:
            return f"{size // divisor}{suffix}"
    return f"{size}B"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_payload(payload_type: str, size: int) -> Any:
    if payload_type == "bytes":
        return bytes(size)
    if payload_type == "tensor":
        return torch.empty(size, dtype=torch.uint8)
    if payload_type == "tensor_dict":
        first = size // 2
        second = size - first
        return {
            "tokens": torch.empty(first, dtype=torch.uint8),
            "codes": torch.empty(second, dtype=torch.uint8),
        }
    raise ValueError(f"Unsupported payload type: {payload_type}")


def _release_result(result: tuple[Any, int] | None) -> None:
    if result is None:
        return
    value, _ = result
    if hasattr(value, "release"):
        value.release()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _run_case(args: argparse.Namespace, protocol: str, payload_type: str, size: int) -> dict[str, Any]:
    from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
        MooncakeTransferEngineConnector,
    )

    sender_port = _free_port()
    receiver_port = _free_port()
    pool_size = max(args.memory_pool_size, size * 4, 64 * 1024 * 1024)
    common = {
        "host": args.host,
        "protocol": protocol,
        "memory_pool_size": pool_size,
        "memory_pool_device": args.memory_pool_device,
        "pool_prewarm_slots": args.pool_prewarm_slots,
        "pool_prewarm_size": args.pool_prewarm_size or size,
        "socket_health_interval_s": args.socket_health_interval_s,
        "enable_tensor_dict_fast_path": not args.disable_tensor_dict_fast_path,
    }
    sender = MooncakeTransferEngineConnector({**common, "role": "sender", "zmq_port": sender_port})
    receiver = MooncakeTransferEngineConnector(
        {
            **common,
            "role": "receiver",
            "zmq_port": receiver_port,
            "sender_host": args.host,
            "sender_zmq_port": sender_port,
        }
    )

    latencies_ms: list[float] = []
    profile_records: list[dict[str, Any]] = []
    try:
        for i in range(args.warmup + args.iterations):
            payload = _make_payload(payload_type, size)
            request_id = f"bench-{uuid.uuid4()}"
            start = time.perf_counter()
            ok, written, metadata = sender.put("stage_a", "stage_b", request_id, payload)
            if not ok or metadata is None:
                raise RuntimeError(f"put() failed for {payload_type}/{_format_size(size)}")
            result = receiver.get("stage_a", "stage_b", request_id, metadata)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if result is None:
                raise RuntimeError(f"get() failed for {payload_type}/{_format_size(size)}")
            _release_result(result)

            if i >= args.warmup:
                latencies_ms.append(elapsed_ms)
            if args.profile:
                profile_records.extend(sender.pop_profile_records())
                profile_records.extend(receiver.pop_profile_records())

            if written != metadata["data_size"]:
                raise RuntimeError(f"metadata size mismatch: written={written}, metadata={metadata}")
    finally:
        receiver.close()
        sender.close()

    return {
        "protocol": protocol,
        "payload_type": payload_type,
        "payload_size": size,
        "payload_size_label": _format_size(size),
        "iterations": args.iterations,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": _percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "profile_records": profile_records,
    }


def _markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Payload Size | Protocol | Payload Type | p50 put+get (ms) | p95 put+get (ms) |",
        "|---|---|---|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['payload_size_label']} | {result['protocol']} | {result['payload_type']} | "
            f"{result['p50_ms']:.3f} | {result['p95_ms']:.3f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--protocols", nargs="+", default=["tcp"], choices=["tcp", "rdma"])
    parser.add_argument(
        "--payload-types",
        nargs="+",
        default=["tensor", "tensor_dict"],
        choices=["bytes", "tensor", "tensor_dict"],
    )
    parser.add_argument("--sizes", nargs="+", default=["1MB", "10MB", "100MB", "1GB"])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--memory-pool-size", type=_parse_size, default=1024**3)
    parser.add_argument("--memory-pool-device", default="cpu")
    parser.add_argument("--pool-prewarm-slots", type=int, default=0)
    parser.add_argument("--pool-prewarm-size", type=_parse_size, default=0)
    parser.add_argument("--socket-health-interval-s", type=float, default=30.0)
    parser.add_argument("--disable-tensor-dict-fast-path", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile:
        os.environ["VLLM_OMNI_CONNECTOR_PROFILE"] = "1"

    sizes = [_parse_size(size) for size in args.sizes]
    results = [
        _run_case(args, protocol, payload_type, size)
        for protocol in args.protocols
        for payload_type in args.payload_types
        for size in sizes
    ]

    print(_markdown_table(results))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
