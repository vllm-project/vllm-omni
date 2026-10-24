# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observe a native PROC READ, retire its source, and check request isolation."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import time
import weakref
from pathlib import Path


def main(args):
    import torch

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector
    from vllm_omni.platforms import current_omni_platform

    torch.accelerator.set_device_index(0)
    config = {"receive_device": "cuda", "metadata_query_timeout_ms": 5000}
    if args.role == "producer":
        config.update(role="sender", host=args.host, zmq_port=0)
    else:
        config.update(role="receiver", sender_host=args.host, sender_zmq_port=args.port)
    connector = NixlConnector(config)
    path = Path(inspect.getfile(NixlConnector))
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/" in line}
    plugins = sorted(path for path in mapped if Path(path).name == "libplugin_UCX.so")
    libraries = sorted(path for path in mapped if Path(path).name.startswith(("libucp", "libuct", "libucs", "libucm")))
    report = {
        "role": args.role,
        "connector_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gpu": current_omni_platform.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ucx_tls": os.environ.get("UCX_TLS"),
        "ucx_net_devices": os.environ.get("UCX_NET_DEVICES"),
        "nixl_version": importlib.metadata.version("nixl"),
        "ucx_plugins": {plugin: hashlib.sha256(Path(plugin).read_bytes()).hexdigest() for plugin in plugins},
        "ucx_libraries": {library: hashlib.sha256(Path(library).read_bytes()).hexdigest() for library in libraries},
        "cancel_delay_ms": args.cancel_delay_ms,
        "scope": "Native connector cancellation; inspect UCX logs for actual transport. No model inference.",
    }
    try:
        if args.role == "producer":
            with socket.socket() as control:
                control.settimeout(90)
                control.bind((args.host, 0))
                control.listen(1)
                ok, _, wire = connector.put(
                    "0",
                    "2",
                    "victim_0_0",
                    torch.full((128 * 1024 * 1024,), 0.25, device="cuda"),
                )
                assert ok
                assert connector.put(
                    "0",
                    "2",
                    "healthy_0_0",
                    torch.arange(32, device="cuda", dtype=torch.int32),
                )[0]
                storage = weakref.ref(connector._pending["victim_0_0"].tensors[0])
                with args.endpoint.open("x") as endpoint:
                    endpoint.write(
                        json.dumps(
                            {
                                "port": connector._zmq_port,
                                "control_port": control.getsockname()[1],
                            }
                        )
                    )
                peer, _ = control.accept()
                with peer:
                    peer.settimeout(90)
                    with peer.makefile("rb") as messages:
                        assert messages.readline(64) == b"PROC\n", "Receiver must observe an actual pending READ"
                    assert connector._pending["victim_0_0"].claims
                    native = weakref.ref(connector._pending["victim_0_0"].agent.agent)
                    started = time.monotonic()
                    assert connector.cleanup_prefix("victim_0_") == 1
                    report["cleanup_seconds"] = time.monotonic() - started
                    assert storage() is None and native() is None
                    assert "healthy_0_0" in connector._pending
                    ok, _, reused = connector.put(
                        "0",
                        "2",
                        "victim_0_0",
                        torch.arange(32, device="cuda", dtype=torch.int32) + 7,
                    )
                    assert ok and reused["generation"] != wire["generation"]
                    peer.sendall(b"cancelled\n")
                deadline = time.monotonic() + 60
                while connector._pending:
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
                assert not connector._registered_descs
                report.update(
                    source_storage_released=True,
                    private_agent_reference_released=True,
                    replacement_generation=True,
                    pending_after=0,
                    registrations_after=0,
                )
        else:
            wait = connector._wait_for_transfer

            def observe(handle, key):
                if args.cancel_delay_ms:
                    time.sleep(args.cancel_delay_ms / 1000)
                state = connector._agent.check_xfer_state(handle)
                report["state_before_cancel"] = state
                print("LIVE READ BEFORE CANCEL", state, flush=True)
                assert state == "PROC", "A completed READ cannot establish active cancellation coverage"
                with socket.create_connection((args.host, args.control_port), timeout=30) as peer:
                    peer.sendall(b"PROC\n")
                    with peer.makefile("rb") as messages:
                        assert messages.readline(64) == b"cancelled\n"
                wait(handle, key)

            before = torch.accelerator.memory_allocated()
            connector._wait_for_transfer = observe
            try:
                result = connector.get("0", "2", "victim_0_0")
            finally:
                connector._wait_for_transfer = wait
            assert report["state_before_cancel"] == "PROC", "Active cancellation was not covered"
            if result is not None:
                assert bool((result[0] == 0.25).all()), "Drained READ observed reclaimed source values"
            report["read_outcome"] = "drained" if result is not None else "aborted"
            del result
            connector.wait_for_cleanup("victim_2_", receive_prefix="victim_0_", deadline=time.monotonic() + 10)
            assert not connector._active_gets and not connector._deferred_transfers and not connector._abandoned_queries
            assert torch.accelerator.memory_allocated() == before
            for key, offset in (("healthy_0_0", 0), ("victim_0_0", 7)):
                result = connector.get("0", "2", key)
                assert result is not None
                torch.testing.assert_close(
                    result[0].cpu(),
                    torch.arange(32, dtype=torch.int32) + offset,
                    rtol=0,
                    atol=0,
                )
                del result
            report.update(
                survivor_exact=True,
                reused_key_exact=True,
                active_gets_after=0,
                deferred_transfers_after=0,
                abandoned_queries_after=0,
                destination_storage_released=True,
            )
    finally:
        connector.close()
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print("CROSS-NODE LIVE READ PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("producer", "receiver"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--cancel-delay-ms", type=int, default=0, help="Let READ progress before checking PROC")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cancel_delay_ms < 0:
        parser.error("--cancel-delay-ms must be non-negative")
    if args.role == "producer" and args.endpoint is None:
        parser.error("producer requires --endpoint")
    if args.role == "receiver" and any(
        port is None or not 1 <= port <= 65535 for port in (args.port, args.control_port)
    ):
        parser.error("receiver requires --port and --control-port in 1..65535")
    main(args)
