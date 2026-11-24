"""Lightweight startup checks; no model, CUDA or framework imports."""

import argparse
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path


class StartupError(RuntimeError):
    pass


def check_port(host: str, port: int, role: str) -> None:
    if not 1 <= port <= 65535:
        raise StartupError(f"{role} port must be in 1..65535: {port}")
    # Match the HTTP servers' reuse setting, allowing a recently stopped server.
    for family, kind, protocol, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        with socket.socket(family, kind, protocol) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(address)
            except OSError as exc:
                raise StartupError(f"{role} address {host}:{port} is unavailable ({exc}).") from exc


def check_ports(worker_port: int, policy_args: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("COSMOS3_POLICY_PORT", "8101")))
    parser.add_argument("--vllm-base-url", default=f"http://127.0.0.1:{worker_port}")
    args, _ = parser.parse_known_args(policy_args)
    if args.vllm_base_url.rstrip("/") != f"http://127.0.0.1:{worker_port}":
        raise StartupError(
            "run_server.sh manages its own worker; set COSMOS3_WORKER_PORT to choose its port. "
            "Use run_policy_server.sh to connect to an existing worker."
        )
    if worker_port == args.port:
        raise StartupError("Worker and policy ports must differ.")
    check_port("127.0.0.1", worker_port, "worker")
    check_port(args.host, args.port, "policy")


def process_state(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (OSError, IndexError) as exc:
        raise StartupError(f"Worker process {pid} exited before it became ready; see worker.log.") from exc


def owns_listener(group: int, port: int) -> bool:
    """Verify the listening socket belongs to this launcher's worker group (Linux)."""
    inodes = set()
    for name in ("tcp", "tcp6"):
        for line in Path(f"/proc/net/{name}").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(f"socket:[{fields[9]}]")
    if not inodes:
        return False
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if int(process_state(int(process.name))[2]) != group:
                continue
            for fd in (process / "fd").iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return True
                except OSError:
                    continue
        except (OSError, StartupError):
            continue
    return False


def wait_worker(pid: int, port: int, timeout: float = 1800) -> None:
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process_state(pid)[0] == "Z":
            raise StartupError(f"Worker process {pid} exited before it became ready; see worker.log.")
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                ready = response.status == 200
        except OSError:
            ready = False
        if ready:
            if not owns_listener(pid, port):
                raise StartupError(f"Port {port} responds but does not belong to worker group {pid}.")
            return
        time.sleep(1)
    raise StartupError("Cosmos3 vLLM-Omni worker startup timed out; see worker.log.")


def main() -> None:
    try:
        if sys.argv[1] == "preflight":
            check_ports(int(sys.argv[2]), sys.argv[3:])
        elif sys.argv[1] == "wait-worker":
            wait_worker(int(sys.argv[2]), int(sys.argv[3]))
        else:
            raise StartupError("Unknown launcher check")
    except (StartupError, ValueError) as exc:
        print(f"Cosmos3 startup refused: {exc}", file=sys.stderr)
        print(
            "Existing services are left running. Reuse the policy at /info, or stop its original "
            "launcher with Ctrl-C before restarting. For another instance, set both "
            "COSMOS3_WORKER_PORT and COSMOS3_POLICY_PORT and a separate COSMOS3_LOG_DIR.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
