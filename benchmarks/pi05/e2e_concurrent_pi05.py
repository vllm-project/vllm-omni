#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Concurrent robots against a running π0.5 OpenPI policy server.

``--clients N`` opens N websocket connections, one per simulated robot, and
each sends its next observation as soon as its chunk arrives, so a server with
``max_num_seqs > 1`` sees up to N requests queued and batches them. Client ``c``
draws its observations from a generator seeded by ``--seed`` and ``c``, so the
same stream can be replayed alone (``--clients 1 --client-offset c``) and the
chunks compared with ``--save-actions``.

Reports each client's round-trip p50/p95 and the aggregate requests per
second over the timed phase (every client's ``--iters`` round trips, started
together after ``--warmup``). Observations are random frames by default;
``--data`` draws them from recorded ones (an npz of ``base``, ``wrist``,
``state`` and ``prompt`` arrays, e.g. LIBERO frames), each robot picking its
frames from the same seeded stream.

  python e2e_concurrent_pi05.py --port 8000 --clients 8 --iters 20 --save-actions conc.npz
  python e2e_concurrent_pi05.py --port 8000 --clients 1 --client-offset 3 --iters 20 --save-actions alone3.npz
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid

import numpy as np
from e2e_latency_pi05 import CAMERA_KEYS, OPENPI_PATH, STATE_DIM, make_obs


class RecordedObservations:
    """Real observations from an npz: ``base`` and ``wrist`` images (N, H, W, 3)
    uint8, ``state`` (N, D) float32 and ``prompt`` (N) strings, e.g. frames of
    a LIBERO dataset. A robot sends the base and left-wrist cameras and its
    state padded with zeros to the checkpoint's width, as OpenPI's LIBERO
    policy does."""

    def __init__(self, path: str, state_dim: int) -> None:
        data = np.load(path, allow_pickle=False)
        self.base, self.wrist, self.state, self.prompt = data["base"], data["wrist"], data["state"], data["prompt"]
        self.state_dim = state_dim

    def __len__(self) -> int:
        return len(self.prompt)

    def observation(self, index: int, session_id: str) -> dict:
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[: self.state.shape[1]] = self.state[index]
        return {
            CAMERA_KEYS[0]: np.ascontiguousarray(self.base[index]),
            CAMERA_KEYS[1]: np.ascontiguousarray(self.wrist[index]),
            "state": state,
            "prompt": str(self.prompt[index]),
            "session_id": session_id,
        }


def run_client(
    client: int, args, start: threading.Barrier, latencies: list[list[float]], actions: list[list[np.ndarray]]
) -> None:
    import websockets.sync.client as ws_client
    from openpi_client import msgpack_numpy

    rng = np.random.default_rng([args.seed, args.client_offset + client])
    recorded = RecordedObservations(args.data, args.state_dim) if args.data else None
    packer = msgpack_numpy.Packer()
    session_id = str(uuid.uuid4())
    conn = ws_client.connect(
        f"ws://{args.host}:{args.port}{OPENPI_PATH}", compression=None, max_size=None, ping_interval=300
    )
    try:
        msgpack_numpy.unpackb(conn.recv())  # handshake metadata

        def round_trip() -> tuple[float, np.ndarray]:
            if recorded is None:
                payload = make_obs(views=args.views, prompt=args.prompt, session_id=session_id, rng=rng)
            else:
                payload = recorded.observation(int(rng.integers(len(recorded))), session_id)
            # A seed per request from the same stream (the OpenPI layer reads it
            # from the observation itself), so a replayed stream draws the same
            # noise; without one the server seeds every request at random.
            payload["seed"] = int(rng.integers(0, 2**31 - 1))
            if args.num_inference_steps is not None:
                payload["sampling_params"] = {"num_inference_steps": args.num_inference_steps}
            payload["endpoint"] = "infer"
            t0 = time.perf_counter()
            conn.send(packer.pack(payload))
            raw = conn.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Inference failed: {raw}")
            reply = msgpack_numpy.unpackb(raw)
            if isinstance(reply, dict) and reply.get("type") == "error":
                raise RuntimeError(f"Inference failed: {reply.get('message', reply)}")
            return (time.perf_counter() - t0) * 1000.0, np.asarray(reply, dtype=np.float32)

        for _ in range(args.warmup):
            round_trip()
        start.wait()
        for _ in range(args.iters):
            ms, chunk = round_trip()
            latencies[client].append(ms)
            actions[client].append(chunk)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--clients", type=int, default=1, help="Concurrent robots, one connection each")
    ap.add_argument("--client-offset", type=int, default=0, help="Seed client 0 as client OFFSET of a larger run")
    ap.add_argument("--views", type=int, default=3, choices=[1, 2, 3], help="Cameras of the random observations")
    ap.add_argument("--prompt", default="pick up the red block and place it in the bin")
    ap.add_argument("--data", default=None, help="npz of recorded observations to draw from instead of random ones")
    ap.add_argument("--state-dim", type=int, default=STATE_DIM, help="Width the recorded state is zero-padded to")
    ap.add_argument("--warmup", type=int, default=3, help="Untimed round trips per client")
    ap.add_argument("--iters", type=int, default=20, help="Timed round trips per client")
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Write the JSON result here")
    ap.add_argument("--save-actions", default=None, help="Write the chunks here: actions[client, iter]")
    args = ap.parse_args()

    latencies: list[list[float]] = [[] for _ in range(args.clients)]
    actions: list[list[np.ndarray]] = [[] for _ in range(args.clients)]
    errors: list[BaseException] = []
    timed = {}

    def timed_phase_starts():
        timed["start"] = time.perf_counter()

    start = threading.Barrier(args.clients, action=timed_phase_starts)

    def guarded(client: int) -> None:
        try:
            run_client(client, args, start, latencies, actions)
        except BaseException as exc:  # noqa: BLE001 - report every client's failure
            errors.append(exc)
            start.abort()

    threads = [threading.Thread(target=guarded, args=(client,), daemon=True) for client in range(args.clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise SystemExit(f"{len(errors)} client(s) failed; first: {errors[0]!r}")
    wall_s = time.perf_counter() - timed["start"]

    every = sorted(ms for client in latencies for ms in client)
    per_client = [
        {"p50_ms": round(statistics.median(ms), 2), "p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 2)}
        for ms in latencies
    ]
    result = {
        "clients": args.clients,
        "client_offset": args.client_offset,
        "iters": args.iters,
        "warmup": args.warmup,
        "views": args.views if args.data is None else 2,
        "data": args.data,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "requests": len(every),
        "wall_s": round(wall_s, 3),
        "req_per_s": round(len(every) / wall_s, 2),
        "p50_ms": round(statistics.median(every), 2),
        "p95_ms": round(every[int(0.95 * (len(every) - 1))], 2),
        "max_ms": round(every[-1], 2),
        "per_client": per_client,
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
    if args.save_actions:
        np.savez(args.save_actions, actions=np.stack([np.stack(chunks) for chunks in actions]))
        print("wrote", args.save_actions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
