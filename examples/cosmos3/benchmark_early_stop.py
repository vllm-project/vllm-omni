"""Replay a fixed observation to verify early-stop work and completed prefixes.

This measures serial requests, not closed-loop LIBERO success or serving capacity.
All modes retain the same 16-position model context and fixed seed.
"""

import argparse
import base64
import io
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from vllm_omni.entrypoints.cosmos3.libero_remote_policy import LiberoRemotePolicyService, RemotePolicyConfig
from vllm_omni.entrypoints.cosmos3.remote_action_stream import RemoteActionPolicyHTTPServer
from vllm_omni.entrypoints.cosmos3.vllm_omni_action import VllmOmniActionTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    variants = [
        ("constant_full", "constant_euler", None, False),
        ("constant_k4", "constant_euler", 4, False),
        ("has_full", "has_euler", None, False),
        ("has_k4", "has_euler", 4, False),
        ("has_k1", "has_euler", 1, False),
        ("has_k16", "has_euler", 16, False),
        ("has_k4_stream", "has_euler", 4, True),
    ]
    transports = {
        name: VllmOmniActionTransport(
            args.base_url, seed=args.seed, action_time_schedule=schedule, early_stop_actions=count, poll_interval=0.05
        )
        for name, schedule, count, _ in variants
    }
    service = LiberoRemotePolicyService(transports[variants[0][0]], RemotePolicyConfig(streaming_enabled=True))
    server = RemoteActionPolicyHTTPServer(("127.0.0.1", 0), service)
    url = f"http://127.0.0.1:{server.server_port}"
    service.cfg = RemotePolicyConfig(streaming_enabled=True, callback_base_url=url)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = requests.Session()
    session.trust_env = False
    buffer = io.BytesIO()
    Image.fromarray(np.full((256, 512, 3), 30, dtype=np.uint8)).save(buffer, format="PNG")
    item = {
        "image": base64.b64encode(buffer.getvalue()).decode(),
        "prompt": "pick up the object",
        "domain_name": "libero",
        "image_size": 256,
    }
    rows, references, differences = [], {}, []
    try:
        for index in range(args.warmup + args.repeats):
            # Warm up every mode, then rotate execution order to reduce drift.
            order = (
                variants
                if index < args.warmup
                else variants[index % len(variants) :] + variants[: index % len(variants)]
            )
            for name, schedule, count, stream in order:
                service.transport = transports[name]
                started = time.perf_counter()
                response = session.post(url + ("/predict_stream" if stream else "/predict"), json=item, timeout=1800)
                response.raise_for_status()
                elapsed = (time.perf_counter() - started) * 1000
                events = [json.loads(line) for line in response.text.splitlines()] if stream else []
                final = events[-1] if stream else response.json()
                if stream and final.get("type") != "final":
                    raise RuntimeError(final)
                actions = np.asarray(final["action"], dtype=np.float32)
                assert actions.shape == (count or 16, 10)
                if count is None:
                    references[schedule] = actions
                if schedule in references:
                    diff = float(np.max(np.abs(actions - references[schedule][: len(actions)])))
                    differences.append(diff)
                    np.testing.assert_array_equal(actions, references[schedule][: len(actions)])
                if stream:
                    recovered = np.zeros_like(actions)
                    seen = []
                    for event in events[:-1]:
                        positions = event["action_indices"]
                        seen.extend(positions)
                        recovered[positions] = event["actions"]
                    assert sorted(seen) == list(range(len(actions)))
                    np.testing.assert_array_equal(recovered, actions)
                metrics = final["performance"]["worker_metrics"][0]
                steps = (1 if count == 1 else 5 if count == 4 else 10) if schedule == "has_euler" else 10
                assert metrics["denoise_steps"] == steps and metrics["requested_denoise_steps"] == 10
                row = {"variant": name, "client_ms": elapsed, "returned_actions": len(actions), **metrics}
                if index >= args.warmup:
                    rows.append(row)
                print(json.dumps({"round": index, **row}), flush=True)
        summary = {}
        for name, _, _, _ in variants:
            group = [row for row in rows if row["variant"] == name]
            summary[name] = {
                "requests": len(group),
                "denoise_steps": sorted({row["denoise_steps"] for row in group}),
                "mean_client_ms": statistics.mean(row["client_ms"] for row in group),
                "mean_worker_ms": statistics.mean(row["worker_inference_ms"] for row in group),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "scope": "serial fixed-observation replay; synthetic uint8 RGB=30, 256x512; not closed-loop evaluation",
                    "seed": args.seed,
                    "warmup_per_variant": args.warmup,
                    "max_prefix_abs_difference": max(differences),
                    "summary": summary,
                    "requests": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        session.close()
        for transport in transports.values():
            transport.close()


if __name__ == "__main__":
    main()
