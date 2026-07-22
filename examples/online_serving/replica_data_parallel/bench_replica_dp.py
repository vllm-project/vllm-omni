#!/usr/bin/env python3
"""Throughput-scaling + isolation benchmark for diffusion replica data parallelism.

Sends `--num-requests` text-to-video requests (up to `--concurrency` in flight)
to a running vLLM-Omni video server via `POST /v1/videos/sync` (blocks until the
clip is produced) and reports:

    completed / failed, makespan (wall clock), throughput (videos/min),
    per-request latency p50/p90/max (nearest-rank), and -- when a baseline
    manifest is supplied -- a per-request isolation check.

Each request gets a *distinct* input (a prompt from the pool + a unique seed),
so an output is a deterministic function of its own input and nothing else.

Measuring replica scaling
-------------------------
This client is replica-agnostic. Vary the *server's* replica count
(`runtime.num_replicas` in the stage config; see run_server.sh) and keep the
client workload fixed, then compare throughput:

    N=1 -> baseline
    N=2 -> expect ~2x
    N=4 -> expect ~4x   (near-linear is the goal)

Pass the single-replica videos/min via `--baseline-tpm` to print a scaling
factor for the current run.

Isolation / correctness
-----------------------
Byte-identical outputs under a *shared* prompt+seed prove nothing -- every
request would be identical regardless of isolation. Instead, give each request a
distinct input and check that its output is unchanged whether it runs alone or
concurrently amid other distinct requests:

    1. Against an N=1 server, capture a per-input baseline:
         bench_replica_dp.py --num-requests 8 --concurrency 1 \
             --write-baseline baseline.json
    2. Against an N>=2 server, replay the *same* inputs concurrently and verify
       each output matches its own baseline:
         bench_replica_dp.py --num-requests 8 --concurrency 8 \
             --check-baseline baseline.json

A mismatch means a request's output changed under concurrency -- cross-request
state bleed or misrouting -- and the run exits non-zero.

Requires: requests (`pip install requests`).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import requests

# Distinct prompts, cycled across requests; combined with a per-request unique
# seed this makes every request a distinct input (see input_key()).
DEFAULT_PROMPTS = [
    "A cat playing piano, cinematic, high detail",
    "A drone shot over a snowy mountain range at sunrise",
    "A red sports car driving through a neon-lit city at night",
    "Ocean waves crashing on a rocky shore, slow motion",
    "A hot air balloon floating over green rolling hills",
    "Timelapse of clouds moving over a desert canyon at dusk",
    "A steam train crossing a stone viaduct in autumn",
    "A jellyfish drifting through deep blue water, backlit",
]


def input_key(prompt, seed, args):
    """Stable identity of a request's input; matches baseline<->verify by content, not order."""
    return "\x1f".join([prompt, str(seed), args.size, str(args.num_frames), str(args.steps)])


def nearest_rank(xs, q):
    """Nearest-rank percentile; q in [0, 1]. Correct for tiny samples (unlike int(q*n)-1)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    rank = max(1, math.ceil(q * len(s)))
    return s[rank - 1]


def one_request(args, spec, idx, save):
    """Fire one /v1/videos/sync request; return (ok, latency, nbytes, md5, key, err)."""
    prompt, seed = spec["prompt"], spec["seed"]
    key = input_key(prompt, seed, args)
    # multipart form fields match POST /v1/videos/sync in the omni API server
    form = {
        "prompt": (None, prompt),
        "size": (None, args.size),  # e.g. "832x480"
        "num_frames": (None, str(args.num_frames)),
        "num_inference_steps": (None, str(args.steps)),
        "seed": (None, str(seed)),  # per-request unique -> distinct, reproducible input
    }
    if args.model:
        form["model"] = (None, args.model)
    if args.negative_prompt:
        form["negative_prompt"] = (None, args.negative_prompt)

    t0 = time.perf_counter()
    try:
        r = requests.post(f"{args.url}/v1/videos/sync", files=form, timeout=args.timeout)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return (False, dt, 0, "", key, f"HTTP {r.status_code}: {r.text[:160]}")
        body = r.content
        md5 = hashlib.md5(body).hexdigest()
        if save and args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.save_dir) / f"req{idx:03d}_seed{seed}.mp4").write_bytes(body)
        return (True, dt, len(body), md5, key, "")
    except Exception as e:  # noqa: BLE001 - report any client-side failure
        return (False, time.perf_counter() - t0, 0, "", key, repr(e))


def build_specs(args):
    """Deterministic list of distinct (prompt, seed) inputs; identical across baseline/verify runs."""
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    return [{"prompt": prompts[i % len(prompts)], "seed": args.base_seed + i} for i in range(args.num_requests)]


def run_batch(args, specs, save):
    """Fire specs concurrently; return (results, makespan). results = list of one_request tuples."""
    results = []
    wall0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one_request, args, spec, i, save): i for i, spec in enumerate(specs)}
        for done, f in enumerate(cf.as_completed(futs), 1):
            res = f.result()
            results.append(res)
            good, dt, nbytes, _md5, _key, err = res
            print(f"  [{done}/{len(specs)}] {'ok' if good else 'FAIL'} {dt:6.1f}s {(nbytes / 1e6):5.1f}MB {err}")
    return results, time.perf_counter() - wall0


def main():
    ap = argparse.ArgumentParser(description="Diffusion replica-DP throughput + isolation benchmark")
    ap.add_argument("--url", default="http://127.0.0.1:8098", help="server base URL (no path)")
    ap.add_argument("--num-requests", type=int, default=16, help="total requests to send")
    ap.add_argument("--concurrency", type=int, default=4, help="max in-flight (set >= replica count)")
    ap.add_argument("--warmup", type=int, default=-1, help="warmup requests before timing (default: =concurrency)")
    ap.add_argument("--base-seed", type=int, default=42, help="request i uses seed base_seed+i (distinct inputs)")
    ap.add_argument("--prompts-file", default="", help="optional file, one prompt per line (else a built-in pool)")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--size", default="832x480", help="WIDTHxHEIGHT")
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=30, help="num_inference_steps")
    ap.add_argument("--model", default="", help="optional; defaults to the served model")
    ap.add_argument("--timeout", type=float, default=1800, help="per-request timeout (s)")
    ap.add_argument("--save-dir", default="", help="save returned clips")
    ap.add_argument("--label", default="", help="run label, e.g. 'replicas=2'")
    ap.add_argument(
        "--baseline-tpm", type=float, default=0.0, help="single-replica videos/min, to print a scaling factor"
    )
    ap.add_argument("--write-baseline", default="", help="run against N=1: write per-input md5 manifest to this path")
    ap.add_argument("--check-baseline", default="", help="run against N>=2: verify each output matches this manifest")
    args = ap.parse_args()

    if args.write_baseline and args.check_baseline:
        ap.error("--write-baseline and --check-baseline are mutually exclusive")

    specs = build_specs(args)
    warmup = args.concurrency if args.warmup < 0 else args.warmup

    print(
        f"-> {args.url}/v1/videos/sync | {args.num_requests} reqs "
        f"| concurrency {args.concurrency} | warmup {warmup} | {args.size} {args.num_frames}f {args.steps}steps "
        f"| {args.label}"
    )

    # Warmup: absorb each replica's first-call compile/init cost so it does not
    # skew the measured makespan. Results are discarded and never saved.
    if warmup > 0:
        print(f"warming up ({warmup} reqs, not timed)...")
        run_batch(args, [specs[i % len(specs)] for i in range(warmup)], save=False)

    results, makespan = run_batch(args, specs, save=True)

    ok = sum(1 for r in results if r[0])
    fail = len(results) - ok
    lat = [r[1] for r in results if r[0]]
    errs = [r[5] for r in results if not r[0]]
    got = {r[4]: r[3] for r in results if r[0]}  # input_key -> md5

    print(f"\n===== result {args.label} =====")
    print(f"completed {ok} / failed {fail} | makespan {makespan:.1f}s")
    if ok:
        tpm = ok / makespan * 60.0
        print(f"throughput: {tpm:.2f} videos/min  ({ok / makespan:.4f} videos/s)")
        print(
            f"latency (s): p50={nearest_rank(lat, 0.50):.1f} "
            f"p90={nearest_rank(lat, 0.90):.1f} "
            f"min={min(lat):.1f} max={max(lat):.1f}"
        )
        if args.baseline_tpm > 0:
            print(f"scaling vs baseline ({args.baseline_tpm:.2f}/min): {tpm / args.baseline_tpm:.2f}x")
    if errs:
        print("sample failures:", errs[:3])

    # Isolation check: compare each request's output against its own single-replica baseline.
    isolation_failed = False
    if args.write_baseline:
        Path(args.write_baseline).write_text(json.dumps(got, indent=2))
        print(f"baseline: wrote {len(got)} per-input hashes -> {args.write_baseline}")
    elif args.check_baseline:
        baseline = json.loads(Path(args.check_baseline).read_text())
        matched = mismatched = unknown = 0
        for key, md5 in got.items():
            if key not in baseline:
                unknown += 1
                print(f"  isolation: NO BASELINE for input {key!r} (not in the baseline manifest)")
            elif baseline[key] == md5:
                matched += 1
            else:
                mismatched += 1
                print(f"  isolation: MISMATCH for input {key!r} (output changed under concurrency)")
        # PASS requires the run to cover the *whole* baseline: a subset (fewer requests, or
        # some failed) must not pass just because the inputs it did run happened to match.
        unverified = sorted(set(baseline) - set(got))
        for key in unverified:
            print(f"  isolation: NOT VERIFIED for baseline input {key!r} (absent from this run)")
        isolation_failed = mismatched > 0 or unknown > 0 or len(unverified) > 0
        verdict = "PASS (isolated)" if not isolation_failed else "FAIL"
        print(
            f"isolation: {matched} matched / {mismatched} mismatched / {unknown} unknown / "
            f"{len(unverified)} unverified vs baseline({len(baseline)}) -> {verdict}"
        )

    if fail > 0 or ok == 0 or isolation_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
