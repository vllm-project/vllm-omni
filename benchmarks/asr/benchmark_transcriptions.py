#!/usr/bin/env python3
"""Steady-state throughput benchmark for ``/v1/audio/transcriptions``.

N workers each loop back-to-back for a fixed duration. Completions in the
leading warmup window and the trailing drain window are discarded, so the
reported rate covers steady state only -- a single burst of N concurrent
requests measures ramp-up plus drain instead, which here produced +/-30-50%
run-to-run spread versus 1-5% for this method. Repeated R times, reporting
median and range so a number can be trusted or visibly distrusted.

Example (ASR only, then with word timestamps)::

    vllm-omni serve Qwen/Qwen3-ASR-1.7B --omni --trust-remote-code \
        --deploy-config vllm_omni/deploy/qwen3_asr.yaml

    python benchmarks/asr/benchmark_transcriptions.py \
        --audio clip_30s.mp3 --concurrency 256 384 512 --label "ASR only"

    python benchmarks/asr/benchmark_transcriptions.py \
        --audio clip_30s.mp3 --concurrency 256 384 512 \
        --response-format verbose_json --granularities word \
        --label "with alignment"

Word timestamps additionally need ``--forced-aligner
Qwen/Qwen3-ForcedAligner-0.6B`` on the server, and the aligner loads lazily,
so send one request before measuring or the load lands inside a run.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


async def _worker(session, url, model, audio, fname, grans, fmt, stop_at, done):
    while True:
        now = time.perf_counter()
        if now >= stop_at:
            return
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=fname, content_type="audio/mpeg")
        form.add_field("model", model)
        form.add_field("response_format", fmt)
        for g in grans or []:
            form.add_field("timestamp_granularities[]", g)
        t0 = time.perf_counter()
        try:
            async with session.post(url, data=form) as r:
                body = await r.json()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {str(body)[:200]}")
        except Exception as e:
            done.append((time.perf_counter(), t0, None, repr(e)[:120]))
            continue
        done.append((time.perf_counter(), t0, body, None))


async def _one_run(url, model, audio, fname, n, grans, fmt, duration, warmup, drain):
    done: list = []
    to = aiohttp.ClientTimeout(total=600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=to, connector=conn) as s:
        start = time.perf_counter()
        stop_at = start + duration
        await asyncio.gather(*(_worker(s, url, model, audio, fname, grans, fmt, stop_at, done) for _ in range(n)))

    lo, hi = start + warmup, start + duration - drain
    window = [d for d in done if lo <= d[0] <= hi]
    errs = [d for d in window if d[3] is not None]
    ok = [d for d in window if d[3] is None]
    span = hi - lo
    lats = sorted(t_end - t_start for t_end, t_start, _, _ in ok)
    return {
        "rps": len(ok) / span if span > 0 else 0.0,
        "n_ok": len(ok),
        "n_err": len(errs),
        "err": errs[0][3] if errs else None,
        "p50": statistics.median(lats) if lats else 0.0,
        "p95": lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else 0.0,
        "sample": next((d[2] for d in ok if d[2]), None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8077")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--audio-seconds", type=float, default=30.0)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--duration", type=float, default=45.0, help="seconds per run")
    ap.add_argument("--warmup", type=float, default=12.0, help="leading seconds discarded")
    ap.add_argument("--drain", type=float, default=5.0, help="trailing seconds discarded")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--granularities", nargs="*", default=None)
    ap.add_argument("--response-format", default="json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    audio = Path(a.audio).read_bytes()
    fname = Path(a.audio).name
    url = f"{a.base_url}/v1/audio/transcriptions"

    print(f"\n=== {a.label} | {fname} ({a.audio_seconds:.0f}s) | grans={a.granularities or 'none'} ===")
    print(f"    {a.repeats} runs x {a.duration:.0f}s (warmup {a.warmup:.0f}s, drain {a.drain:.0f}s discarded)")
    header = (
        f"{'conc':>5} {'rps_med':>9} {'rps_min':>9} {'rps_max':>9} "
        f"{'spread':>7} {'RTFx':>8} {'p50':>7} {'p95':>7} {'err':>5}"
    )
    print(header)

    results = {}
    for n in a.concurrency:
        runs = [_run_sync(url, a, audio, fname, n) for _ in range(a.repeats)]
        rps = sorted(r["rps"] for r in runs)
        med = statistics.median(rps)
        spread = (rps[-1] - rps[0]) / med * 100 if med else 0.0
        p50 = statistics.median([r["p50"] for r in runs])
        p95 = statistics.median([r["p95"] for r in runs])
        nerr = sum(r["n_err"] for r in runs)
        print(
            f"{n:>5} {med:>9.2f} {rps[0]:>9.2f} {rps[-1]:>9.2f} {spread:>6.0f}% "
            f"{med * a.audio_seconds:>8.0f} {p50:>7.2f} {p95:>7.2f} {nerr:>5}"
        )
        if nerr:
            print(f"      first error: {next(r['err'] for r in runs if r['err'])}")
        results[n] = {"median_rps": med, "runs": rps, "p50": p50, "p95": p95, "errors": nerr}

    sample = next((r for n in a.concurrency for r in [results[n]] if r), None)
    if sample:
        last = _last_sample
        if last is not None:
            w = last.get("words")
            print(f"    words returned: {len(w) if w else 0}")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"label": a.label, "results": results}, indent=2))


_last_sample = None


def _run_sync(url, a, audio, fname, n):
    global _last_sample
    r = asyncio.run(
        _one_run(
            url,
            a.model,
            audio,
            fname,
            n,
            a.granularities,
            a.response_format,
            a.duration,
            a.warmup,
            a.drain,
        )
    )
    if r["sample"]:
        _last_sample = r["sample"]
    return r


if __name__ == "__main__":
    main()
