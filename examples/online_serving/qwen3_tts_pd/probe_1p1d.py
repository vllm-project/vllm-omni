"""1P1D PD probe for Qwen3-TTS Talker.

Purpose
-------
Verify that splitting the Qwen3-TTS Talker into a prefill-only stage
and a decode-only stage (driven by ``MooncakeConnector``) preserves
both correctness and the latency benefit of PD disaggregation.

Specifically this script answers two questions before we let the
M-P-1-D work proceed (Task 3.x and beyond):

1. **Correctness probe** -- under greedy sampling, the audio_codes
   sequence emitted by the PD pipeline must be bit-equal to the audio
   sequence emitted by the single-process baseline.  Any drift here
   means MooncakeConnector failed to capture the talker attention KV
   (e.g. because the talker code-predictor sub-module's KV is not on
   the standard vLLM ``Qwen3Model`` attention path).
2. **KV transfer probe** -- the decode stage must NOT replay prefill.
   This is detected by comparing ``time-to-first-audio-codes`` between
   the PD run and the baseline single-process run; with a hot prefill
   the PD first-frame latency should be on par with or better than the
   single-process run.  If PD is markedly slower, KV transfer is
   silently failing and the consumer is restarting prefill.

Usage
-----
Two servers must be running side-by-side on the same host:

    # Baseline (single-process talker + code2wav)
    vllm serve <Qwen3-TTS path> --omni --port 8090 \\
        --stage-configs-path vllm_omni/deploy/qwen3_tts.yaml

    # PD probe (1P1D split)
    vllm serve <Qwen3-TTS path> --omni --port 8091 \\
        --stage-configs-path vllm_omni/deploy/qwen3_tts_pd_1p1d.yaml

    # Then run the probe:
    python examples/online_serving/qwen3_tts_pd/probe_1p1d.py \\
        --baseline-url http://localhost:8090 \\
        --pd-url       http://localhost:8091 \\
        --runs 3

Exit code is 0 only if both probes pass; non-zero exit lets CI gate the
PD work.  All findings are appended to ``migration_inventory.md`` via
the ``--inventory-path`` flag (optional).

This script intentionally has no test/conftest dependency so it can be
invoked from a one-off shell.  It only uses ``httpx`` (already a hard
dep of ``examples/online_serving``).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("qwen3_tts_pd.probe_1p1d")

# Default short / deterministic test prompt.  Greedy sampling on a
# fixed prompt is required for bit-equal comparison.
_DEFAULT_PROMPT = "vLLM-Omni Qwen3 TTS prefill decode disaggregation probe."

# Time-to-first-audio-codes ratio above which we suspect the decode
# stage replayed prefill instead of consuming remote KV.  3x baseline
# is conservative; tune via --ttfb-ratio-threshold if your hardware
# is known to have substantially different prefill / decode profiles.
_DEFAULT_TTFB_RATIO_THRESHOLD = 3.0


@dataclass
class RunResult:
    """One end-to-end TTS request observation."""

    label: str
    ttfb_ms: float
    total_ms: float
    num_frames: int
    audio_codes_hash: str
    raw_codes_head: list[list[int]] = field(default_factory=list)
    error: str | None = None


def _hash_audio_codes(frames: Iterable[list[int]]) -> str:
    """Hex digest of the entire audio_codes stream (order-sensitive)."""
    import hashlib

    h = hashlib.sha256()
    for frame in frames:
        # ``int`` -> ascii bytes; tab-separated to avoid ambiguity.
        h.update(("\t".join(str(v) for v in frame) + "\n").encode("ascii"))
    return h.hexdigest()


def _run_once_ndjson(
    base_url: str,
    label: str,
    prompt: str,
    timeout_s: float,
) -> RunResult:
    """Drive one streaming TTS request and capture timing + tokens.

    Uses NDJSON streaming because (a) the AR-only patch family also
    speaks NDJSON so the same probe can be reused there, and (b) it
    gives a precise wall-clock for "first non-zero audio_codes frame".
    """
    url = base_url.rstrip("/") + "/v1/audio/speech"
    payload = {
        "model": "qwen3_tts",
        "input": prompt,
        "voice": "default",
        "stream": True,
        "response_format": "pcm",  # serving layer ignores in AR-only NDJSON path
    }

    t0 = time.perf_counter()
    first_frame_ts: float | None = None
    frames: list[list[int]] = []
    err: str | None = None

    try:
        with httpx.Client(timeout=timeout_s) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # PCM/WAV bytes leak: the deploy is NOT in
                        # AR-only NDJSON mode.  Bail out — this probe
                        # is for AR-only / PD-on-AR streams.
                        err = (
                            "Server returned non-NDJSON line; ensure the "
                            "deploy uses final_output_type=latent OR run "
                            "with PCM-mode probe (not yet implemented)."
                        )
                        break
                    audio_codes = rec.get("audio_codes")
                    if audio_codes is None:
                        continue
                    if first_frame_ts is None:
                        first_frame_ts = time.perf_counter()
                    frames.append(list(audio_codes))
    except httpx.HTTPError as e:
        err = f"HTTP error: {e!r}"

    total_ms = (time.perf_counter() - t0) * 1000.0
    ttfb_ms = (first_frame_ts - t0) * 1000.0 if first_frame_ts is not None else float("inf")

    return RunResult(
        label=label,
        ttfb_ms=ttfb_ms,
        total_ms=total_ms,
        num_frames=len(frames),
        audio_codes_hash=_hash_audio_codes(frames),
        raw_codes_head=frames[:3],
        error=err,
    )


def _summarise(results: list[RunResult]) -> dict[str, float]:
    """Return median TTFB / total_ms across non-error runs."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return {"median_ttfb_ms": float("inf"), "median_total_ms": float("inf"), "n_ok": 0}
    return {
        "median_ttfb_ms": statistics.median(r.ttfb_ms for r in ok),
        "median_total_ms": statistics.median(r.total_ms for r in ok),
        "n_ok": len(ok),
    }


def _judge(
    baseline: list[RunResult],
    pd: list[RunResult],
    ttfb_ratio_threshold: float,
) -> tuple[bool, list[str], dict]:
    """Return (passed, findings, metrics).

    The probe passes iff:
      * baseline and PD both produced at least one non-error run,
      * every (baseline, pd) pair has identical ``audio_codes_hash``
        and ``num_frames`` (greedy ⇒ bit-equal),
      * median PD TTFB <= ``ttfb_ratio_threshold`` * median baseline TTFB
        (proxy for "decode did not replay prefill").
    """
    findings: list[str] = []
    base_summary = _summarise(baseline)
    pd_summary = _summarise(pd)

    metrics = {"baseline": base_summary, "pd": pd_summary}

    if base_summary["n_ok"] == 0:
        findings.append("Baseline produced 0 successful runs; cannot compare.")
        return False, findings, metrics
    if pd_summary["n_ok"] == 0:
        findings.append("PD probe produced 0 successful runs; cannot compare.")
        return False, findings, metrics

    # Bit-equal correctness: pair runs by index across baseline / PD.
    n_pairs = min(len(baseline), len(pd))
    bit_equal = True
    for i in range(n_pairs):
        b, p = baseline[i], pd[i]
        if b.error or p.error:
            continue
        if b.audio_codes_hash != p.audio_codes_hash:
            bit_equal = False
            findings.append(
                f"Run {i}: audio_codes hash mismatch "
                f"baseline={b.audio_codes_hash[:12]}.. pd={p.audio_codes_hash[:12]}.. "
                f"(num_frames base={b.num_frames} pd={p.num_frames})"
            )
        elif b.num_frames != p.num_frames:
            bit_equal = False
            findings.append(
                f"Run {i}: num_frames mismatch base={b.num_frames} pd={p.num_frames} "
                "(hash matched -- likely a length/stream-end bug)"
            )

    if not bit_equal:
        findings.append(
            "Greedy bit-equal probe FAILED: KV transfer probably did not "
            "capture all attention layers (suspect: code-predictor sub-module)."
        )

    # KV-transfer-success heuristic: PD TTFB shouldn't be wildly larger.
    base_ttfb = base_summary["median_ttfb_ms"]
    pd_ttfb = pd_summary["median_ttfb_ms"]
    ratio = pd_ttfb / base_ttfb if base_ttfb > 0 else float("inf")
    metrics["pd_to_baseline_ttfb_ratio"] = ratio
    if ratio > ttfb_ratio_threshold:
        findings.append(
            f"PD median TTFB {pd_ttfb:.1f}ms vs baseline {base_ttfb:.1f}ms "
            f"(ratio={ratio:.2f}x > {ttfb_ratio_threshold:.2f}x): "
            "decode stage likely replayed prefill -- suspect KV transfer broken."
        )
    else:
        findings.append(
            f"PD TTFB ratio {ratio:.2f}x <= threshold {ttfb_ratio_threshold:.2f}x: KV transfer likely succeeded."
        )

    passed = bit_equal and ratio <= ttfb_ratio_threshold
    return passed, findings, metrics


def _maybe_append_inventory(
    inventory_path: str | None,
    passed: bool,
    findings: list[str],
    metrics: dict,
) -> None:
    """Optionally append the probe outcome to ``migration_inventory.md``.

    Fills in the ``KV 可分离层清单`` section produced by Task 1 so
    Task 3.x has explicit go/no-go.
    """
    if not inventory_path:
        return
    try:
        block = [
            "",
            "## KV 可分离层清单（task 2 探针结果）",
            "",
            f"- **Probe verdict**: {'PASS' if passed else 'FAIL'}",
            f"- **Baseline median TTFB**: {metrics['baseline']['median_ttfb_ms']:.2f} ms",
            f"- **PD median TTFB**: {metrics['pd']['median_ttfb_ms']:.2f} ms",
            f"- **TTFB ratio (PD/baseline)**: {metrics.get('pd_to_baseline_ttfb_ratio', float('nan')):.2f}x",
            "",
            "**Findings**:",
            "",
            *(f"  - {line}" for line in findings),
            "",
        ]
        with open(inventory_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(block))
    except OSError as e:
        logger.warning("Could not append to %s: %s", inventory_path, e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-url",
        default="http://localhost:8090",
        help="Single-process Qwen3-TTS server (vllm_omni/deploy/qwen3_tts.yaml).",
    )
    parser.add_argument(
        "--pd-url",
        default="http://localhost:8091",
        help="1P1D PD server (vllm_omni/deploy/qwen3_tts_pd_1p1d.yaml).",
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="Test prompt (greedy sampling).")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per server.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-request HTTP timeout.")
    parser.add_argument(
        "--ttfb-ratio-threshold",
        type=float,
        default=_DEFAULT_TTFB_RATIO_THRESHOLD,
        help="Fail if PD TTFB exceeds this multiple of baseline TTFB.",
    )
    parser.add_argument(
        "--inventory-path",
        default=None,
        help="Optional migration_inventory.md path to append the verdict to.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Probing baseline @ %s", args.baseline_url)
    baseline_results: list[RunResult] = [
        _run_once_ndjson(args.baseline_url, f"baseline_{i}", args.prompt, args.timeout_s) for i in range(args.runs)
    ]
    for r in baseline_results:
        logger.info(
            "  baseline run: ttfb=%.1fms total=%.1fms frames=%d hash=%s err=%s",
            r.ttfb_ms,
            r.total_ms,
            r.num_frames,
            r.audio_codes_hash[:12],
            r.error,
        )

    logger.info("Probing PD @ %s", args.pd_url)
    pd_results: list[RunResult] = [
        _run_once_ndjson(args.pd_url, f"pd_{i}", args.prompt, args.timeout_s) for i in range(args.runs)
    ]
    for r in pd_results:
        logger.info(
            "  pd run: ttfb=%.1fms total=%.1fms frames=%d hash=%s err=%s",
            r.ttfb_ms,
            r.total_ms,
            r.num_frames,
            r.audio_codes_hash[:12],
            r.error,
        )

    passed, findings, metrics = _judge(baseline_results, pd_results, args.ttfb_ratio_threshold)

    print()
    print("=" * 72)
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    print(f"Baseline median TTFB: {metrics['baseline']['median_ttfb_ms']:.2f} ms")
    print(f"PD median TTFB:       {metrics['pd']['median_ttfb_ms']:.2f} ms")
    print(f"TTFB ratio:           {metrics.get('pd_to_baseline_ttfb_ratio', float('nan')):.2f}x")
    print("Findings:")
    for line in findings:
        print(f"  - {line}")
    print("=" * 72)

    _maybe_append_inventory(args.inventory_path, passed, findings, metrics)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
