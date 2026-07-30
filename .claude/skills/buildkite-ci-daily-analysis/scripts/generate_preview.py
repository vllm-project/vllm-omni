#!/usr/bin/env python3
"""
Generate a preview HTML report with synthetic but realistic Buildkite job
data, so you can eyeball the layout / styling / filtering UX without a
real Buildkite API token.

Usage:
    python scripts/generate_preview.py
    python scripts/generate_preview.py --output /tmp/preview.html
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the sibling script importable
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

import ci_daily_analysis as m  # noqa: E402

# Synthetic but plausible seed data -------------------------------------------------

# Each scenario represents a (pipeline, branch, ci_bucket, message) tuple
# covering all four CI buckets: merge / ready / nightly / weekly. The
# `message` field is what classify_build() uses to bucket scheduled runs.
PIPELINE_BRANCH_SCENARIOS = [
    # ── merge bucket (main, ordinary runs) ──
    {
        "pipeline": "vllm-omni",
        "branch": "main",
        "ci_bucket": m.CI_BUCKET_MERGE,
        "message": "PR #4521 merged",
        "build_nums": [4521, 4520, 4519],
        "jobs": [
            ("unit-tests", 0.92, 480),
            ("lint", 0.97, 90),
            ("type-check", 0.95, 60),
            ("diffusion-model-test", 0.78, 1450),
            ("diffusion-cache-perf", 0.80, 1820),
            ("tts-model-test", 0.82, 1280),
            ("tts-streaming-test", 0.85, 980),
            ("omni-e2e-nightly", 0.65, 2700),
            ("quantization-test", 0.70, 1600),
            ("gpu-h200-trigger", 0.88, 540),
            ("docs-build", 0.99, 30),
            ("precommit-hooks", 0.96, 45),
        ],
    },
    # ── ready bucket (non-main) ──
    {
        "pipeline": "vllm-omni",
        "branch": "alice/diffusion-fp8",
        "ci_bucket": m.CI_BUCKET_READY,
        "message": "alice pushed 3 commits",
        "build_nums": [4522, 4523],
        "jobs": [
            ("unit-tests", 0.85, 510),
            ("lint", 0.90, 95),
            ("diffusion-fp8-bench", 0.55, 2200),
            ("diffusion-cache-perf", 0.50, 1900),
            ("quantization-test", 0.60, 1700),
            ("gpu-h200-trigger", 0.75, 600),
        ],
    },
    {
        "pipeline": "vllm-omni",
        "branch": "bob/tts-streaming",
        "ci_bucket": m.CI_BUCKET_READY,
        "message": "bob pushed 1 commit",
        "build_nums": [4524],
        "jobs": [
            ("unit-tests", 0.93, 470),
            ("tts-streaming-test", 0.70, 1100),
            ("tts-model-test", 0.72, 1320),
            ("docs-build", 1.0, 28),
        ],
    },
    # ── nightly bucket (main + scheduled nightly) ──
    {
        "pipeline": "vllm-omni",
        "branch": "main",
        "ci_bucket": m.CI_BUCKET_NIGHTLY,
        "message": "Scheduled nightly build",
        "build_nums": [9001],
        "jobs": [
            ("diffusion-model-test", 0.55, 2400),
            ("diffusion-cache-perf", 0.50, 3100),
            ("tts-model-test", 0.60, 1900),
            ("tts-streaming-test", 0.65, 1500),
            ("omni-e2e-nightly", 0.50, 4500),
            ("quantization-test", 0.55, 2700),
            ("unit-tests", 0.88, 600),
            ("lint", 0.95, 110),
            ("nightly-collect-results", 0.97, 60),
        ],
    },
    # ── weekly bucket (main + scheduled weekly) ──
    {
        "pipeline": "vllm-omni",
        "branch": "main",
        "ci_bucket": m.CI_BUCKET_WEEKLY,
        "message": "Scheduled weekly build",
        "build_nums": [9100],
        "jobs": [
            ("full-regression-suite", 0.45, 5400),
            ("long-context-bench", 0.55, 4200),
            ("throughput-bench", 0.60, 3700),
            ("unit-tests", 0.90, 700),
            ("lint", 0.95, 130),
            ("weekly-collect-results", 0.98, 80),
        ],
    },
    # ── npu main: merge ──
    {
        "pipeline": "vllm-omni-npu-ci",
        "branch": "main",
        "ci_bucket": m.CI_BUCKET_MERGE,
        "message": "PR #219 merged",
        "build_nums": [219, 218, 217],
        "jobs": [
            ("npu-910b-unit", 0.86, 720),
            ("npu-omni-e2e", 0.62, 3000),
            ("npu-tts-inference", 0.74, 1620),
            ("npu-diffusion-cann", 0.55, 2100),
            ("npu-quantization-ascend", 0.60, 1900),
            ("lint", 0.95, 110),
        ],
    },
    # ── npu ready (non-main) ──
    {
        "pipeline": "vllm-omni-npu-ci",
        "branch": "carol/ascend-graph-opt",
        "ci_bucket": m.CI_BUCKET_READY,
        "message": "carol pushed 2 commits",
        "build_nums": [220],
        "jobs": [
            ("npu-910b-unit", 0.80, 760),
            ("npu-graph-opt-bench", 0.45, 2400),
            ("npu-quantization-ascend", 0.55, 2050),
            ("lint", 0.92, 100),
        ],
    },
    # ── npu nightly ──
    {
        "pipeline": "vllm-omni-npu-ci",
        "branch": "main",
        "ci_bucket": m.CI_BUCKET_NIGHTLY,
        "message": "Scheduled nightly build",
        "build_nums": [9010],
        "jobs": [
            ("npu-nightly-cann-kernels", 0.45, 3600),
            ("npu-omni-e2e", 0.50, 4800),
            ("npu-tts-inference", 0.60, 2200),
            ("npu-diffusion-cann", 0.45, 3200),
            ("npu-quantization-ascend", 0.50, 2800),
            ("npu-910b-unit", 0.85, 800),
            ("lint", 0.93, 130),
            ("nightly-collect-results", 0.97, 70),
        ],
    },
    # ── npu weekly ──
    {
        "pipeline": "vllm-omni-npu-ci",
        "branch": "main",
        "ci_bucket": m.CI_BUCKET_WEEKLY,
        "message": "Scheduled weekly build",
        "build_nums": [9110],
        "jobs": [
            ("npu-full-regression", 0.40, 6000),
            ("npu-throughput-bench", 0.55, 4400),
            ("npu-910b-unit", 0.85, 850),
            ("lint", 0.93, 130),
            ("weekly-collect-results", 0.98, 90),
        ],
    },
]

# Additional state bucket weights — what state each job ends up in,
# aside from passed/failed.
EXTRA_STATE_PROBS = {
    "canceled": 0.02,
    "running": 0.01,
    "other": 0.005,
}


def synthesize() -> list[m.JobRecord]:
    rng = random.Random(42)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    records: list[m.JobRecord] = []

    cursor = today + timedelta(hours=0)  # start of "today"
    for scenario in PIPELINE_BRANCH_SCENARIOS:
        pipeline = scenario["pipeline"]
        branch = scenario["branch"]
        ci_bucket = scenario["ci_bucket"]
        build_message = scenario["message"]
        for build_num in scenario["build_nums"]:
            commit = f"{(rng.randrange(0, 0xFFFF) & 0xFFFF):04x}{(rng.randrange(0, 0xFFFF) & 0xFFFF):04x}"[:7]
            build_url = f"https://buildkite.com/vllm/{pipeline}/builds/{build_num}"
            # Spread jobs across the day
            cursor = cursor + timedelta(minutes=rng.randint(2, 15))
            for job_name, success_rate, mean_dur in scenario["jobs"]:
                # Pick outcome
                roll = rng.random()
                if roll < success_rate:
                    state, bucket = "passed", m.STATE_PASSED
                    # passed jobs sometimes finish a bit slower than expected
                    dur_factor = rng.uniform(0.85, 1.25)
                elif roll < success_rate + 0.06:
                    state, bucket = "failed", m.STATE_FAILED
                    # failed jobs often bail out early, but can also run long
                    dur_factor = rng.choice([rng.uniform(0.05, 0.6), rng.uniform(1.0, 1.8)])
                elif roll < success_rate + 0.06 + EXTRA_STATE_PROBS["canceled"]:
                    state, bucket = "canceled", m.STATE_CANCELED
                    dur_factor = rng.uniform(0.2, 0.9)
                elif roll < (success_rate + 0.06 + EXTRA_STATE_PROBS["canceled"] + EXTRA_STATE_PROBS["running"]):
                    state, bucket = "running", m.STATE_RUNNING
                    # No duration — still in flight
                    dur_factor = None
                    started_at = cursor + timedelta(minutes=rng.randint(0, 5))
                    records.append(
                        m.JobRecord(
                            pipeline=pipeline,
                            branch=branch,
                            build_number=build_num,
                            build_url=build_url,
                            commit=commit,
                            job_id=f"jid-{len(records)}",
                            job_name=job_name,
                            state=state,
                            bucket=bucket,
                            ci_bucket=ci_bucket,
                            build_message=build_message,
                            started_at=started_at,
                            finished_at=None,
                            duration_seconds=None,
                            job_url=f"{build_url}#{job_name}",
                            exit_status=None,
                        )
                    )
                    continue
                else:
                    state, bucket = "skipped", m.STATE_OTHER
                    dur_factor = None
                    records.append(
                        m.JobRecord(
                            pipeline=pipeline,
                            branch=branch,
                            build_number=build_num,
                            build_url=build_url,
                            commit=commit,
                            job_id=f"jid-{len(records)}",
                            job_name=job_name,
                            state=state,
                            bucket=bucket,
                            ci_bucket=ci_bucket,
                            build_message=build_message,
                            started_at=None,
                            finished_at=None,
                            duration_seconds=None,
                            job_url=f"{build_url}#{job_name}",
                            exit_status=None,
                        )
                    )
                    continue

                duration = mean_dur * dur_factor
                started_at = cursor + timedelta(seconds=rng.randint(0, 30))
                finished_at = started_at + timedelta(seconds=duration)
                records.append(
                    m.JobRecord(
                        pipeline=pipeline,
                        branch=branch,
                        build_number=build_num,
                        build_url=build_url,
                        commit=commit,
                        job_id=f"jid-{len(records)}",
                        job_name=job_name,
                        state=state,
                        bucket=bucket,
                        ci_bucket=ci_bucket,
                        build_message=build_message,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_seconds=duration,
                        job_url=f"{build_url}#{job_name}",
                        exit_status=0 if state == "passed" else (rng.randint(1, 9) if state == "failed" else None),
                    )
                )
                cursor = finished_at + timedelta(seconds=rng.randint(10, 90))

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a preview HTML report with synthetic data.")
    parser.add_argument(
        "--output",
        default="ci-daily-preview.html",
        metavar="PATH",
        help="Output HTML path. Default: ci-daily-preview.html",
    )
    parser.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD", help="Override the report's UTC date label. Default: today UTC."
    )
    args = parser.parse_args()

    records = synthesize()
    today = (datetime.now(timezone.utc)).date().isoformat()
    date_str = args.date or today
    html_out = m.render_html(records, date_str, ["vllm-omni", "vllm-omni-npu-ci"])
    out_path = Path(args.output)
    out_path.write_text(html_out, encoding="utf-8")

    # Summary breakdown so the user can eyeball the synthesis.
    by_ci = {b: 0 for b in m.CI_BUCKET_ORDER}
    for r in records:
        by_ci[r.ci_bucket] = by_ci.get(r.ci_bucket, 0) + 1

    print(f"Preview HTML written to {out_path}")
    print(f"  synthetic jobs: {len(records)}")
    print("  pipelines:      vllm-omni, vllm-omni-npu-ci")
    print(f"  branches:       {sorted({r.branch for r in records})}")
    print("  ci buckets:     " + ", ".join(f"{b}={by_ci.get(b, 0)}" for b in m.CI_BUCKET_ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
