# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P50/P95 + bootstrap CI, throughput, memory and separate operating points."""

import argparse
import json
from pathlib import Path

from benchmarks.zonos2.protocol import quantiles_ci


def summary(run: Path) -> dict:
    data = json.loads((run / "result.json").read_text())
    if data.get("profile_run"):
        raise ValueError("Profile runs must not enter latency comparison")
    rows = [row for row in data["rows"] if row["round"] >= 0]
    if data["backend"] != "official":
        raw_records = [
            json.loads(line)
            for file in (run / "record").glob("record-*.jsonl")
            for line in file.read_text().splitlines()
        ]
        for row in rows:
            if "ttfp_consumer_s" not in row:
                events = [
                    r for r in raw_records if r["kind"] == "dac" and r["request"] == row["request"] and r["samples"] > 0
                ]
                row["ttfp_consumer_s"] = row["ttfp_s"]
                row["ttfp_s"] = events[0]["time"] - row["start"]

    stats = {
        name: quantiles_ci([row[name] for row in rows])
        for name in ("latency_s", "ttfc_s", "ttfp_s", "rtf", "duration_s")
    }
    for row in rows:
        assert 0 <= row["ttfc_s"] <= row["ttfp_s"] <= row["latency_s"]
        assert (
            row["samples"]
            == (row["eos_frame"] if row["eos_frame"] is not None and row["eos_frame"] >= 0 else row["frames"]) * 512
        )
    waves = data["waves"]
    seconds = sum(wave["elapsed_s"] for wave in waves)
    throughput = {
        "requests_per_s": len(rows) / seconds,
        "audio_seconds_per_s": sum(row["duration_s"] for row in rows) / seconds,
        "raw_frames_per_s": sum(row["frames"] for row in rows) / seconds,
    }
    memory = json.loads((run / "gpu_samples.json").read_text())
    if data["backend"] == "official":
        ar_gpu = data["ar_gpu_ms"]
        ar_host = data["ar_host_ms"]
        dac = [row for row in data["dac_timings"] if row["audio_bytes"] > 0]
    else:
        records = [
            json.loads(line)
            for file in (run / "record").glob("record-*.jsonl")
            for line in file.read_text().splitlines()
        ]
        valid = [
            row
            for row in records
            if row["kind"] == "ar_timings" and not all(str(key).startswith("warmup") for key in row.get("requests", []))
        ]
        ar_gpu = [value for row in valid for value in row["gpu_ms"]]
        ar_host = [value for row in valid for value in row["host_ms"]]
        dac = [
            row
            for row in records
            if row["kind"] == "dac" and row["samples"] > 0 and not row["request"].startswith("warmup")
        ]
    return {
        "run": run.name,
        "backend": data["backend"],
        "concurrency": data["concurrency"],
        "sync": data.get("sync", False),
        "request_metrics": stats,
        "throughput": throughput,
        "peak_gpu_memory_mib": max(row["memory_mib"] for row in memory),
        "ar_device_interval_ms": quantiles_ci(ar_gpu),
        "ar_forward_host_ms": quantiles_ci(ar_host),
        "dac_device_interval_ms": quantiles_ci([row["gpu_ms"] for row in dac]),
        "dac_host_ms": quantiles_ci([row["host_ms"] for row in dac]),
        "failures": [row["label"] for row in rows if row["reached_cap"]],
        "stats_protocol": (
            f"{sum(row['round'] < 0 for row in data['rows'])} warmup requests excluded; "
            f"{len({row['round'] for row in rows})} measured rounds; "
            "request-level percentile bootstrap 2000 draws, seed42"
        ),
        "metric_boundary": (
            "Prepared-input offline driver. TTFC CPU-ready frame; TTFP CPU waveform ready. No HTTP/frontend costs."
        ),
        "torch": data["torch"],
        "params": data["params"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "rows": [summary(run) for run in args.runs],
        "notes": [
            "Official: frozen offline scheduler + official streaming vocoder observer; not an HTTP benchmark.",
            "Native: production AsyncOmni process/IPC path. Driver overhead is included in per-request latency.",
            "CUDA event intervals include submission gaps; profiler kernel times are reported separately.",
            "RNG implementations differ; same effective sampling knobs/seed does not imply identical draws.",
            "CI describes this fixed small corpus, not a production SLA or a kernel-only backend comparison.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
