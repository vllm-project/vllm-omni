# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent A/B decisions, with explicit quality and scope limitations."""

import argparse
import json
from pathlib import Path


def quality_gate(baseline, candidate):
    checks = {
        "en_wer": candidate["en_wer"] - baseline["en_wer"] <= 0.02,
        "zh_cer": candidate["zh_cer"] - baseline["zh_cer"] <= 0.01,
        "utmos": candidate["utmos_mean"] >= baseline["utmos_mean"] - 0.05,
        "speaker_cosine": candidate["speaker_cosine_mean"] >= baseline["speaker_cosine_mean"] - 0.02,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "limits": {"wer_increase": 0.02, "cer_increase": 0.01, "utmos_drop": 0.05, "cosine_drop": 0.02},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    quality = json.loads((root / "quality_ab.json").read_text())["aggregate"]
    baseline_quality = json.loads((root / "quality_baseline.json").read_text())["aggregate"]["native_c1"]
    eager = json.loads((root / "dac_eager/result.json").read_text())
    decisions: dict[str, object] = {}
    for variant in ("compile", "graph"):
        path = root / f"dac_{variant}/result.json"
        if not path.exists():
            decisions[variant] = {
                "accepted": False,
                "reason": "candidate failed; no fallback used",
                "failure": json.loads((root / f"dac_{variant}/failure.json").read_text()),
            }
            continue
        candidate = json.loads(path.read_text())
        speed = {
            n: eager["timing"][n]["wall_ms"]["p50"] / candidate["timing"][n]["wall_ms"]["p50"] for n in ("16", "20")
        }
        numeric = all(row["numeric_gate_pass"] for row in candidate["rows"])
        q = quality_gate(quality["dac_eager"], quality[f"dac_{variant}"])
        decisions[variant] = {
            "scope": "DAC component only",
            "median_speedup": speed,
            "numeric_gate": numeric,
            "quality_gate": q,
            "component_success": numeric and q["pass"] and min(speed.values()) > 1.05,
            "production_enabled": False,
            "reason": "Main bottleneck is AR dispatch; no runtime AR graph support is implied.",
        }
    sync = json.loads((root / "native_sync1/result.json").read_text())
    asynchronous = json.loads((root / "native_c1/result.json").read_text())
    sync_rows = [row for row in sync["rows"] if row["round"] >= 0]
    async_rows = [row for row in asynchronous["rows"] if row["round"] >= 0]
    from benchmarks.zonos2.protocol import quantiles_ci

    sync_time = quantiles_ci([row["ttfp_s"] for row in sync_rows])
    async_time = quantiles_ci([row["ttfp_s"] for row in async_rows])
    decisions["async_decode"] = {
        "scope": "Full native pipeline, sync vs async_chunk only",
        "sync_ttfp_s": sync_time,
        "async_ttfp_s": async_time,
        "median_ttfp_speedup": sync_time["p50"] / async_time["p50"],
        "quality_gate": quality_gate(quality["native_sync1"], baseline_quality),
        "production_changed": False,
        "reason": "Measures the already-supported P4 streaming configuration.",
    }
    decisions["full_runtime_ar_graph"] = json.loads((root / "ar_graph_feasibility.json").read_text())
    decisions["quality_caveat"] = "Baseline concurrency-4 zh_01 cap/CER failure remains; no global quality-pass claim."
    (root / "optimization_gates.json").write_text(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
