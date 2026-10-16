# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Merge per-stage startup numbers from a bench_startup.py log with its BENCH_JSON line.

The stage engine cores run in subprocesses and report weight-loading / init timings
only through log lines such as:

  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... Loading weights took 39.66 seconds
  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... Model loading took 21.45 GiB memory and 40.59 seconds
  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... init engine (profile, create kv cache, warmup model) took 7.00 s
  (StageEngineCoreProc_stage0_replica0 pid=1) INFO ... Filesystem type for checkpoints: NFS4.
      Checkpoint size: 34.52 GiB. Available RAM: 22.76 GiB.
  INFO ... [Omni] AsyncOmniEngine initialized in 159.74 seconds

Usage:
  python benchmarks/mammoth_moda2/parse_startup_log.py run.log [more.log ...] [--markdown] [--json-out summary.json]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from typing import Any

STAGE = r"\(StageEngineCoreProc_stage(?P<stage>\d+)_replica\d+ pid=\d+\)"
PATTERNS = {
    "weights_load_s": re.compile(STAGE + r".*Loading weights took (?P<v>[\d.]+) seconds"),
    "model_load_s": re.compile(STAGE + r".*Model loading took (?P<mem>[\d.]+) GiB memory and (?P<v>[\d.]+) seconds"),
    "engine_init_s": re.compile(
        STAGE + r".*init engine \(profile, create kv cache, warmup model\) took (?P<v>[\d.]+) s"
    ),
    # Only present when the stage does not run with enforce_eager.
    "graph_capture_s": re.compile(
        STAGE + r".*Graph capturing finished in (?P<v>[\d.]+) secs, took (?P<mem>[\d.]+) GiB"
    ),
    "compile_s": re.compile(STAGE + r".*torch\.compile takes (?P<v>[\d.]+) s in total"),
    "ckpt_fs": re.compile(
        STAGE
        + r".*Filesystem type for checkpoints: (?P<fs>\w+)\. Checkpoint size: (?P<size>[\d.]+) GiB\."
        + r" Available RAM: (?P<ram>[\d.]+) GiB"
    ),
}
ENGINE_READY = re.compile(r"AsyncOmniEngine initialized in (?P<v>[\d.]+) seconds")
PREFETCH_SKIP = re.compile(STAGE + r".*exceeds 90% of available RAM")
COMPILE_UNSUPPORTED = re.compile(STAGE + r".*`torch\.compile` is turned on, but the model .* does not support it")

# Startup timeline: (milestone key, regex). Timestamps in vllm logs are "MM-DD HH:MM:SS" (1 s resolution).
TS = re.compile(r"(?:INFO|WARNING|ERROR) (?P<ts>\d\d-\d\d \d\d:\d\d:\d\d)")
MILESTONES = [
    ("omni_init_start", re.compile(r"\[Omni\] Initializing with model")),
    ("stage{stage}_launch", re.compile(r"Stage-(?P<stage>\d+) set runtime devices")),
    ("stage{stage}_proc_up", re.compile(STAGE + r".*world_size=\d+ rank=\d+")),
    ("stage{stage}_load_start", re.compile(STAGE + r".*Starting to load model")),
    ("stage{stage}_load_done", re.compile(STAGE + r".*Loading weights took")),
    ("stage{stage}_init_done", re.compile(STAGE + r".*init engine \(profile, create kv cache, warmup model\) took")),
    ("stage{stage}_ready", re.compile(r"\[StageRuntime\] Stage (?P<stage>\d+) initialized")),
    ("engine_ready", re.compile(r"AsyncOmniEngine initialized in")),
]


def _ts_seconds(ts: str) -> int:
    # Use a leap year because engine timestamps do not include the year.
    value = datetime.strptime("2000-" + ts, "%Y-%m-%d %H:%M:%S")
    return int((value - datetime(2000, 1, 1)).total_seconds())


def timeline(lines: list[str]) -> dict:
    """Return milestone timestamps (s, relative to omni_init_start) and derived per-stage phase durations."""
    marks: dict[str, int] = {}
    for line in lines:
        tsm = TS.search(line)
        if not tsm:
            continue
        for key, pat in MILESTONES:
            m = pat.search(line)
            if m:
                k = key.format(stage=m.group("stage")) if "{stage}" in key else key
                marks.setdefault(k, _ts_seconds(tsm.group("ts")))
                break
    if "omni_init_start" not in marks:
        return {}
    t0 = marks["omni_init_start"]
    rel = {k: (v - t0) % (366 * 86400) for k, v in marks.items()}
    phases: dict[str, Any] = {}
    names = ("launch", "proc_up", "load_start", "load_done", "init_done", "ready")
    for s in sorted({k.split("_")[0] for k in rel if k.startswith("stage")}):
        found = {name: rel.get(f"{s}_{name}") for name in names}
        if any(v is None for v in found.values()):
            continue
        ts = {name: int(v) for name, v in found.items() if v is not None}
        phases[s] = {
            "spawn_import_config_s": ts["proc_up"] - ts["launch"],
            "device_dist_init_s": ts["load_start"] - ts["proc_up"],
            "model_and_weights_load_s": ts["load_done"] - ts["load_start"],
            "profile_kv_warmup_s": ts["init_done"] - ts["load_done"],
            "ready_handoff_s": ts["ready"] - ts["init_done"],
            "total_s": ts["ready"] - ts["launch"],
        }
    if "engine_ready" in rel:
        last_ready = max((v for k, v in rel.items() if k.endswith("_ready") and k != "engine_ready"), default=0)
        phases["post_stages_wiring_s"] = rel["engine_ready"] - last_ready
        phases["engine_ready_s"] = rel["engine_ready"]
    return {"marks_rel_s": rel, "phases": phases}


def parse(path: str) -> dict:
    stages: dict[str, dict] = {}
    rec: dict[str, Any] = {"log": path, "stages": stages}
    with open(path, errors="replace") as f:
        lines = f.readlines()
    rec["timeline"] = timeline(lines)
    for line in lines:
        if line.startswith("BENCH_JSON "):
            rec["bench"] = json.loads(line[len("BENCH_JSON ") :])
            continue
        m = ENGINE_READY.search(line)
        if m:
            rec["engine_ready_s_reported"] = float(m.group("v"))
            continue
        m = PREFETCH_SKIP.search(line)
        if m:
            stages.setdefault(m.group("stage"), {})["page_cache_prefetch_skipped"] = True
            continue
        m = COMPILE_UNSUPPORTED.search(line)
        if m:
            stages.setdefault(m.group("stage"), {})["torch_compile_unsupported"] = True
            continue
        for key, pat in PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            st = stages.setdefault(m.group("stage"), {})
            if key == "ckpt_fs":
                st["checkpoint_fs"] = m.group("fs")
                st["checkpoint_size_gib"] = float(m.group("size"))
                st["available_ram_gib"] = float(m.group("ram"))
            elif key == "graph_capture_s":
                st["graph_capture_s"] = float(m.group("v"))
                st["graph_capture_mem_gib"] = float(m.group("mem"))
            elif key == "model_load_s":
                st["model_load_s"] = float(m.group("v"))
                st["model_load_mem_gib"] = float(m.group("mem"))
                # "Model loading took" covers model construction + weight loading; "Loading weights took" only
                # the weights. The difference also includes loader setup/teardown overhead.
                if "weights_load_s" in st:
                    st["model_setup_estimate_s"] = round(st["model_load_s"] - st["weights_load_s"], 2)
            else:
                st[key] = float(m.group("v"))
            break
    rec["status"] = "complete" if "bench" in rec else "incomplete"
    return rec


def _table(columns: dict[str, str], records: list[dict]) -> str:
    def row(values):
        return "| " + " | ".join("" if v is None else str(v) for v in values) + " |"

    lines = [row(columns.values()), row(["---"] * len(columns))]
    lines.extend(row(record.get(key) for key in columns) for record in records)
    return "\n".join(lines)


def to_markdown(recs: list[dict]) -> str:
    summary, stages, phases, warnings = [], [], [], []
    for r in recs:
        b = r.get("bench", {})
        label = b.get("label", r["log"])
        if r.get("status") == "incomplete":
            warnings.append(f"Incomplete log: {r['log']} (no BENCH_JSON).")
        summary.append({"label": label, **b.get("summary_s", {})})
        for sid, st in sorted(r["stages"].items()):
            stages.append({"label": label, "stage": sid, **st})
            if st.get("torch_compile_unsupported"):
                stages[-1]["compile_s"] = "unsupported"
        ph = r.get("timeline", {}).get("phases", {})
        for sid, values in sorted(ph.items()):
            if sid.startswith("stage"):
                phases.append({"label": label, "stage": sid, **values})
        if "post_stages_wiring_s" in ph:
            phases.append({"label": label, "stage": "after last stage", "total_s": ph["post_stages_wiring_s"]})
    tables = [
        (
            {
                "label": "label",
                "imports": "imports (s)",
                "engine_init": "engine ready (s)",
                "first_request": "first request (s)",
                "steady_request_avg": "subsequent avg (s)",
                "time_to_first_image_from_process_start": "process→first image (s)",
                "host_rss_peak_gib": "sampled host RSS (GiB)",
                "gpu_used_peak_gib": "sampled GPU used (GiB)",
            },
            summary,
        ),
        (
            {
                "label": "label",
                "stage": "stage",
                "model_setup_estimate_s": "model setup estimate (s)",
                "weights_load_s": "weights load (s)",
                "compile_s": "compile (s)",
                "graph_capture_s": "graph capture (s)",
                "engine_init_s": "profile/KV/warmup reported (s)",
            },
            stages,
        ),
        (
            {
                "label": "label",
                "stage": "stage",
                "spawn_import_config_s": "spawn/import (s)",
                "device_dist_init_s": "device init (s)",
                "model_and_weights_load_s": "model+weights load (s)",
                "profile_kv_warmup_s": "profile/KV/warmup interval (s)",
                "total_s": "total (s)",
            },
            phases,
        ),
    ]
    return "\n\n".join(warnings + [_table(columns, rows) for columns, rows in tables])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--require-complete", action="store_true", help="fail if any log has no BENCH_JSON")
    args = ap.parse_args()
    recs = [parse(p) for p in args.logs]
    if args.require_complete and any(r["status"] != "complete" for r in recs):
        ap.error("incomplete benchmark log: missing BENCH_JSON")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(recs, f, indent=2)
    if args.markdown:
        print(to_markdown(recs))
    else:
        json.dump(recs, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
