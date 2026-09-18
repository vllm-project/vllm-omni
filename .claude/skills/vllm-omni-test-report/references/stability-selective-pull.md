# Selective stability-log pull (manifest-only)

Long-stability nightly runs (e.g. `nightly_stability_jobs_*` on H200 / H800)
produce per-job `.log` files that are routinely tens to hundreds of
megabytes each — and most of those jobs pass. Pulling every `.log` just to
discover pass/fail is wasteful, both in bandwidth and in laptop disk.

The selective-pull optimization fetches the run's **manifest** (`timing_summary.log`)
first, decides which jobs need their full logs, and only re-issues a tarball
for the failed ones. The full log is never downloaded for `OK` jobs; their
status still lands in the HTML report via a synthetic manifest-only row.

## Why this works

Every nightly run writes a tiny per-run rollup at
`<run_dir>/timing_summary.log`:

```text
=== Job timing summary ===
  stability_wan22  24h 47m  OK
  stability_qwen3_omni  1h 12m 34s  FAILED (exit 1)
  stability_flux  5m  TIMED OUT
Total wall time: 25h 53m 34s (3 jobs)
Failed jobs: 2/3
Result: one or more jobs failed. See logs under <dir>.
```

The rollup format is produced by
[`tools/run_jobs_common.sh::_run_jobs_print_timing_summary`][run_jobs_common],
which records `name + duration + status` per job after each job completes.

[run_jobs_common]: ../../../../../vllm-omni/tools/run_jobs_common.sh

Because every `<run_dir>` always has this rollup (when the nightly wrapper
finishes cleanly), the rollup is a reliable enough signal to skip the
gigabytes of `.log` for the `OK` entries.

## Two-phase pull workflow

```
cluster/container                              laptop
─────────────────                              ──────
Phase 1 (manifest):
  tar czf - <run_dir>/{timing_summary.log,jobs} ───────►  parse each timing_summary.log
                                                        → StabilityManifest

Phase 2 (selective):
  tar czf - <run_dir>/{failed_job.log,...}  ───────►     extract only the failures
                       ▲
                       └── only FAILED / TIMED OUT jobs from the manifest
```

## Components

| File | Role |
|------|------|
| [`scripts/stability_log_manifest.py`][manifest_py] | Parses `timing_summary.log` into a typed `StabilityManifest` (entries, OK / failed lists, total wall time, result line). Recursive discovery via `discover_manifests(log_dir)`. |
| [`scripts/selective_stability_pull.py`][selective_py] | Two-phase cluster/container pull driver: phase 1 packs only the rollup + `jobs/`; phase 2 packs only the failed-job `.log` files. Writes `logs/.selective_pull_manifest.json` sidecar. |
| `scripts/nightly_local_log_report.py` | Augmented `discover_job_logs(log_dir)` pipeline: after reading `<job>.log` files, it scans every `timing_summary.log` and injects synthetic `(manifest only)` rows for OK jobs whose `.log` is missing. Failed-but-not-pulled jobs are listed in a one-line note above the Local Summary body. |

[manifest_py]: ../scripts/stability_log_manifest.py
[selective_py]: ../scripts/selective_stability_pull.py

## CLI

`selective_stability_pull.py` reuses the regular log-sync plumbing
([`nightly-local-log-fetch.md`][fetch_md]) so H200 / H800 work the same
way as `tools/nightly/…` runs:

[fetch_md]: ../../vllm-omni-local-test/references/nightly-local-log-fetch.md

```bash
# H200 — direct SSH (already in container)
python3 scripts/selective_stability_pull.py \
    --ssh-host my_h200 \
    --repo-root ~/vllm-omni \
    --sync-scope stability

# H800 — srun + docker exec; --slurm-jobid can be auto-resolved via --slurm-user
python3 scripts/selective_stability_pull.py \
    --ssh-host h800.example.com \
    --slurm-user fq9hpsacuser07 \
    --slurm-jobid 12345 \
    --container-name omni_wy_24g \
    --repo-root ~/vllm-omni \
    --sync-scope stability
```

Flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--ssh-host` | required | SSH connection name (e.g. `my_h200`). |
| `--cluster-repo-root` | `/rebase/vllm-omni` | Path to the cluster checkout. |
| `--repo-root` | `~/vllm-omni` | Local repo root; pulled files land under `<repo>/logs/`. |
| `--sync-scope` | `stability` | `stability` / `default` / `all` — same vocabulary as the regular log-fetch flow. |
| `--include-local` | off | Also pull `nightly_jobs_local_*` (small; no phase-2 selective). |
| `--slurm-user` / `--slurm-jobid` / `--container-name` | off | H800 only. `--slurm-jobid` is auto-resolved from `--slurm-user` when omitted. |
| `--dry-run` | off | Print the planned `ssh` call without doing it. |
| `--print-sidecar` | off | After pull, dump the JSON sidecar to stdout. |

## Manifest-only mode in the report

After `selective_stability_pull.py` extracts only `timing_summary.log` for
every stability run, `nightly_local_log_report.py::emit_report_html`
performs an extra pass:

1. Calls `discover_manifests(log_dir)` → list of `StabilityManifest`.
2. For each manifest entry whose `<job>.log` is **not** present locally:
   - **OK** entries → emit a synthetic summary row:
     `<job_name> (manifest only)` with summary
     `"1 passed in <duration> (manifest only — log not pulled, status: OK)"`.
     Row is classified as `ok` (CSS class `summary-row--ok`); elapsed
     time, total / passed / failed counts use the manifest's totals.
   - **FAILED / TIMED OUT** entries → do **not** synthesize a row (no log
     available to derive pytest-shaped counts), but record the count in
     the manifest-only summary so the report surfaces "N failed job(s)
     marked manifest-only — log not pulled".

A one-line `<p class="manifest-only-note">` is rendered above the Local
Summary body whenever any manifest-only row exists, explaining the
suffix and listing the failed-but-not-pulled jobs.

## Sidecar (`logs/.selective_pull_manifest.json`)

The pull script writes a JSON sidecar recording every run, its manifest
results, and the bytes pulled / skipped. Downstream tooling reads it to
avoid re-parsing `timing_summary.log`:

```json
{
  "tool": "selective_stability_pull",
  "repo_root": "/home/wy/vllm-omni",
  "sync_scope": "stability",
  "include_local": false,
  "runs": [
    {
      "run_dir": "nightly_stability_jobs_20260808-031552",
      "ok_jobs": ["stability_wan22"],
      "failed_jobs": ["stability_qwen3_omni"],
      "pulled_logs": ["stability_qwen3_omni"],
      "skipped_logs": [{"job": "stability_wan22", "status": "OK"}],
      "pulled_bytes": 41298432
    }
  ],
  "total_pulled_bytes": 41298432,
  "total_skipped_bytes": 0
}
```

## When NOT to use selective pull

- **`local` / `general` runs** (`nightly_jobs_local_*`,
  `nightly_jobs_YYYYMMDD-*`) are typically short (minutes, not days) and
  the per-job `.log` is small enough that selective pull offers no
  meaningful savings. Pass `--include-local` only when explicitly asked;
  phase-2 selective is skipped for non-stability runs even when included.
- **In-progress runs** — the rollup is only written when the nightly
  wrapper completes, so a half-finished stability run cannot be selectively
  pulled yet. Either wait for completion or fall back to the regular
  `tar czf - nightly_stability_jobs_*` flow.
- **All jobs failed** — selective pull still helps (you skip the OK
  column that doesn't exist), but if every job failed you might as well
  pull everything and get the full pytest traces.

## Manual fallback

If the selective pull script is unavailable or misbehaving, the regular
log-fetch flow (steps 2–4 in
[`nightly-local-log-fetch.md`][fetch_md]) still works. After the merge,
the report script will detect the missing manifest (no
`timing_summary.log` under the run dir) and fall back to scanning
`*.log` files directly.

## Smoke test

```bash
TMPDIR=$(mktemp -d)
mkdir -p "$TMPDIR/logs/nightly_jobs/nightly_stability_jobs_20260808-031552/jobs"
cat > "$TMPDIR/logs/nightly_jobs/nightly_stability_jobs_20260808-031552/timing_summary.log" <<'EOF'
=== Job timing summary ===
  stability_wan22  24h 47m  OK
Total wall time: 24h 47m (1 jobs)
Result: all jobs finished OK. Logs: /rebase/vllm-omni/logs/nightly_stability_jobs_20260808-031552/*.log
EOF
echo "#!/bin/bash" > "$TMPDIR/logs/nightly_jobs/nightly_stability_jobs_20260808-031552/jobs/stability_wan22.sh"
chmod +x "$TMPDIR/logs/nightly_jobs/nightly_stability_jobs_20260808-031552/jobs/stability_wan22.sh"

python3 scripts/nightly_local_log_report.py --no-buildkite \
    --report-date 2026-08-20 \
    --log-dir "$TMPDIR/logs/nightly_jobs" \
    --html-report /tmp/manifest-only-test.html

grep -c "summary-row--ok" /tmp/manifest-only-test.html   # ≥ 1 (the manifest-only row is ok)
grep "manifest-only-note\|manifest only" /tmp/manifest-only-test.html | head -3
```
