#!/usr/bin/env python3
"""
Fetch vllm-omni Buildkite nightly jobs (excluding Upload * Pipeline, :docker: Build image,
:email: Nightly Collection & Email, :pipeline: init), pull each job raw log,
and emit Markdown rows for a per-job pytest summary table.

Requires: BUILDKITE_TOKEN or BUILDKITE_API_TOKEN in the environment.
Optional: BUILDKITE_BUILD_NUMBER to pin a build; otherwise picks latest main build whose
message matches (?i)scheduled nightly.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from md_table import render_markdown_table  # noqa: E402
from pytest_log_parse import parse_pytest_log  # noqa: E402

ORG = "vllm"
PIPELINE = "vllm-omni"
BRANCH = "main"


def extract_ci_versions_from_log(log: str) -> dict[str, str]:
    """
    Best-effort ``vllm`` / ``vllm-omni`` version strings from a CI step log.

    Uses common patterns (pip list, pip install, Successfully installed, etc.).
    """
    out: dict[str, str] = {"vllm": "", "vllm_omni": ""}
    if not (log and log.strip()):
        return out
    sample = log if len(log) <= 900_000 else log[:550_000] + "\n" + log[-350_000:]

    omni_patterns = (
        re.compile(
            r"Requirement already satisfied:\s*vllm[_-]omni(?:==|>=|~=|!=|<=|>|<)?\s*([0-9][0-9A-Za-z.+-]*)",
            re.I,
        ),
        re.compile(
            r"(?:Downloading|Collecting)\s+vllm[_\-_]omni[^\s]*-([0-9][0-9A-Za-z.+-]*)",
            re.I,
        ),
        re.compile(
            r"vllm[_-]omni(?:\[[^\]]*\])?\s*[=~<>!]+\s*([0-9][0-9A-Za-z.+-]*)",
            re.I,
        ),
        re.compile(
            r"Successfully installed[^\n]*\bvllm[_-]omni-([0-9][^\s,]*)",
            re.I,
        ),
        re.compile(r"^\s*vllm[_-]omni\s+([0-9][0-9A-Za-z.+-]+)\s*$", re.I | re.M),
        re.compile(
            r"['\"]vllm[_-]omni['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            re.I,
        ),
    )
    for pat in omni_patterns:
        m = pat.search(sample)
        if m:
            out["vllm_omni"] = m.group(1).strip()
            break

    vllm_patterns = (
        re.compile(
            r"Requirement already satisfied:\s*vllm(?:==|>=|~=|!=|<=|>|<)?\s*([0-9][0-9A-Za-z.+-]*)",
            re.I,
        ),
        re.compile(
            r"Requirement already satisfied:\s*vllm\s+in\s+[^\n(]+\(([0-9][0-9A-Za-z.+-]+)\)",
            re.I,
        ),
        re.compile(
            r"(?:^|[\s/])vllm\s*[=~<>!]+\s*([0-9][0-9A-Za-z.+-]*)(?![^\n]*omni)",
            re.I,
        ),
        re.compile(
            r"Successfully installed[^\n]*?(?<![\w-])vllm-(\d[\w.+-]*)(?!-omni)",
            re.I,
        ),
        re.compile(r"^\s*vllm\s+([0-9][0-9A-Za-z.+-]+)\s*$", re.I | re.M),
    )
    for pat in vllm_patterns:
        m = pat.search(sample)
        if m:
            cand = m.group(1).strip()
            if cand.lower() != "omni" and not cand.lower().startswith("omni"):
                out["vllm"] = cand
                break

    return out


# Ignore artifact/upload steps when reporting test outcomes.
UPLOAD_PIPELINE_RE = re.compile(r"^Upload .+ Pipeline$", re.IGNORECASE)
# Non-test steps: omit from per-job pytest table (no useful pytest footer).
SKIP_NON_PYTEST_JOB_RES = (
    re.compile(r"^:docker:\s*Build image\s*$", re.IGNORECASE),
    re.compile(r"^:email:\s*Nightly Collection\s*&\s*Email\s*$", re.IGNORECASE),
    re.compile(r"^:pipeline:\s*init\s*$", re.IGNORECASE),
    # :bar_chart: Testcase Statistics aggregates results from prior jobs and
    # does not run pytest; its log has no pytest footer so it surfaces as
    # ``unknown`` in the Daily Focus Buildkite-jobs card.
    re.compile(r"^:bar_chart:\s*Testcase Statistics\s*$", re.IGNORECASE),
    # :github: Resolve skip-ci (docs / skip marks) & upload pipeline is a
    # GitHub pipeline-resolution step that uploads commit statuses; no pytest.
    re.compile(r"^:github:\s*Resolve skip-ci.*upload pipeline\s*$", re.IGNORECASE),
)


def http_json(url: str, token: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def http_text_tail(
    url: str,
    token: str,
    *,
    max_read: int = 32_000_000,
    tail_keep: int = 10_000_000,
) -> str:
    """Download log; if it exceeds max_read bytes, keep only the last tail_keep bytes."""
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    last_err: Exception | None = None
    for attempt in range(3):
        buf = bytearray()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                while True:
                    chunk = resp.read(262_144)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > max_read:
                        buf = buf[-tail_keep:]
            return buf.decode("utf-8", errors="replace")
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < 2:
                time.sleep(min(8, 2**attempt))
    assert last_err is not None
    raise last_err


def resolve_latest_scheduled_nightly_number(
    token: str,
    *,
    org: str = ORG,
    pipeline: str = PIPELINE,
    branch: str = BRANCH,
) -> int | None:
    """Return latest **scheduled nightly** build number on ``branch``.

    Falls back to the most recent green/red ``branch`` build if no scheduled
    nightly message matches within the first 50 builds (the upstream
    ``releases`` also recommends this fallback for rebase / API-cache edge cases).
    Returns ``None`` only if the pipeline has zero builds on ``branch``.
    """
    url = f"https://api.buildkite.com/v2/organizations/{org}/pipelines/{pipeline}/builds?branch={branch}&per_page=50"
    builds = http_json(url, token)
    for b in builds:
        if re.search(r"scheduled\s+nightly", b.get("message") or "", re.I):
            return int(b["number"])
    # Fallback: most recent main build (any state) — last entry is newest.
    if builds:
        return int(builds[0]["number"])
    return None


def latest_scheduled_nightly_number(
    token: str,
    *,
    org: str = ORG,
    pipeline: str = PIPELINE,
    branch: str = BRANCH,
) -> int:
    n = resolve_latest_scheduled_nightly_number(token, org=org, pipeline=pipeline, branch=branch)
    if n is None:
        sys.exit(f"No scheduled nightly build found on {org}/{pipeline} (branch={branch}, per_page=50).")
    return n


def fetch_nightly_build(
    token: str,
    build_number: int | None,
    *,
    org: str = ORG,
    pipeline: str = PIPELINE,
    branch: str = BRANCH,
) -> dict[str, Any]:
    """Load build JSON; ``build_number=None`` = latest scheduled nightly on ``main``."""
    if build_number is None:
        n = resolve_latest_scheduled_nightly_number(token, org=org, pipeline=pipeline, branch=branch)
        if n is None:
            raise RuntimeError(f"No scheduled nightly build found on {org}/{pipeline} (branch={branch}, per_page=50).")
    else:
        n = build_number
    url = f"https://api.buildkite.com/v2/organizations/{org}/pipelines/{pipeline}/builds/{n}"
    return http_json(url, token)


def collect_nightly_job_log_analyses(
    build: dict[str, Any],
    token: str,
    *,
    org: str = ORG,
    pipeline: str = PIPELINE,
) -> list[dict[str, Any]]:
    """
    One record per reportable job: name, state, step_link, raw_url,
    info (``parse_pytest_log`` output) or log_error.
    """
    build_no = int(build["number"])
    commit_full = (build.get("commit") or "").strip()
    build_commit_short = commit_full[:12] if commit_full else ""
    jobs = build.get("jobs") or []
    report_jobs = [j for j in jobs if not should_skip_job(j.get("name") or "")]
    report_jobs.sort(key=lambda x: (x.get("name") or ""))
    out: list[dict[str, Any]] = []
    for j in report_jobs:
        jid = j.get("id") or ""
        name = j.get("name") or ""
        state = j.get("state") or ""
        link = job_anchor(build_no, jid, org=org, pipeline=pipeline)
        raw_url = j.get("raw_log_url") or j.get("log_url")
        rec: dict[str, Any] = {
            "name": name,
            "state": state,
            "step_link": link,
            "raw_url": raw_url,
            "info": None,
            "log_error": None,
            "build_commit_short": build_commit_short,
            "ci_versions": None,
        }
        if not raw_url:
            out.append(rec)
            continue
        try:
            log = http_text_tail(str(raw_url), token)
            rec["info"] = parse_pytest_log(log)
            rec["ci_versions"] = extract_ci_versions_from_log(log)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            rec["log_error"] = str(e)
        out.append(rec)
    return out


def should_skip_job(name: str) -> bool:
    n = (name or "").strip()
    if UPLOAD_PIPELINE_RE.match(n):
        return True
    return any(r.match(n) for r in SKIP_NON_PYTEST_JOB_RES)


def job_anchor(
    build_no: int,
    job_id: str,
    *,
    org: str = ORG,
    pipeline: str = PIPELINE,
) -> str:
    return f"https://buildkite.com/{org}/{pipeline}/builds/{build_no}#{job_id}"


def md_cell(s: str) -> str:
    return (s or "").replace("|", "/")


def append_markdown_rows_for_nightly_job(rows: list[list[str]], rec: dict[str, Any]) -> None:
    """Append Markdown table rows for one Buildkite job record."""
    name = md_cell(rec["name"])
    state = rec["state"]
    link = rec["step_link"]
    em_dash = "—"

    if not rec["raw_url"]:
        rows.append([name, f"{md_cell(state)} — no log URL", em_dash, em_dash, f"[open]({link})"])
        return
    if rec["log_error"]:
        rows.append(
            [
                name,
                f"{md_cell(state)} — log fetch failed",
                em_dash,
                em_dash,
                f"[open]({link})",
            ]
        )
        return

    info = rec["info"]
    assert info is not None
    summary = info["summary"]
    fails = info["failed_nodes"]
    errors = info["error_nodes"]

    if summary is None and not fails and not errors:
        rows.append(
            [
                name,
                f"{md_cell(state)} — non-pytest or log truncated",
                em_dash,
                em_dash,
                f"[open]({link})",
            ]
        )
        return

    agg_result = md_cell(state or "unknown")
    if fails or errors:
        if fails and errors:
            agg_result = "failed/error"
        elif fails:
            agg_result = "failed"
        else:
            agg_result = "error"
    elif summary and re.search(r"\b[1-9]\d*\s+failed\b", summary, re.I):
        agg_result = "failed"
    elif summary and re.search(r"\b[1-9]\d*\s+error\b", summary, re.I):
        agg_result = "error"
    elif (state == "passed" or state == "finished") and not fails and not errors:
        agg_result = "passed"

    if agg_result == "passed" and not fails and not errors:
        rows.append([name, "passed", em_dash, em_dash, f"[open]({link})"])
        return

    summ_short = md_cell((summary or "")[:260])
    if fails or errors:
        hint = "This step has failed/error cases; see rows below."
    elif agg_result in ("failed", "error", "failed/error"):
        hint = (
            "Build state is failed, but no FAILED/ERROR lines were parsed from the log "
            "(possibly truncated or non-pytest)."
        )
    else:
        hint = em_dash

    rows.append([name, md_cell(agg_result), summ_short, hint, f"[open]({link})"])

    for node in fails:
        rows.append(
            [
                md_cell(node),
                "failed",
                md_cell(info["failed_reasons"].get(node, "")),
                md_cell(info["failure_analyses"].get(node, "")),
                f"[open]({link})",
            ]
        )
    for node in errors:
        rows.append(
            [
                md_cell(node),
                "error",
                md_cell(info["error_reasons"].get(node, "")),
                md_cell(info["error_analyses"].get(node, "")),
                f"[open]({link})",
            ]
        )


def emit_markdown(build: dict[str, Any], token: str, *, org: str = ORG, pipeline: str = PIPELINE) -> None:
    rows: list[list[str]] = []
    for rec in collect_nightly_job_log_analyses(build, token, org=org, pipeline=pipeline):
        append_markdown_rows_for_nightly_job(rows, rec)
    print("## Per-job test execution (pytest)")
    print()
    print(
        render_markdown_table(
            ["Job / test node", "Result", "Reason (from log)", "Heuristic analysis", "Step link"],
            rows,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown table: nightly jobs vs pytest outcomes.")
    parser.add_argument(
        "--build",
        type=int,
        default=None,
        help="Build number (default: latest scheduled nightly on main).",
    )
    args = parser.parse_args()

    token = (os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN") or "").strip()
    if not token:
        print(
            "BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is not set.",
            file=sys.stderr,
        )
        sys.exit(2)

    build = fetch_nightly_build(token, args.build)
    emit_markdown(build, token)


if __name__ == "__main__":
    main()
