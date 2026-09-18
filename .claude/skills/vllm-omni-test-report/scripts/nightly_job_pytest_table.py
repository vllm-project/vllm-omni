#!/usr/bin/env python3
"""
Shared Buildkite nightly helpers (data-fetch only; no Markdown rendering).

Exports:
  - constants: ``ORG`` / ``PIPELINE`` / ``BRANCH``
  - :func:`resolve_latest_scheduled_nightly_number` — pick the latest scheduled-nightly
    build number on ``branch`` (falls back to most recent green/red ``branch`` build)
  - :func:`fetch_nightly_build` — load a build JSON
  - :func:`collect_nightly_job_log_analyses` — pull every reportable job's raw log
    and run it through :func:`pytest_log_parse.parse_pytest_log`

Excluded jobs (not pytest, no useful footer): ``Upload * Pipeline``, ``:docker: Build image``,
``:email: Nightly Collection & Email``, ``:pipeline: init``, ``:bar_chart: Testcase Statistics``,
``:github: Resolve skip-ci…upload pipeline``.

Requires: ``BUILDKITE_TOKEN`` / ``BUILDKITE_API_TOKEN`` in the environment for any HTTP call.
"""

from __future__ import annotations

import http.client
import json
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
