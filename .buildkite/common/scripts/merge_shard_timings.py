#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Merge per-job timing artifacts into one per-step p80 profile.

Job files look like ``timings/jobs/<job>-<pid>.json`` with ``file_seconds``,
``job_id``, ``label``, ``parallel_job``, and ``queue``. Only builds whose state
is ``passed`` or ``finished`` contribute. Each selected build uses its raw job
files, never a previously merged profile. Across builds, the profile stores the
p80 of those observations under the step label (a ``Shard N/M`` suffix is
removed).

The uploader calls ``prepare_profile_from_history``, which downloads recent
``timings/jobs/*.json`` via ``buildkite-agent``. Ready planning excludes
``main`` and therefore uses other pull-request builds, whose durations include
whatever those changes did. Merge planning stays on ``main``. The Buildkite
builds API is read in pages of 100.
The download stops after enough builds with job files, after a run of builds
with none, or at a short time budget. A newer job whose Buildkite state is
``timed_out`` raises that step's next plan by one job, up to the shard cap. A
newer passed job clears the raise. A failed job that did not time out does not
add a job. Missing history is not an error: planning stays on one job.

Within one build, invocations that share a job id are summed. Retries of the
same ``parallel_job`` index then keep the max. Different indexes are
complementary pytest-shard slices of a file, so those durations are summed.
Jobs with no index keep the older max-across-jobs rule. A step later ignores a
profile recorded on a different agent queue.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from runtime_shard import MAX_AUTOMATIC_SHARDS

DEFAULT_BUILDS = 20
DEFAULT_BRANCH = "main"
# Many passed builds have no timing artifacts. Scan a wider window, then keep
# the newest ``DEFAULT_BUILDS`` that actually recorded raw job files.
HISTORY_CANDIDATE_FACTOR = 5
# Buildkite rejects per_page above 100. Follow Link rel="next" for the rest.
API_PAGE_SIZE = 100
MAX_API_PAGES = 10
# Timeout decisions only need the newest builds, and this request does not download artifacts.
TIMEOUT_SCAN_PAGES = 3
# A cold window has no timing artifacts. Stop before upload issues ~100 serial lookups.
MAX_CONSECUTIVE_MISSES = 10
HISTORY_LOOKUP_BUDGET_SECONDS = 90
HISTORY_DOWNLOAD_TIMEOUT_SECONDS = 20
JOB_GLOB = "timings/jobs/*.json"
SCHEMA = "vllm_omni.shard_timings.merged.v1"
SUCCESS_STATES = frozenset({"passed", "finished"})
_SHARD_SUFFIX = re.compile(r" · Shard(?: %N/%t| \d+/\d+)$")
_SHARD_TOTAL = re.compile(r" · Shard \d+/(\d+)$")
_SUCCESS_JOB_STATES = frozenset({"passed", "finished"})
_NONTERMINAL_JOB_STATES = frozenset(
    {
        "pending",
        "waiting",
        "blocked",
        "unblocked",
        "limiting",
        "limited",
        "scheduled",
        "assigned",
        "accepted",
        "running",
        "canceling",
        "cancelling",
        "timing_out",
    }
)
_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?', re.IGNORECASE)


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (Postgres ``percentile_cont``)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    below = int(position)
    if below >= len(ordered) - 1:
        return ordered[-1]
    frac = position - below
    return ordered[below] + (ordered[below + 1] - ordered[below]) * frac


def _base_label(label: str) -> str:
    return _SHARD_SUFFIX.sub("", label)


def _parallel_job(payload: dict) -> int | None:
    raw = payload.get("parallel_job")
    if isinstance(raw, bool) or raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def load_job_record(path: Path) -> tuple[str, str, dict[str, float], int | None, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected object")
    raw = payload.get("file_seconds")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: missing file_seconds")
    seconds = {str(key).replace("\\", "/"): float(value) for key, value in raw.items() if float(value) > 0}
    job_id = str(payload.get("job_id") or path.stem)
    label = _base_label(str(payload.get("label") or ""))
    queue = str(payload.get("queue") or "")
    return job_id, label, seconds, _parallel_job(payload), queue


def _max_files(attempts: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for files in attempts:
        for name, value in files.items():
            merged[name] = max(merged.get(name, 0.0), value)
    return merged


def _sum_files(parts: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for files in parts:
        for name, value in files.items():
            merged[name] = merged.get(name, 0.0) + value
    return merged


def _dominant_queue(queues: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for queue in queues:
        counts[queue] += 1
    return max(counts, key=lambda queue: (counts[queue], bool(queue), queue))


def build_step_details(job_files: list[Path]) -> dict[str, tuple[dict[str, float], str]]:
    """Per-step ``(file seconds, queue)`` for one build.

    Same-job invocations are summed. A ``parallel_job`` index then keeps the
    max across retries and sums those indexes, because pytest-shard gives each
    index a different slice of the file. Jobs without an index stay on the max
    rule and are ignored when indexed jobs exist for that step.
    """
    per_job: dict[tuple[str, str], tuple[dict[str, float], int | None, str]] = {}
    for path in job_files:
        job_id, label, seconds, parallel_job, queue = load_job_record(path)
        current = per_job.get((job_id, label))
        if current is None:
            per_job[(job_id, label)] = (dict(seconds), parallel_job, queue)
            continue
        files, known_index, known_queue = current
        for name, value in seconds.items():
            files[name] = files.get(name, 0.0) + value
        if known_index is None:
            known_index = parallel_job
        if not known_queue:
            known_queue = queue
        per_job[(job_id, label)] = (files, known_index, known_queue)

    grouped: dict[str, list[tuple[dict[str, float], int | None, str]]] = defaultdict(list)
    for (_, label), item in per_job.items():
        grouped[label].append(item)

    details: dict[str, tuple[dict[str, float], str]] = {}
    for label, jobs in grouped.items():
        indexed = [job for job in jobs if job[1] is not None]
        chosen = indexed or jobs
        queue = _dominant_queue([job[2] for job in chosen])
        chosen = [job for job in chosen if job[2] == queue]
        if indexed:
            by_index: dict[int, list[dict[str, float]]] = defaultdict(list)
            for files, parallel_job, _queue in chosen:
                assert parallel_job is not None
                by_index[parallel_job].append(files)
            files = _sum_files([_max_files(attempts) for attempts in by_index.values()])
        else:
            files = _max_files([job[0] for job in chosen])
        if files:
            details[label] = (files, queue)
    return details


def build_step_observations(job_files: list[Path]) -> dict[str, dict[str, float]]:
    """One build's per-step file seconds from raw job files."""
    return {label: files for label, (files, _queue) in build_step_details(job_files).items()}


def merge_build_observations(per_build: list[dict[str, float]], *, q: float = 0.8) -> dict[str, float]:
    by_file: dict[str, list[float]] = defaultdict(list)
    for observed in per_build:
        for file, seconds in observed.items():
            by_file[file].append(float(seconds))
    return {file: round(percentile(values, q), 3) for file, values in sorted(by_file.items())}


def merge_step_observations(
    per_build: list[dict[str, dict[str, float]]],
    *,
    q: float = 0.8,
) -> dict[str, dict[str, float]]:
    by_step: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for observed in per_build:
        for label, files in observed.items():
            for file, seconds in files.items():
                by_step[label][file].append(float(seconds))
    return {
        label: {file: round(percentile(values, q), 3) for file, values in sorted(files.items())}
        for label, files in sorted(by_step.items())
    }


def merge_build_details(
    per_build: list[dict[str, tuple[dict[str, float], str]]],
    *,
    q: float = 0.8,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """p80 per step, using only the queue that produced the most builds."""
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for observed in per_build:
        for label, (files, queue) in observed.items():
            grouped[(label, queue)].append(files)
    by_label: dict[str, list[str]] = defaultdict(list)
    for (label, queue), observations in grouped.items():
        by_label[label].extend([queue] * len(observations))
    steps: dict[str, dict[str, float]] = {}
    queues: dict[str, str] = {}
    for label, label_queues in sorted(by_label.items()):
        queue = _dominant_queue(label_queues)
        steps[label] = merge_build_observations(grouped[(label, queue)], q=q)
        if queue:
            queues[label] = queue
    return steps, queues


def write_merged_profile(
    steps: dict[str, dict[str, float]],
    output: Path,
    *,
    builds: int,
    build_ids: list[str] | None = None,
    step_queues: dict[str, str] | None = None,
    timeout_floors: dict[str, int] | None = None,
    timeout_floor_queues: dict[str, str] | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "measurement": "p80_over_passed_builds_of_raw_job_file_seconds",
        "builds": builds,
        "build_ids": build_ids or [],
        "steps": steps,
        "step_queues": {label: queue for label, queue in sorted((step_queues or {}).items()) if queue},
        "timeout_floors": {label: int(count) for label, count in sorted((timeout_floors or {}).items())},
        "timeout_floor_queues": {
            label: queue for label, queue in sorted((timeout_floor_queues or {}).items()) if queue
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def collect_local_job_files(root: Path) -> list[Path]:
    return sorted(root.glob(JOB_GLOB))


def merge_local_jobs(root: Path, output: Path) -> Path:
    job_files = collect_local_job_files(root)
    if not job_files:
        raise FileNotFoundError(f"no job timing files under {root / 'timings' / 'jobs'}")
    details = build_step_details(job_files)
    observed = {label: files for label, (files, _queue) in details.items()}
    queues = {label: queue for label, (_files, queue) in details.items() if queue}
    return write_merged_profile(observed, output, builds=1, build_ids=[], step_queues=queues)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _pipeline_url() -> str | None:
    org = _env("BUILDKITE_ORGANIZATION_SLUG")
    slug = _env("BUILDKITE_PIPELINE_SLUG")
    if not org or not slug:
        return None
    return f"https://buildkite.com/{org}/{slug}"


def _api_builds_url(branch: str, *, state: str = "passed") -> str | None:
    org = _env("BUILDKITE_ORGANIZATION_SLUG")
    slug = _env("BUILDKITE_PIPELINE_SLUG")
    if not org or not slug:
        return None
    params: dict[str, str] = {}
    if state:
        params["state"] = state
    params["per_page"] = str(API_PAGE_SIZE)
    if branch:
        params["branch"] = branch
    query = urllib.parse.urlencode(params)
    return f"https://api.buildkite.com/v2/organizations/{org}/pipelines/{slug}/builds?{query}"


def _next_link(header: str) -> str | None:
    match = _NEXT_LINK.search(header)
    return match.group(1) if match else None


def _fetch_page(url: str, *, token: str | None = None) -> tuple[str, str | None]:
    headers = {"User-Agent": "vllm-omni-merge-shard-timings"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        return body, _next_link(response.headers.get("Link", ""))


def _fetch_text(url: str, *, token: str | None = None) -> str:
    body, _next = _fetch_page(url, token=token)
    return body


def _body(result: object) -> str:
    if isinstance(result, tuple):
        return str(result[0])
    return str(result)


def _read_listing(fetch: Callable[..., object], url: str, token: str) -> tuple[str, str | None]:
    """Accept a body or ``(body, next_url)``. A bare body is a single page."""
    result = fetch(url, token=token)
    body = _body(result)
    if isinstance(result, tuple):
        nxt = result[1] if len(result) > 1 else None
        return body, (str(nxt) if nxt else None)
    return body, None


def list_recent_build_ids(
    *,
    branch: str,
    limit: int,
    current_build_id: str = "",
    exclude_branches: frozenset[str] | set[str] = frozenset(),
    fetch: Callable[..., object] = _fetch_text,
) -> list[tuple[str, str]]:
    """Return ``(build_id, build_number)`` newest first, excluding the current build.

    ``branch=""`` lists every branch of this pipeline. ``exclude_branches`` drops
    named branches after that listing (ready planning excludes ``main``).
    """
    excluded = frozenset(exclude_branches)
    token = _env("BUILDKITE_API_TOKEN") or _env("BUILDKITE_TOKEN")
    api = _api_builds_url(branch)
    if api and token:
        found: list[tuple[str, str]] = []
        url: str | None = api
        pages = 0
        reader = _fetch_page if fetch is _fetch_text else fetch
        try:
            while url and len(found) < limit and pages < MAX_API_PAGES:
                pages += 1
                body, next_url = _read_listing(reader, url, token)
                payload = json.loads(body)
                if not isinstance(payload, list):
                    break
                for entry in payload:
                    if not isinstance(entry, dict):
                        continue
                    build_id = str(entry.get("id") or "")
                    number = entry.get("number")
                    state = str(entry.get("state") or "")
                    entry_branch = str(entry.get("branch") or entry.get("branch_name") or "")
                    if not build_id or number is None or state not in SUCCESS_STATES:
                        continue
                    if entry_branch in excluded or build_id == current_build_id:
                        continue
                    found.append((build_id, str(number)))
                    if len(found) >= limit:
                        break
                url = next_url
        except (OSError, UnicodeError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"merge-shard-timings: API list failed ({exc})", file=sys.stderr)
            if found:
                return found
        else:
            return found

    base = _pipeline_url()
    if not base:
        return []
    listing_url = f"{base}/builds"
    if branch:
        listing_url = f"{listing_url}?{urllib.parse.urlencode({'branch': branch})}"
    try:
        listing = _body(fetch(listing_url))
    except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
        print(f"merge-shard-timings: public list failed ({exc})", file=sys.stderr)
        return []
    slug = _env("BUILDKITE_PIPELINE_SLUG")
    org = _env("BUILDKITE_ORGANIZATION_SLUG")
    numbers = list(dict.fromkeys(re.findall(rf'href="/{re.escape(org)}/{re.escape(slug)}/builds/([0-9]+)"', listing)))
    found = []
    for number in numbers:
        try:
            details = json.loads(_body(fetch(f"{base}/builds/{number}.json")))
        except (OSError, UnicodeError, ValueError, urllib.error.URLError, json.JSONDecodeError):
            continue
        if not isinstance(details, dict):
            continue
        entry_branch = str(details.get("branch_name") or details.get("branch") or "")
        if branch and entry_branch != branch:
            continue
        if entry_branch in excluded:
            continue
        state = str(details.get("state") or details.get("build_state") or "")
        if state not in SUCCESS_STATES:
            continue
        build_id = str(details.get("id") or "")
        if not build_id or build_id == current_build_id:
            continue
        found.append((build_id, str(number)))
        if len(found) >= limit:
            break
    return found


def _positive_int(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _job_parallelism(job: dict) -> int:
    total = _positive_int(job.get("parallel_group_total"))
    if total is not None:
        return total
    match = _SHARD_TOTAL.search(str(job.get("name") or job.get("label") or ""))
    parsed = _positive_int(match.group(1)) if match else None
    return parsed or 1


def _job_queue(job: dict) -> str:
    rules = job.get("agent_query_rules") or []
    if isinstance(rules, str):
        rules = [rules]
    if not isinstance(rules, list):
        return ""
    for rule in rules:
        text = str(rule)
        if text.startswith("queue="):
            return text.split("=", 1)[1].strip()
    return ""


def timeout_floors_from_builds(
    builds: list[dict],
    *,
    exclude_branches: frozenset[str] | set[str] = frozenset(),
    current_build_id: str = "",
    max_shards: int = MAX_AUTOMATIC_SHARDS,
) -> tuple[dict[str, int], dict[str, str]]:
    """Map a step label to the next job count after its newest timeout.

    Builds are ordered by number, newest first. The newest terminal attempt
    decides the label. ``timed_out`` adds one job, capped by ``max_shards``.
    ``passed`` / ``finished`` clears an older timeout. ``failed`` does not add
    a job. Retried, canceled, and still-running jobs are not a decision.
    """
    excluded = frozenset(exclude_branches)
    decided: set[str] = set()
    floors: dict[str, int] = {}
    queues: dict[str, str] = {}
    ordered = sorted(
        (build for build in builds if isinstance(build, dict)),
        key=lambda build: _positive_int(build.get("number")) or 0,
        reverse=True,
    )
    for build in ordered:
        build_id = str(build.get("id") or "")
        if build_id and build_id == current_build_id:
            continue
        entry_branch = str(build.get("branch") or build.get("branch_name") or "")
        if entry_branch in excluded:
            continue
        jobs = build.get("jobs")
        if not isinstance(jobs, list):
            continue
        grouped: dict[str, list[dict]] = defaultdict(list)
        for job in jobs:
            if not isinstance(job, dict):
                continue
            if job.get("retried") is True:
                continue
            if str(job.get("type") or "script") not in {"script", "command"}:
                continue
            label = _base_label(str(job.get("name") or job.get("label") or ""))
            if not label or label in decided:
                continue
            grouped[label].append(job)
        for label, label_jobs in grouped.items():
            states = {str(job.get("state") or "") for job in label_jobs}
            if not states or states & _NONTERMINAL_JOB_STATES or "" in states:
                continue
            if "timed_out" in states:
                previous = max(_job_parallelism(job) for job in label_jobs)
                floors[label] = min(max_shards, previous + 1)
                queue = _dominant_queue([_job_queue(job) for job in label_jobs])
                if queue:
                    queues[label] = queue
                decided.add(label)
                continue
            if states & _SUCCESS_JOB_STATES or "failed" in states:
                decided.add(label)
    return floors, queues


def list_recent_builds(
    *,
    branch: str,
    limit_pages: int,
    state: str = "",
    current_build_id: str = "",
    exclude_branches: frozenset[str] | set[str] = frozenset(),
    fetch: Callable[..., object] = _fetch_text,
) -> list[dict]:
    """Return newest builds, excluding ``exclude_branches`` and the current build."""
    excluded = frozenset(exclude_branches)
    token = _env("BUILDKITE_API_TOKEN") or _env("BUILDKITE_TOKEN")
    api = _api_builds_url(branch, state=state)
    if not api or not token:
        return []
    reader = _fetch_page if fetch is _fetch_text else fetch
    found: list[dict] = []
    url: str | None = api
    pages = 0
    try:
        while url and pages < limit_pages:
            pages += 1
            body, next_url = _read_listing(reader, url, token)
            payload = json.loads(body)
            if not isinstance(payload, list):
                break
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                build_id = str(entry.get("id") or "")
                entry_branch = str(entry.get("branch") or entry.get("branch_name") or "")
                if entry_branch in excluded or (build_id and build_id == current_build_id):
                    continue
                found.append(entry)
            url = next_url
    except (OSError, UnicodeError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"merge-shard-timings: build list failed ({exc})", file=sys.stderr)
    return found


def _run_agent(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["buildkite-agent", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def download_build_timings(build_id: str, dest: Path, *, timeout: int = 180) -> None:
    """Download ``timings/jobs/*.json``. Callers read that directory themselves."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        completed = _run_agent(
            ["artifact", "download", JOB_GLOB, str(dest), "--build", build_id],
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"merge-shard-timings: download {JOB_GLOB} from {build_id} failed ({exc})", file=sys.stderr)
        return
    if completed.returncode != 0:
        return


def _job_files(root: Path) -> list[Path]:
    jobs_dir = root / "timings" / "jobs"
    return sorted(jobs_dir.glob("*.json")) if jobs_dir.is_dir() else []


def observations_from_download(root: Path) -> dict[str, dict[str, float]] | None:
    """Read one build's raw job files. A previous ``merged.json`` is not an observation."""
    job_files = _job_files(root)
    if not job_files:
        return None
    observed = build_step_observations(job_files)
    return observed or None


def details_from_download(root: Path) -> dict[str, tuple[dict[str, float], str]] | None:
    job_files = _job_files(root)
    if not job_files:
        return None
    details = build_step_details(job_files)
    return details or None


def prepare_profile_from_history(
    output: Path,
    *,
    branch: str = DEFAULT_BRANCH,
    builds: int = DEFAULT_BUILDS,
    include_local: bool = True,
    exclude_branches: frozenset[str] | set[str] = frozenset(),
) -> Path | None:
    """Build a profile from passed builds.

    Ready planning passes ``branch=""`` and ``exclude_branches={"main"}`` so
    profiles come from other pull-request builds, including whatever those
    changes did to the step. Merge planning stays on ``main``. A newer
    ``timed_out`` job for a step raises the next plan by one, even when no
    passed build has timings yet.
    """
    per_build: list[dict[str, tuple[dict[str, float], str]]] = []
    build_ids: list[str] = []
    if include_local:
        local_jobs = collect_local_job_files(Path.cwd())
        if local_jobs:
            per_build.append(build_step_details(local_jobs))
            build_ids.append(_env("BUILDKITE_BUILD_ID") or "local")

    current = _env("BUILDKITE_BUILD_ID")
    candidates = max(builds * HISTORY_CANDIDATE_FACTOR, builds)
    deadline = time.monotonic() + HISTORY_LOOKUP_BUDGET_SECONDS
    misses = 0
    for build_id, number in list_recent_build_ids(
        branch=branch,
        limit=candidates,
        current_build_id=current,
        exclude_branches=exclude_branches,
    ):
        if time.monotonic() >= deadline:
            print(
                f"merge-shard-timings: stopped history scan at the {HISTORY_LOOKUP_BUDGET_SECONDS}s lookup budget",
                file=sys.stderr,
            )
            break
        with tempfile.TemporaryDirectory(prefix="vllm-omni-timings-") as tmp:
            root = Path(tmp)
            download_build_timings(build_id, root, timeout=HISTORY_DOWNLOAD_TIMEOUT_SECONDS)
            observed = details_from_download(root)
            if not observed:
                misses += 1
                print(f"merge-shard-timings: build #{number} has no timing artifacts", file=sys.stderr)
                if misses >= MAX_CONSECUTIVE_MISSES:
                    print(
                        f"merge-shard-timings: stopped history scan after {misses} consecutive builds without timings",
                        file=sys.stderr,
                    )
                    break
                continue
            misses = 0
            per_build.append(observed)
            build_ids.append(build_id)
            if len(build_ids) >= builds:
                break

    floors: dict[str, int] = {}
    floor_queues: dict[str, str] = {}
    try:
        raw_builds = list_recent_builds(
            branch=branch,
            limit_pages=TIMEOUT_SCAN_PAGES,
            state="",
            current_build_id=current,
            exclude_branches=exclude_branches,
        )
        floors, floor_queues = timeout_floors_from_builds(
            raw_builds,
            exclude_branches=exclude_branches,
            current_build_id=current,
        )
    except Exception as exc:  # noqa: BLE001 - a timeout scan must not drop a usable profile
        print(f"merge-shard-timings: timeout history skipped ({exc})", file=sys.stderr)
    for label, count in sorted(floors.items()):
        print(f"merge-shard-timings: {label!r} timed out; next plan is at least {count} jobs", file=sys.stderr)

    merged: dict[str, dict[str, float]] = {}
    queues: dict[str, str] = {}
    if per_build:
        merged, queues = merge_build_details(per_build)
    if not merged and not floors:
        return None
    return write_merged_profile(
        merged,
        output,
        builds=len(per_build),
        build_ids=build_ids,
        step_queues=queues,
        timeout_floors=floors,
        timeout_floor_queues=floor_queues,
    )
