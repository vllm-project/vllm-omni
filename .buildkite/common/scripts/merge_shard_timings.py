#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Merge per-job timing artifacts into one per-step p80 profile.

Job files look like ``timings/jobs/<job>-<pid>.json`` with ``file_seconds``,
``job_id``, and ``label``. Only builds whose state is ``passed`` or
``finished`` contribute. Each selected build uses its raw job files, never a
previously merged profile. Invocations that share a job id are summed, then
different jobs contribute the max for that file. Across builds, the profile
stores the p80 of those observations under the step label (a ``Shard N/M``
suffix is removed).

``--prepare-for-upload`` downloads recent ``timings/jobs/*.json`` via
``buildkite-agent``. Missing history is not an error: planning stays on one job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

DEFAULT_BUILDS = 20
DEFAULT_BRANCH = "main"
JOB_GLOB = "timings/jobs/*.json"
MERGED_NAME = "timings/merged.json"
SCHEMA = "vllm_omni.shard_timings.merged.v1"
SUCCESS_STATES = frozenset({"passed", "finished"})
_SHARD_SUFFIX = re.compile(r" · Shard(?: %N/%t| \d+/\d+)$")


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


def load_job_record(path: Path) -> tuple[str, str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected object")
    raw = payload.get("file_seconds")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: missing file_seconds")
    seconds = {str(key).replace("\\", "/"): float(value) for key, value in raw.items() if float(value) > 0}
    job_id = str(payload.get("job_id") or path.stem)
    label = _base_label(str(payload.get("label") or ""))
    return job_id, label, seconds


def build_step_observations(job_files: list[Path]) -> dict[str, dict[str, float]]:
    """One build's per-step file seconds from raw job files.

    Invocations that share a job id are summed, so two pytest processes in one
    job both count, including when they touch the same file. Different jobs
    then contribute the max, so a duplicated full run is not added twice.
    """
    per_job: dict[tuple[str, str], dict[str, float]] = {}
    for path in job_files:
        job_id, label, seconds = load_job_record(path)
        bucket = per_job.setdefault((job_id, label), {})
        for file, value in seconds.items():
            bucket[file] = bucket.get(file, 0.0) + value
    observed: dict[str, dict[str, float]] = {}
    for (_, label), files in per_job.items():
        step = observed.setdefault(label, {})
        for file, value in files.items():
            step[file] = max(step.get(file, 0.0), value)
    return observed


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


def write_merged_profile(
    steps: dict[str, dict[str, float]],
    output: Path,
    *,
    builds: int,
    build_ids: list[str] | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "measurement": "p80_over_passed_builds_of_raw_job_file_seconds",
        "builds": builds,
        "build_ids": build_ids or [],
        "steps": steps,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def collect_local_job_files(root: Path) -> list[Path]:
    return sorted(root.glob(JOB_GLOB))


def merge_local_jobs(root: Path, output: Path) -> Path:
    job_files = collect_local_job_files(root)
    if not job_files:
        raise FileNotFoundError(f"no job timing files under {root / 'timings' / 'jobs'}")
    observed = build_step_observations(job_files)
    return write_merged_profile(observed, output, builds=1, build_ids=[])


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _pipeline_url() -> str | None:
    org = _env("BUILDKITE_ORGANIZATION_SLUG")
    slug = _env("BUILDKITE_PIPELINE_SLUG")
    if not org or not slug:
        return None
    return f"https://buildkite.com/{org}/{slug}"


def _api_builds_url(branch: str, per_page: int) -> str | None:
    org = _env("BUILDKITE_ORGANIZATION_SLUG")
    slug = _env("BUILDKITE_PIPELINE_SLUG")
    if not org or not slug:
        return None
    query = urllib.parse.urlencode(
        {
            "branch": branch,
            "state": "passed",
            "per_page": str(per_page),
        }
    )
    return f"https://api.buildkite.com/v2/organizations/{org}/pipelines/{slug}/builds?{query}"


def _fetch_text(url: str, *, token: str | None = None) -> str:
    headers = {"User-Agent": "vllm-omni-merge-shard-timings"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def list_recent_build_ids(
    *,
    branch: str,
    limit: int,
    current_build_id: str = "",
    fetch: Callable[..., str] = _fetch_text,
) -> list[tuple[str, str]]:
    """Return ``(build_id, build_number)`` newest first, excluding the current build."""
    token = _env("BUILDKITE_API_TOKEN") or _env("BUILDKITE_TOKEN")
    api = _api_builds_url(branch, max(limit * 2, limit))
    if api and token:
        try:
            payload = json.loads(fetch(api, token=token))
        except (OSError, UnicodeError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"merge-shard-timings: API list failed ({exc})", file=sys.stderr)
            payload = None
        if isinstance(payload, list):
            found: list[tuple[str, str]] = []
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                build_id = str(entry.get("id") or "")
                number = entry.get("number")
                state = str(entry.get("state") or "")
                if not build_id or number is None or state not in SUCCESS_STATES:
                    continue
                if build_id == current_build_id:
                    continue
                found.append((build_id, str(number)))
                if len(found) >= limit:
                    break
            return found

    base = _pipeline_url()
    if not base:
        return []
    query = urllib.parse.urlencode({"branch": branch})
    try:
        listing = fetch(f"{base}/builds?{query}")
    except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
        print(f"merge-shard-timings: public list failed ({exc})", file=sys.stderr)
        return []
    slug = _env("BUILDKITE_PIPELINE_SLUG")
    org = _env("BUILDKITE_ORGANIZATION_SLUG")
    numbers = list(dict.fromkeys(re.findall(rf'href="/{re.escape(org)}/{re.escape(slug)}/builds/([0-9]+)"', listing)))
    found = []
    for number in numbers:
        try:
            details = json.loads(fetch(f"{base}/builds/{number}.json"))
        except (OSError, UnicodeError, ValueError, urllib.error.URLError, json.JSONDecodeError):
            continue
        if not isinstance(details, dict):
            continue
        if details.get("branch_name") != branch:
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


def _run_agent(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["buildkite-agent", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def download_build_timings(build_id: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for pattern in (JOB_GLOB,):
        try:
            completed = _run_agent(
                ["artifact", "download", pattern, str(dest), "--build", build_id],
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"merge-shard-timings: download {pattern} from {build_id} failed ({exc})", file=sys.stderr)
            continue
        if completed.returncode != 0:
            continue
    merged = dest / MERGED_NAME
    if merged.is_file():
        downloaded.append(merged)
    downloaded.extend(
        sorted((dest / "timings" / "jobs").glob("*.json")) if (dest / "timings" / "jobs").is_dir() else []
    )
    return downloaded


def observations_from_download(root: Path) -> dict[str, dict[str, float]] | None:
    """Read one build's raw job files. A previous ``merged.json`` is not an observation."""
    jobs_dir = root / "timings" / "jobs"
    job_files = sorted(jobs_dir.glob("*.json")) if jobs_dir.is_dir() else []
    if not job_files:
        return None
    observed = build_step_observations(job_files)
    return observed or None


def prepare_profile_from_history(
    output: Path,
    *,
    branch: str = DEFAULT_BRANCH,
    builds: int = DEFAULT_BUILDS,
    include_local: bool = True,
) -> Path | None:
    per_build: list[dict[str, dict[str, float]]] = []
    build_ids: list[str] = []
    if include_local:
        local_jobs = collect_local_job_files(Path.cwd())
        if local_jobs:
            per_build.append(build_step_observations(local_jobs))
            build_ids.append(_env("BUILDKITE_BUILD_ID") or "local")

    current = _env("BUILDKITE_BUILD_ID")
    for build_id, number in list_recent_build_ids(branch=branch, limit=builds, current_build_id=current):
        with tempfile.TemporaryDirectory(prefix="vllm-omni-timings-") as tmp:
            root = Path(tmp)
            download_build_timings(build_id, root)
            observed = observations_from_download(root)
            if not observed:
                print(f"merge-shard-timings: build #{number} has no timing artifacts", file=sys.stderr)
                continue
            per_build.append(observed)
            build_ids.append(build_id)
            if len(per_build) >= builds:
                break

    if not per_build:
        return None
    merged = merge_step_observations(per_build)
    if not merged:
        return None
    return write_merged_profile(merged, output, builds=len(per_build), build_ids=build_ids)


def upload_artifact(path: Path) -> None:
    completed = _run_agent(["artifact", "upload", str(path)], timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "artifact upload failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(MERGED_NAME))
    parser.add_argument("--builds", type=int, default=DEFAULT_BUILDS)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--prepare-for-upload",
        action="store_true",
        help="Download recent timing artifacts and write a profile for the uploader",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Merge only timings/jobs under the current directory",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the merged profile with buildkite-agent artifact upload",
    )
    args = parser.parse_args(argv)

    try:
        if args.local_only:
            path = merge_local_jobs(Path.cwd(), args.output)
        elif args.prepare_for_upload:
            path = prepare_profile_from_history(
                args.output,
                branch=args.branch,
                builds=args.builds,
                include_local=False,
            )
            if path is None:
                print("merge-shard-timings: no historical timings; uploader keeps one job", file=sys.stderr)
                return 0
        else:
            path = prepare_profile_from_history(
                args.output,
                branch=args.branch,
                builds=args.builds,
                include_local=True,
            )
            if path is None:
                # Still try a pure local merge so a post-test step uploads something.
                try:
                    path = merge_local_jobs(Path.cwd(), args.output)
                except FileNotFoundError as exc:
                    print(f"merge-shard-timings: {exc}", file=sys.stderr)
                    return 0
        print(f"merge-shard-timings: wrote {path}")
        if args.upload:
            upload_artifact(path)
            print(f"merge-shard-timings: uploaded {path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CI soft path: log and exit non-zero for soft_fail
        print(f"merge-shard-timings: failed ({exc})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
