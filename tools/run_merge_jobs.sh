#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Extract steps from .buildkite/test-merge.yml that contain pytest, synthesize
# small bash wrappers (exports + pytest), run them, and tee output to logs named
# after each step's Buildkite "key" when present (otherwise a slug of the label).
#
# Model area (--model-type / MODEL_TYPE), multiple allowed (OR semantics):
#   e.g. --model-type omni,tts
#   omni     — label contains "Omni ·"
#   tts      — label contains "TTS ·"
#   diffusion— label contains "Diffusion"
#   all      — no model filter (default)
#
#   --skip-simple — skip steps in the "Simple Test" Buildkite group and labels
#                   starting with "Simple ·" (L1-style unit tests in the same YAML)
#
# Requirements: bash, python3, PyYAML (pip install pyyaml)
#
# Usage:
#   bash tools/merge/run_merge_jobs.sh
#   REPO_ROOT=/path/to/vllm-omni bash tools/merge/run_merge_jobs.sh --model-type diffusion --dry-run
#   YML=/path/to/vllm-omni/.buildkite/test-merge.yml bash tools/merge/run_merge_jobs.sh
#
# Repository / YAML (no dependency on where this script lives):
#   • Set REPO_ROOT (or pass --repo-root) — default YAML is $REPO_ROOT/.buildkite/test-merge.yml
#   • Or set YML (or --yaml) — repo root is inferred as parent of the .buildkite directory
#   • Or run from inside the clone: git rev-parse --show-toplevel, else walk up from $PWD,
#     then from the script's directory, until .buildkite/test-merge.yml exists
#
# Optional environment:
#   REPO_ROOT     - vllm-omni root (working directory for pytest); see above
#   YML           - path to test-merge.yml (default: $REPO_ROOT/.buildkite/test-merge.yml)
#   LOG_DIR       - logs + generated job scripts (default: $REPO_ROOT/logs/merge_jobs);
#                   per-job *.log plus timing_summary.log after the run
#   MODEL_TYPE    - comma-separated and/or repeated flags (default: all); see above
#   LABEL_SUBSTR  - substring of Buildkite step label
#   DRY_RUN=1     - print extracted commands only; do not write scripts or run pytest
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LABEL_SUBSTR="${LABEL_SUBSTR:-}"
MODEL_TYPE_ENV="${MODEL_TYPE:-all}"
MODEL_TYPE_CLI_PARTS=()
MODEL_TYPE_FROM_CLI=0
DRY_RUN="${DRY_RUN:-0}"
SKIP_SIMPLE=0

usage() {
  sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'
}

_split_append_csv_array() {
  local -n _arr="${1}"
  local _raw="${2}"
  local _ifs="${IFS}"
  local _part
  IFS=','
  for _part in ${_raw}; do
    IFS="${_ifs}"
    _part="$(printf '%s' "${_part}" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${_part}" ]] || continue
    _arr+=("${_part}")
  done
}

_finalize_model_type_csv() {
  if [[ "${MODEL_TYPE_FROM_CLI}" -eq 1 ]]; then
    if ((${#MODEL_TYPE_CLI_PARTS[@]} == 0)); then
      echo "--model-type requires a non-empty value" >&2
      exit 2
    fi
    local -A _seen=()
    local -a _out=()
    local _x
    for _x in "${MODEL_TYPE_CLI_PARTS[@]}"; do
      if [[ "${_x}" == all ]]; then
        printf '%s' "all"
        return 0
      fi
      [[ ${_seen["${_x}"]+isset} ]] && continue
      _seen["${_x}"]=1
      _out+=("${_x}")
    done
    (IFS=','; printf '%s' "${_out[*]}")
  else
    printf '%s' "${MODEL_TYPE_ENV}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --yaml)
      YML="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --label-substr)
      LABEL_SUBSTR="$2"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE_FROM_CLI=1
      _split_append_csv_array MODEL_TYPE_CLI_PARTS "$2"
      shift 2
      ;;
    --skip-simple)
      SKIP_SIMPLE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MODEL_TYPE="$(_finalize_model_type_csv)"

BUILDKITE_REL=".buildkite/test-merge.yml"

_find_repo_containing_yml() {
  local dir="${1:-}"
  [[ -n "$dir" ]] || return 1
  dir="$(cd "$dir" && pwd)" || return 1
  while true; do
    if [[ -f "${dir}/${BUILDKITE_REL}" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    [[ "${dir}" == "/" ]] && return 1
    dir="$(dirname "${dir}")"
  done
}

_derive_repo_root_from_yml() {
  local yml="$1"
  local d
  d="$(cd "$(dirname "${yml}")" && pwd)"
  [[ "$(basename "${d}")" == ".buildkite" ]] || return 1
  printf '%s\n' "$(dirname "${d}")"
}

if [[ -n "${YML:-}" && -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
  YML="$(cd "$(dirname "${YML}")" && pwd)/$(basename "${YML}")"
elif [[ -n "${YML:-}" ]]; then
  YML="$(cd "$(dirname "${YML}")" && pwd)/$(basename "${YML}")"
  if ! REPO_ROOT="$(_derive_repo_root_from_yml "${YML}")"; then
    echo "Could not derive REPO_ROOT from YML=${YML} (expected file at <repo>/.buildkite/test-merge.yml)." >&2
    echo "Set REPO_ROOT explicitly (or pass --repo-root) for pytest working directory." >&2
    exit 2
  fi
elif [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
  YML="${REPO_ROOT}/${BUILDKITE_REL}"
else
  REPO_ROOT=""
  if command -v git >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "${PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
  fi
  if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(_find_repo_containing_yml "${PWD}" || true)"
  fi
  if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(_find_repo_containing_yml "${SCRIPT_DIR}" || true)"
  fi
  if [[ -z "${REPO_ROOT}" ]]; then
    echo "Could not locate ${BUILDKITE_REL}. Set REPO_ROOT or YML, run from inside the vllm-omni clone," >&2
    echo "or place this script (or run from a cwd) under the repository tree." >&2
    exit 2
  fi
  YML="${REPO_ROOT}/${BUILDKITE_REL}"
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/merge_jobs}"

if [[ ! -f "${YML}" ]]; then
  echo "YAML not found: ${YML}" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}/jobs"
LOG_DIR="$(cd "${LOG_DIR}" && pwd)"
export REPO_ROOT LOG_DIR YML LABEL_SUBSTR MODEL_TYPE DRY_RUN SKIP_SIMPLE

if [[ "${DRY_RUN}" != "1" ]]; then
  shopt -s nullglob
  _stale=( "${LOG_DIR}/jobs"/*.sh )
  _stale_logs=( "${LOG_DIR}"/*.log )
  shopt -u nullglob
  if ((${#_stale[@]})); then
    rm -f "${_stale[@]}"
  fi
  if ((${#_stale_logs[@]})); then
    rm -f "${_stale_logs[@]}"
  fi
  rm -f "${LOG_DIR}/jobs/.job_timeouts"
fi

# shellcheck disable=SC2016,SC1078,SC1079
python3 - <<'PY'
from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(os.environ["REPO_ROOT"]).resolve()
LOG_DIR = Path(os.environ["LOG_DIR"]).resolve()
YML = Path(os.environ["YML"]).resolve()
LABEL_SUBSTR = (os.environ.get("LABEL_SUBSTR") or "").strip()
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
SKIP_SIMPLE = os.environ.get("SKIP_SIMPLE", "0") == "1"

ALLOWED_MODEL_TYPES = frozenset({"all", "omni", "tts", "diffusion"})


def parse_model_types(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        parts = ["all"]
    bad = [p for p in parts if p not in ALLOWED_MODEL_TYPES]
    if bad:
        print(
            f"Invalid MODEL_TYPE / --model-type value(s): {bad!r} "
            f"(allowed: {', '.join(sorted(ALLOWED_MODEL_TYPES))})",
            file=sys.stderr,
        )
        sys.exit(2)
    if "all" in parts:
        return ["all"]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


PYTEST_CMD_RE = re.compile(
    r"(?:timeout\s+\S+\s+)?(?:python3? -m\s+)?pytest\s+[^\n&|;]*"
)


def iter_leaf_steps(steps, group: str | None = None):
    for raw in steps or []:
        if not isinstance(raw, dict):
            continue
        nested = raw.get("steps")
        if isinstance(nested, list) and nested:
            g = raw.get("group")
            next_group: str | None
            if isinstance(g, str):
                next_group = g
            elif g is not None:
                next_group = str(g)
            else:
                next_group = group
            yield from iter_leaf_steps(nested, next_group)
            continue
        if raw.get("commands"):
            yield raw, group


def raw_command_text(step: dict) -> str:
    raw = step.get("commands") or []
    if isinstance(raw, str):
        raw = [raw]
    text = "\n".join((c.strip() if isinstance(c, str) else "") for c in raw if c)
    return text.replace("$$", "$")


def export_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            out.append(s)
    return out


def pytest_lines(text: str) -> list[str]:
    out: list[str] = []
    for m in PYTEST_CMD_RE.finditer(text):
        line = m.group(0).strip()
        line_start = text.rfind("\n", 0, m.start()) + 1
        before = text[line_start : m.start()]
        if before.lstrip().startswith("#"):
            continue
        if line:
            out.append(line)
    return out


def is_simple_test_step(label: str, group: str | None) -> bool:
    if group and "Simple Test" in group:
        return True
    return label.startswith("Simple ·")


def label_matches_model_type(label: str, model_types: list[str]) -> bool:
    if "all" in model_types:
        return True
    if "omni" in model_types and "Omni ·" in label:
        return True
    if "tts" in model_types and "TTS ·" in label:
        return True
    if "diffusion" in model_types and "Diffusion" in label:
        return True
    return False


def job_key_from_step(step: dict, label: str) -> str:
    k = step.get("key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    slug = re.sub(r"[^\w\-.]+", "_", label or "job", flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_") or "job"
    return slug


def step_timeout_minutes(step: dict) -> int | None:
    raw = step.get("timeout_in_minutes")
    if raw is None:
        return None
    try:
        minutes = int(raw)
    except (TypeError, ValueError):
        return None
    return minutes if minutes > 0 else None


def _write_job_timeouts_manifest(jobs_dir: Path, job_timeouts: dict[str, int]) -> None:
    manifest_path = jobs_dir / ".job_timeouts"
    if job_timeouts:
        manifest_path.write_text(
            "\n".join(f"{k}={v}" for k, v in sorted(job_timeouts.items())) + "\n",
            encoding="utf-8",
        )
    elif manifest_path.is_file():
        manifest_path.unlink()


def _write_job_script(key: str, script_lines: list[str], jobs_dir: Path) -> None:
    body = "\n".join(script_lines) + "\n"
    if DRY_RUN:
        print(f"=== {key} ===")
        print(body)
        return
    job_path = jobs_dir / f"{key}.sh"
    job_path.write_text(body, encoding="utf-8")
    try:
        mode = job_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        job_path.chmod(mode)
    except OSError:
        pass
    print(f"generated {job_path}", file=sys.stderr)


def main() -> None:
    model_types = parse_model_types(os.environ.get("MODEL_TYPE", "all"))

    jobs_dir = LOG_DIR / "jobs"
    if not DRY_RUN:
        jobs_dir.mkdir(parents=True, exist_ok=True)

    if not YML.is_file():
        print(f"YAML not found: {YML}", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(YML.read_text(encoding="utf-8"))
    top_steps = (data or {}).get("steps") or []

    matched_yaml = 0
    job_timeouts: dict[str, int] = {}
    for step, grp in iter_leaf_steps(top_steps):
        label = step.get("label") or ""
        if SKIP_SIMPLE and is_simple_test_step(label, grp):
            continue
        if LABEL_SUBSTR and LABEL_SUBSTR not in label:
            continue
        if not label_matches_model_type(label, model_types):
            continue
        text = raw_command_text(step)
        exports = export_lines(text)
        pys = pytest_lines(text)
        if not pys:
            print(f"# skip (no pytest line): {label!r}", file=sys.stderr)
            continue

        key = job_key_from_step(step, label)
        matched_yaml += 1

        script_lines = [
            "#!/usr/bin/env bash",
            f'# From Buildkite label: {label.replace(chr(10), " ")}',
            "set -euo pipefail",
            f'cd "{REPO_ROOT}"',
        ]
        timeout_min = step_timeout_minutes(step)
        if timeout_min is not None:
            script_lines.append(f"# Buildkite timeout_in_minutes: {timeout_min}")
            job_timeouts[key] = timeout_min
        script_lines.extend(exports)
        script_lines.extend(pys)
        _write_job_script(key, script_lines, jobs_dir)

    if not DRY_RUN:
        _write_job_timeouts_manifest(jobs_dir, job_timeouts)

    if matched_yaml == 0:
        print(
            f"No YAML steps matched MODEL_TYPE={model_types!r} "
            f"LABEL_SUBSTR={LABEL_SUBSTR!r} SKIP_SIMPLE={SKIP_SIMPLE!r} in {YML}",
            file=sys.stderr,
        )
        sys.exit(2)


main()
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

# shellcheck source=tools/run_jobs_common.sh
source "${SCRIPT_DIR}/run_jobs_common.sh"

run_generated_jobs_with_tee() {
  set -o pipefail
  local any_fail=0
  local _run_start
  _run_jobs_reset_timing
  _run_start="$(_run_jobs_epoch_seconds)"

  local _job
  shopt -s nullglob
  for _job in "${LOG_DIR}/jobs"/*.sh; do
    _run_one_job_with_timing "${_job}" || any_fail=1
  done
  shopt -u nullglob

  _run_jobs_print_timing_summary "${_run_start}" "${any_fail}"
  if [[ "${any_fail}" -ne 0 ]]; then
    return 1
  fi
  return 0
}

run_generated_jobs_with_tee || exit 1
