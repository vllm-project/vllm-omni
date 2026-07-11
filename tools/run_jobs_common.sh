#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared helpers for tools/run_*_jobs.sh (source this file; do not execute).
#
# Contents:
#   1) Timing / tee runners used by ready, merge, and nightly.
#   2) Ready/merge CLI + YAML extract entrypoint: run_yaml_ci_jobs_main
#      (set BUILDKITE_REL and DEFAULT_LOG_SUBDIR in the entry script, then call it).
#
# Job timeouts: generated job scripts embed GNU timeout on the pytest line itself
# (see prepend_timeout_to_pytest in the Python heredoc below). This runner only
# executes bash job.sh and records exit 124 as TIMED OUT via jobs/.job_timeouts
# metadata.

if [[ -n "${_RUN_JOBS_COMMON_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
_RUN_JOBS_COMMON_LOADED=1

if [[ -z "${SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ---------------------------------------------------------------------------
# Timing / tee helpers (ready, merge, nightly)
# ---------------------------------------------------------------------------

RUN_JOB_NAMES=()
RUN_JOB_SECONDS=()
RUN_JOB_STATUSES=()

_run_jobs_reset_timing() {
  RUN_JOB_NAMES=()
  RUN_JOB_SECONDS=()
  RUN_JOB_STATUSES=()
}

_run_jobs_epoch_seconds() {
  date +%s
}

_run_jobs_format_duration() {
  local total="${1}"
  local h m s
  if [[ "${total}" -lt 60 ]]; then
    printf '%ss' "${total}"
    return 0
  fi
  m=$((total / 60))
  s=$((total % 60))
  if [[ "${m}" -lt 60 ]]; then
    if [[ "${s}" -eq 0 ]]; then
      printf '%dm' "${m}"
    else
      printf '%dm %ss' "${m}" "${s}"
    fi
    return 0
  fi
  h=$((m / 60))
  m=$((m % 60))
  if [[ "${m}" -eq 0 && "${s}" -eq 0 ]]; then
    printf '%dh' "${h}"
  elif [[ "${s}" -eq 0 ]]; then
    printf '%dh %dm' "${h}" "${m}"
  else
    printf '%dh %dm %ss' "${h}" "${m}" "${s}"
  fi
}

_run_jobs_record_timing() {
  local name="${1}"
  local seconds="${2}"
  local status="${3}"
  RUN_JOB_NAMES+=("${name}")
  RUN_JOB_SECONDS+=("${seconds}")
  RUN_JOB_STATUSES+=("${status}")
}

_run_jobs_lookup_timeout_minutes() {
  local base="${1}"
  local manifest="${LOG_DIR}/jobs/.job_timeouts"
  local line key mins
  [[ -f "${manifest}" ]] || return 1
  while IFS='=' read -r key mins; do
    [[ -n "${key}" ]] || continue
    if [[ "${key}" == "${base}" ]]; then
      printf '%s' "${mins}"
      return 0
    fi
  done < "${manifest}"
  return 1
}

_run_one_job_with_timing() {
  local _job="$1"
  local base out job_status start end elapsed timeout_min=""
  base="$(basename "${_job}" .sh)"
  out="${LOG_DIR}/${base}.log"
  if timeout_min="$(_run_jobs_lookup_timeout_minutes "${base}")"; then
    echo "==> ${_job}  (tee ${out}, pytest inline timeout ${timeout_min}m)" >&2
  else
    echo "==> ${_job}  (tee ${out})" >&2
  fi
  start="$(_run_jobs_epoch_seconds)"
  (cd "${REPO_ROOT}" && bash "${_job}") 2>&1 | tee "${out}"
  job_status="${PIPESTATUS[0]}"
  end="$(_run_jobs_epoch_seconds)"
  elapsed=$((end - start))
  _run_jobs_record_timing "${base}" "${elapsed}" "${job_status}"
  if [[ "${job_status}" -eq 0 ]]; then
    echo "    finished in $(_run_jobs_format_duration "${elapsed}")" >&2
  elif [[ "${job_status}" -eq 124 && -n "${timeout_min}" ]]; then
    echo "    timed out (inline timeout ${timeout_min}m on pytest, exit 124)" >&2
  elif [[ "${job_status}" -eq 124 ]]; then
    echo "    timed out (exit 124)" >&2
  else
    echo "    failed after $(_run_jobs_format_duration "${elapsed}") (exit ${job_status})" >&2
  fi
  return "${job_status}"
}

_run_jobs_print_timing_summary() {
  local wall_start="${1:-}"
  local any_fail="${2:-0}"
  local summary_path="${LOG_DIR}/timing_summary.log"
  local -a lines=()
  local i name secs status status_str failed_count=0
  local wall_elapsed job_count total_elapsed=0

  lines+=("=== Job timing summary ===")
  for i in "${!RUN_JOB_NAMES[@]}"; do
    name="${RUN_JOB_NAMES[$i]}"
    secs="${RUN_JOB_SECONDS[$i]}"
    status="${RUN_JOB_STATUSES[$i]}"
    if [[ "${status}" -eq 0 ]]; then
      status_str="OK"
    elif [[ "${status}" -eq 124 ]]; then
      status_str="TIMED OUT"
      failed_count=$((failed_count + 1))
    else
      status_str="FAILED (exit ${status})"
      failed_count=$((failed_count + 1))
    fi
    lines+=("  ${name}  $(_run_jobs_format_duration "${secs}")  ${status_str}")
  done

  job_count="${#RUN_JOB_NAMES[@]}"
  if [[ -n "${wall_start}" ]]; then
    wall_elapsed=$(( $(_run_jobs_epoch_seconds) - wall_start ))
    lines+=("Total wall time: $(_run_jobs_format_duration "${wall_elapsed}") (${job_count} jobs)")
  elif [[ "${job_count}" -gt 0 ]]; then
    for secs in "${RUN_JOB_SECONDS[@]}"; do
      total_elapsed=$((total_elapsed + secs))
    done
    lines+=("Total job time: $(_run_jobs_format_duration "${total_elapsed}") (${job_count} jobs)")
  else
    lines+=("Total wall time: 0s (0 jobs)")
  fi

  if [[ "${failed_count}" -gt 0 ]]; then
    lines+=("Failed jobs: ${failed_count}/${job_count}")
  fi

  if [[ "${any_fail}" -ne 0 ]]; then
    lines+=("Result: one or more jobs failed. See logs under ${LOG_DIR}.")
  else
    lines+=("Result: all jobs finished OK. Logs: ${LOG_DIR}/*.log")
  fi

  {
    for line in "${lines[@]}"; do
      printf '%s\n' "${line}"
    done
  } | tee "${summary_path}" >&2
}

# ---------------------------------------------------------------------------
# Ready / merge: CLI + YAML extract + run (tools/run_ready_jobs.sh,
# tools/run_merge_jobs.sh). Nightly does not use this entrypoint.
#
# Entry scripts must set before calling run_yaml_ci_jobs_main:
#   BUILDKITE_REL      - e.g. .buildkite/test-ready.yml
#   DEFAULT_LOG_SUBDIR - e.g. ready_jobs
# ---------------------------------------------------------------------------

_run_yaml_ci_jobs_usage() {
  # Print the entry script header (caller is run_ready_jobs.sh / run_merge_jobs.sh).
  sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'
}

_run_jobs_split_append_csv_array() {
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

_run_jobs_finalize_model_type_csv() {
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

_run_jobs_find_repo_containing_yml() {
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

_run_jobs_derive_repo_root_from_yml() {
  local yml="$1"
  local d
  d="$(cd "$(dirname "${yml}")" && pwd)"
  [[ "$(basename "${d}")" == ".buildkite" ]] || return 1
  printf '%s\n' "$(dirname "${d}")"
}

run_yaml_ci_jobs_main() {
  if [[ -z "${BUILDKITE_REL:-}" || -z "${DEFAULT_LOG_SUBDIR:-}" ]]; then
    echo "run_yaml_ci_jobs_main: set BUILDKITE_REL and DEFAULT_LOG_SUBDIR before calling." >&2
    exit 1
  fi

  LABEL_SUBSTR="${LABEL_SUBSTR:-}"
  MODEL_TYPE_ENV="${MODEL_TYPE:-all}"
  MODEL_TYPE_CLI_PARTS=()
  MODEL_TYPE_FROM_CLI=0
  DRY_RUN="${DRY_RUN:-0}"
  SKIP_SIMPLE=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        _run_yaml_ci_jobs_usage
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
        _run_jobs_split_append_csv_array MODEL_TYPE_CLI_PARTS "$2"
        shift 2
        ;;
      --skip-simple)
        SKIP_SIMPLE=1
        shift
        ;;
      *)
        echo "Unknown option: $1" >&2
        _run_yaml_ci_jobs_usage >&2
        exit 2
        ;;
    esac
  done

  MODEL_TYPE="$(_run_jobs_finalize_model_type_csv)"

  if [[ -n "${YML:-}" && -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
    YML="$(cd "$(dirname "${YML}")" && pwd)/$(basename "${YML}")"
  elif [[ -n "${YML:-}" ]]; then
    YML="$(cd "$(dirname "${YML}")" && pwd)/$(basename "${YML}")"
    if ! REPO_ROOT="$(_run_jobs_derive_repo_root_from_yml "${YML}")"; then
      echo "Could not derive REPO_ROOT from YML=${YML} (expected file at <repo>/${BUILDKITE_REL})." >&2
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
      REPO_ROOT="$(_run_jobs_find_repo_containing_yml "${PWD}" || true)"
    fi
    if [[ -z "${REPO_ROOT}" ]]; then
      REPO_ROOT="$(_run_jobs_find_repo_containing_yml "${SCRIPT_DIR}" || true)"
    fi
    if [[ -z "${REPO_ROOT}" ]]; then
      echo "Could not locate ${BUILDKITE_REL}. Set REPO_ROOT or YML, run from inside the vllm-omni clone," >&2
      echo "or place this script (or run from a cwd) under the repository tree." >&2
      exit 2
    fi
    YML="${REPO_ROOT}/${BUILDKITE_REL}"
  fi

  LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${DEFAULT_LOG_SUBDIR}}"

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


_INLINE_TIMEOUT_PREFIX = re.compile(r"^timeout\s+(?:-\S+\s+)*\S+\s+", re.IGNORECASE)


def timeout_kill_after_seconds() -> int:
    raw = os.environ.get("RUN_JOB_TIMEOUT_KILL_AFTER", "60")
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        return 60
    return seconds if seconds > 0 else 60


def prepend_timeout_to_pytest(pytest_line: str, timeout_min: int | None) -> str:
    """Prepend GNU timeout directly before pytest (Buildkite inline style)."""
    if timeout_min is None:
        return pytest_line.strip()
    line = _INLINE_TIMEOUT_PREFIX.sub("", pytest_line.strip())
    kill_after = timeout_kill_after_seconds()
    return (
        f"timeout --foreground --verbose --kill-after={kill_after} "
        f"{timeout_min}m {line}"
    )


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
        script_lines.extend(prepend_timeout_to_pytest(p, timeout_min) for p in pys)
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

  set -o pipefail
  local any_fail=0
  local _run_start
  local _job
  _run_jobs_reset_timing
  _run_start="$(_run_jobs_epoch_seconds)"

  shopt -s nullglob
  for _job in "${LOG_DIR}/jobs"/*.sh; do
    _run_one_job_with_timing "${_job}" || any_fail=1
  done
  shopt -u nullglob

  _run_jobs_print_timing_summary "${_run_start}" "${any_fail}"
  if [[ "${any_fail}" -ne 0 ]]; then
    exit 1
  fi
  exit 0
}
