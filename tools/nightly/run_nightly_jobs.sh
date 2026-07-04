#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Extract steps from .buildkite/test-nightly.yml that contain pytest, synthesize
# small bash wrappers (exports + pytest), run them, and tee output to logs named
# after each step's Buildkite "key" when present (otherwise a slug of the label).
# YAML steps whose label contains "Perf Test" run first, then
# python tools/nightly/generate_nightly_perf_excel.py runs (even if some perf jobs failed;
# the script only aggregates JSON already present under tests/dfx/perf/results).
# Excel and generate_nightly_perf_excel.log are written under $REPO_ROOT/logs/;
# per-job tee logs use LOG_DIR (see --log-dir; default is timestamped under $REPO_ROOT/logs/).
#
# Test kind (--test-type / TEST_TYPE), multiple allowed (OR semantics):
#   Repeat the flag and/or use comma-separated values, e.g.
#     --test-type perf,acc   OR   --test-type perf --test-type function
#   If any selected value is all, YAML steps are unconstrained except that perf/acc/function tokens
#   are ignored; all,stability / all,local / all,stability,local still run those extra bundles.
#   perf     — label contains "Perf Test" (throughput / benchmark jobs)
#   acc      — label contains "Accuracy Test" (incl. GEBench / GEdit-Bench style)
#   function — label has neither "Perf Test" nor "Accuracy Test" (incl. Doc, Multi-Replica, etc.)
#   stability— fixed dfx stability scripts under tests/dfx/stability/scripts/ (see below)
#   local    — pytest -sv -m "<MODEL_TYPE markers> and local_model" from repo root (no YAML step extract)
#              When LABEL_SUBSTR is set, also runs matching tests/**/test_*.py and perf JSON configs under
#              tests/dfx/perf/tests/*.json via run_benchmark.py / run_diffusion_benchmark.py.
#   all      — no test-kind filter for YAML steps (any leaf step with pytest in test-nightly.yml)
#
# Model area (--model-type / MODEL_TYPE), multiple allowed (OR semantics):
#   e.g. --model-type omni,tts
#   omni     — label contains "Omni ·"
#   tts      — label contains "TTS ·"
#   diffusion— label contains "Diffusion"
#   all      — no model filter (e.g. Quantization-only steps match only when test-type allows)
#
#   Combining stability and/or local with perf/acc/function/all runs each enabled bundle; each enabled
#   mode must match at least one job or the script exits with status 2.
#
#   local (when included in TEST_TYPE):
#     From repo root: pytest -sv -m "<markers> and local_model" (markers from MODEL_TYPE: omni, tts,
#     diffusion; all → "(omni or tts or diffusion) and local_model"). Not filtered by nightly YAML.
#     LABEL_SUBSTR: if set, restrict to tests/**/test_*.py basenames and tests/dfx/perf/tests/*.json
#                   whose filename contains the substring (benchmark runner chosen by JSON family).
#
#   stability (when included in TEST_TYPE):
#     From repo root: pytest -s -v --run-level full_model tests/dfx/stability/scripts/test_stability_*.py
#     model_type: omni → qwen3_omni; tts → qwen3_tts + voxcpm2; diffusion → qwen_image + wan22 + hunyuan_image; all → all six
#     LABEL_SUBSTR: if set, script path / job key / filename must contain it
#
# Requirements: bash, python3, PyYAML (pip install pyyaml)
#
# Usage:
#   bash path/to/run_nightly_jobs.sh
#   REPO_ROOT=/path/to/vllm-omni bash path/to/run_nightly_jobs.sh --test-type perf,acc --dry-run
#   YML=/path/to/vllm-omni/.buildkite/test-nightly.yml bash path/to/run_nightly_jobs.sh
#
# Repository / YAML (no dependency on where this script lives):
#   • Set REPO_ROOT (or pass --repo-root) — default YAML is $REPO_ROOT/.buildkite/test-nightly.yml
#   • Or set YML (or --yaml) — repo root is inferred as parent of the .buildkite directory
#   • Or run from inside the clone: git rev-parse --show-toplevel, else walk up from $PWD,
#     then from the script's directory, until .buildkite/test-nightly.yml exists
#
# Optional environment:
#   REPO_ROOT     - vllm-omni root (working directory for pytest); see above
#   YML           - path to test-nightly.yml (default: $REPO_ROOT/.buildkite/test-nightly.yml)
#   LOG_DIR       - logs + generated job scripts; when unset, a timestamped directory under
#                   $REPO_ROOT/logs/ is created:
#                     nightly_jobs_YYYYMMDD-HHMMSS       (default / YAML nightly steps)
#                     nightly_local_jobs_YYYYMMDD-HHMMSS (--test-type local only)
#                     nightly_stability_jobs_YYYYMMDD-HHMMSS (--test-type stability only)
#                   per-job *.log plus timing_summary.log after the run;
#                   perf JSON from tests/dfx/perf/results/ (produced this run) -> $LOG_DIR/perf_results/
#   TEST_TYPE     - comma-separated and/or repeated flags (default: all); see above
#   MODEL_TYPE    - comma-separated and/or repeated flags (default: all); see above
#   LABEL_SUBSTR  - YAML mode: substring of Buildkite step label; stability: substring of path/key/filename;
#                   local: substring of test_*.py basename or tests/dfx/perf/tests/*.json filename
#   DRY_RUN=1     - print extracted commands only; do not write scripts or run pytest
#   RUN_JOB_TIMEOUT_KILL_AFTER - seconds after SIGTERM before SIGKILL on inline pytest
#                              timeout (default: 60; baked into generated job scripts)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LABEL_SUBSTR="${LABEL_SUBSTR:-}"
TEST_TYPE_ENV="${TEST_TYPE:-all}"
MODEL_TYPE_ENV="${MODEL_TYPE:-all}"
TEST_TYPE_CLI_PARTS=()
MODEL_TYPE_CLI_PARTS=()
TEST_TYPE_FROM_CLI=0
MODEL_TYPE_FROM_CLI=0
DRY_RUN="${DRY_RUN:-0}"
# REPO_ROOT / YML / LOG_DIR resolved after CLI (do not assume script path vs repo root)

usage() {
  sed -n '2,66p' "$0" | sed 's/^# \{0,1\}//'
}

# Append comma-separated tokens from $2 into array named $1 (lowercased, trimmed).
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

# Build comma-separated TEST_TYPE from CLI parts or env; collapse to a single "all" if present.
_finalize_test_type_csv() {
  if [[ "${TEST_TYPE_FROM_CLI}" -eq 1 ]]; then
    if ((${#TEST_TYPE_CLI_PARTS[@]} == 0)); then
      echo "--test-type requires a non-empty value" >&2
      exit 2
    fi
    local _has_all=0 _has_stability=0 _has_local=0
    local _x
    for _x in "${TEST_TYPE_CLI_PARTS[@]}"; do
      if [[ "${_x}" == all ]]; then
        _has_all=1
      fi
      if [[ "${_x}" == stability ]]; then
        _has_stability=1
      fi
      if [[ "${_x}" == local ]]; then
        _has_local=1
      fi
    done
    if [[ "${_has_all}" -eq 1 ]]; then
      local _out="all"
      if [[ "${_has_stability}" -eq 1 ]]; then
        _out="${_out},stability"
      fi
      if [[ "${_has_local}" -eq 1 ]]; then
        _out="${_out},local"
      fi
      printf '%s' "${_out}"
      return 0
    fi
    local -A _seen=()
    local -a _out=()
    for _x in "${TEST_TYPE_CLI_PARTS[@]}"; do
      [[ ${_seen["${_x}"]+isset} ]] && continue
      _seen["${_x}"]=1
      _out+=("${_x}")
    done
    (IFS=','; printf '%s' "${_out[*]}")
  else
    printf '%s' "${TEST_TYPE_ENV}"
  fi
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
    --test-type)
      TEST_TYPE_FROM_CLI=1
      _split_append_csv_array TEST_TYPE_CLI_PARTS "$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

TEST_TYPE="$(_finalize_test_type_csv)"
MODEL_TYPE="$(_finalize_model_type_csv)"

# Default log directory basename: date + local timestamp; prefix depends on TEST_TYPE.
_nightly_log_dir_basename() {
  local ts has_local=0 has_stability=0 has_yaml=0 _t
  ts="$(date +%Y%m%d-%H%M%S)"
  IFS=',' read -ra _TTOK <<< "${TEST_TYPE}"
  for _t in "${_TTOK[@]}"; do
    _t="$(printf '%s' "${_t}" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    case "${_t}" in
      local) has_local=1 ;;
      stability) has_stability=1 ;;
      perf | acc | function | all) has_yaml=1 ;;
    esac
  done
  if [[ "${has_local}" -eq 1 && "${has_yaml}" -eq 0 && "${has_stability}" -eq 0 ]]; then
    printf 'nightly_local_jobs_%s' "${ts}"
  elif [[ "${has_stability}" -eq 1 && "${has_yaml}" -eq 0 && "${has_local}" -eq 0 ]]; then
    printf 'nightly_stability_jobs_%s' "${ts}"
  else
    printf 'nightly_jobs_%s' "${ts}"
  fi
}

BUILDKITE_REL=".buildkite/test-nightly.yml"

_find_repo_containing_nightly() {
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

# Resolve REPO_ROOT, YML (no relative path between script and repo)
if [[ -n "${YML:-}" && -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
  YML="$(cd "$(dirname "${YML}")" && pwd)/$(basename "${YML}")"
elif [[ -n "${YML:-}" ]]; then
  YML="$(cd "$(dirname "${YML}")" && pwd)/$(basename "${YML}")"
  if ! REPO_ROOT="$(_derive_repo_root_from_yml "${YML}")"; then
    echo "Could not derive REPO_ROOT from YML=${YML} (expected file at <repo>/.buildkite/test-nightly.yml)." >&2
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
    REPO_ROOT="$(_find_repo_containing_nightly "${PWD}" || true)"
  fi
  if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(_find_repo_containing_nightly "${SCRIPT_DIR}" || true)"
  fi
  if [[ -z "${REPO_ROOT}" ]]; then
    echo "Could not locate ${BUILDKITE_REL}. Set REPO_ROOT or YML, run from inside the vllm-omni clone," >&2
    echo "or place this script (or run from a cwd) under the repository tree." >&2
    exit 2
  fi
  YML="${REPO_ROOT}/${BUILDKITE_REL}"
fi

if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR="${REPO_ROOT}/logs/$(_nightly_log_dir_basename)"
fi

echo "Log directory: ${LOG_DIR}" >&2

# Require nightly YAML only when at least one non-stability test kind needs it (validated in Python too).
_needs_nightly_yml=0
IFS=',' read -ra _TTOK <<< "${TEST_TYPE}"
for _t in "${_TTOK[@]}"; do
  _t="$(printf '%s' "${_t}" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  case "${_t}" in
    perf | acc | function | all)
      _needs_nightly_yml=1
      ;;
  esac
done
if [[ "${_needs_nightly_yml}" -eq 1 ]] && [[ ! -f "${YML}" ]]; then
  echo "YAML not found: ${YML}" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}/jobs"
LOG_DIR="$(cd "${LOG_DIR}" && pwd)"
export REPO_ROOT LOG_DIR YML LABEL_SUBSTR TEST_TYPE MODEL_TYPE DRY_RUN

# Drop stale wrapper scripts, old tee logs, and perf manifest so a filtered run does not
# mix with the previous outputs.
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
  rm -f "${LOG_DIR}/jobs/.perf_job_keys"
  rm -f "${LOG_DIR}/jobs/.job_timeouts"
  rm -f "${REPO_ROOT}/logs/nightly_perf_manual.xlsx"
  rm -f "${LOG_DIR}/nightly_perf_manual.xlsx"
fi

# shellcheck disable=SC2016,SC1078,SC1079
python3 - <<'PY'
from __future__ import annotations

import os
import re
import shlex
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

ALLOWED_TEST_TYPES = frozenset({"all", "perf", "acc", "function", "stability", "local"})
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


def parse_test_types(raw: str) -> list[str]:
    """If 'all' appears, YAML dimension is unconstrained; extras like stability/local are kept."""
    parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        parts = ["all"]
    bad = [p for p in parts if p not in ALLOWED_TEST_TYPES]
    if bad:
        print(
            f"Invalid TEST_TYPE / --test-type value(s): {bad!r} "
            f"(allowed: {', '.join(sorted(ALLOWED_TEST_TYPES))})",
            file=sys.stderr,
        )
        sys.exit(2)
    if "all" in parts:
        out: list[str] = ["all"]
        if "stability" in parts:
            out.append("stability")
        if "local" in parts:
            out.append("local")
        return out
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
    """Flatten Buildkite steps; yield (leaf_step_dict, group_title)."""
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


def label_matches_test_type(label: str, test_types: list[str]) -> bool:
    """perf / acc / function / all — aligned with nightly step labels (OR across test_types)."""
    if "all" in test_types:
        return True
    has_perf = "Perf Test" in label
    has_acc = "Accuracy Test" in label
    is_function = not has_perf and not has_acc
    for t in test_types:
        if t in ("stability", "local"):
            continue
        if t == "perf" and has_perf:
            return True
        if t == "acc" and has_acc:
            return True
        if t == "function" and is_function:
            return True
    return False


def label_matches_model_type(label: str, model_types: list[str]) -> bool:
    """Omni · / TTS · / Diffusion — OR across model_types."""
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


# Fixed stability scripts (when stability is selected); MODEL_TYPES narrows which run.
STABILITY_CASES: list[tuple[str, str, tuple[str, ...]]] = [
    ("stability_qwen3_omni", "tests/dfx/stability/scripts/test_stability_qwen3_omni.py", ("omni",)),
    ("stability_qwen3_tts", "tests/dfx/stability/scripts/test_stability_qwen3_tts.py", ("tts",)),
    ("stability_voxcpm2", "tests/dfx/stability/scripts/test_stability_voxcpm2.py", ("tts",)),
    ("stability_qwen_image", "tests/dfx/stability/scripts/test_stability_qwen_image.py", ("diffusion",)),
    ("stability_wan22", "tests/dfx/stability/scripts/test_stability_wan22.py", ("diffusion",)),
    ("stability_hunyuan_image", "tests/dfx/stability/scripts/test_stability_hunyuan_image.py", ("diffusion",)),
]


def _stability_model_matches(model_types: list[str], families: tuple[str, ...]) -> bool:
    if "all" in model_types:
        return True
    return bool(set(model_types) & set(families))


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


def local_pytest_marker_expr(model_types: list[str]) -> str:
    """Build -m expression: (omni|tts|diffusion) markers combined with local_model."""
    if "all" in model_types:
        families: tuple[str, ...] = ("omni", "tts", "diffusion")
    else:
        families = tuple(model_types)
    if len(families) == 1:
        return f"{families[0]} and local_model"
    return f"({' or '.join(families)}) and local_model"


def local_test_files_by_filename(substr: str) -> list[str]:
    """Return repo-relative posix paths under tests/ whose basename contains substr."""
    tests_root = REPO_ROOT / "tests"
    if not tests_root.is_dir():
        return []
    matches: list[str] = []
    for path in sorted(tests_root.rglob("test_*.py")):
        if substr in path.name:
            matches.append(path.relative_to(REPO_ROOT).as_posix())
    return matches


PERF_TESTS_REL = Path("tests/dfx/perf/tests")
RUN_BENCHMARK_REL = Path("tests/dfx/perf/scripts/run_benchmark.py")
RUN_DIFFUSION_BENCHMARK_REL = Path("tests/dfx/perf/scripts/run_diffusion_benchmark.py")
TTS_PERF_JSON_HINTS = ("test_tts", "voxcpm", "higgs_audio")


def perf_json_model_family(json_basename: str) -> str:
    """Classify a perf JSON config as omni, tts, or diffusion (mirrors nightly YAML runners)."""
    name = json_basename.lower()
    if name.startswith("test_qwen3_omni"):
        return "omni"
    if any(hint in name for hint in TTS_PERF_JSON_HINTS):
        return "tts"
    return "diffusion"


def perf_json_runner(json_basename: str) -> Path:
    if perf_json_model_family(json_basename) in ("omni", "tts"):
        return RUN_BENCHMARK_REL
    return RUN_DIFFUSION_BENCHMARK_REL


def perf_json_matches_model_type(json_basename: str, model_types: list[str]) -> bool:
    if "all" in model_types:
        return True
    return perf_json_model_family(json_basename) in model_types


def local_perf_json_configs(substr: str, model_types: list[str]) -> list[tuple[str, str]]:
    """Return (json_rel_posix, runner_rel_posix) for perf configs whose filename contains substr."""
    perf_dir = REPO_ROOT / PERF_TESTS_REL
    if not perf_dir.is_dir():
        return []
    matches: list[tuple[str, str]] = []
    for path in sorted(perf_dir.glob("*.json")):
        if substr not in path.name:
            continue
        if not perf_json_matches_model_type(path.name, model_types):
            continue
        runner = perf_json_runner(path.name)
        if not (REPO_ROOT / runner).is_file():
            print(f"# skip (missing runner): {runner}", file=sys.stderr)
            continue
        matches.append(
            (
                path.relative_to(REPO_ROOT).as_posix(),
                runner.as_posix(),
            )
        )
    return matches


def _local_job_slug(label: str) -> str:
    slug = re.sub(r"[^\w\-.]+", "_", label, flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "filter"


def run_local_mode(
    jobs_dir: Path,
    model_types: list[str],
    perf_job_keys: list[str],
) -> int:
    """pytest -sv -m '<model markers> and local_model'; optional LABEL_SUBSTR filters."""
    marker_expr = local_pytest_marker_expr(model_types)
    matched = 0

    if LABEL_SUBSTR:
        rel_paths = local_test_files_by_filename(LABEL_SUBSTR)
        perf_pairs = local_perf_json_configs(LABEL_SUBSTR, model_types)

        if rel_paths:
            paths_arg = " ".join(shlex.quote(p) for p in rel_paths)
            pytest_line = f'pytest -sv -m "{marker_expr}" {paths_arg}'
            slug = _local_job_slug(LABEL_SUBSTR)
            key = f"local_pytest_{slug}"
            header = (
                f"# Local marker tests — MODEL_TYPE={','.join(model_types)}, "
                f"LABEL_SUBSTR={LABEL_SUBSTR!r} (test_*.py), files: {', '.join(rel_paths)}"
            )
            script_lines = [
                "#!/usr/bin/env bash",
                header,
                "set -euo pipefail",
                f'cd "{REPO_ROOT}"',
                pytest_line,
            ]
            _write_job_script(key, script_lines, jobs_dir)
            matched += 1

        for json_rel, runner_rel in perf_pairs:
            json_name = Path(json_rel).name
            pytest_line = (
                f'pytest -s -v -m "{marker_expr}" '
                f"{shlex.quote(runner_rel)} "
                f"--test-config-file {shlex.quote(json_rel)}"
            )
            slug = _local_job_slug(Path(json_rel).stem)
            key = f"local_perf_{slug}"
            header = (
                f"# Local perf benchmark — MODEL_TYPE={','.join(model_types)}, "
                f"config={json_name!r}, runner={Path(runner_rel).name}"
            )
            script_lines = [
                "#!/usr/bin/env bash",
                header,
                "set -euo pipefail",
                f'cd "{REPO_ROOT}"',
                pytest_line,
            ]
            _write_job_script(key, script_lines, jobs_dir)
            if key not in perf_job_keys:
                perf_job_keys.append(key)
            matched += 1

        if matched == 0:
            print(
                f"# skip local: no tests/**/test_*.py or "
                f"tests/dfx/perf/tests/*.json filename matches "
                f"LABEL_SUBSTR={LABEL_SUBSTR!r}",
                file=sys.stderr,
            )
        return matched

    pytest_line = f'pytest -sv -m "{marker_expr}"'
    key = "local_pytest"
    header = f"# Local marker tests — MODEL_TYPE={','.join(model_types)}"
    script_lines = [
        "#!/usr/bin/env bash",
        header,
        "set -euo pipefail",
        f'cd "{REPO_ROOT}"',
        pytest_line,
    ]
    _write_job_script(key, script_lines, jobs_dir)
    return 1


def run_stability_mode(jobs_dir: Path, model_types: list[str]) -> int:
    matched = 0
    for key, rel_posix, families in STABILITY_CASES:
        if not _stability_model_matches(model_types, families):
            continue
        if LABEL_SUBSTR:
            fn = Path(rel_posix).name
            if (
                LABEL_SUBSTR not in rel_posix
                and LABEL_SUBSTR not in key
                and LABEL_SUBSTR not in fn
            ):
                continue
        rel_path = REPO_ROOT / rel_posix
        if not rel_path.is_file():
            print(f"# skip (missing file): {rel_path}", file=sys.stderr)
            continue
        pytest_line = f"pytest -s -v --run-level full_model {rel_posix}"
        matched += 1
        script_lines = [
            "#!/usr/bin/env bash",
            f"# Stability: {key} — {rel_posix}",
            "set -euo pipefail",
            f'cd "{REPO_ROOT}"',
            pytest_line,
        ]
        _write_job_script(key, script_lines, jobs_dir)
    return matched


def main() -> None:
    test_types = parse_test_types(os.environ.get("TEST_TYPE", "all"))
    model_types = parse_model_types(os.environ.get("MODEL_TYPE", "all"))

    want_stability = "stability" in test_types
    want_local = "local" in test_types
    yaml_test_types = [t for t in test_types if t not in ("stability", "local")]
    needs_yaml = bool(yaml_test_types)

    jobs_dir = LOG_DIR / "jobs"
    if not DRY_RUN:
        jobs_dir.mkdir(parents=True, exist_ok=True)

    matched_stability = 0
    matched_yaml = 0
    matched_local = 0
    perf_job_keys: list[str] = []
    job_timeouts: dict[str, int] = {}

    if want_stability:
        matched_stability = run_stability_mode(jobs_dir, model_types)

    if want_local:
        matched_local = run_local_mode(jobs_dir, model_types, perf_job_keys)

    if needs_yaml:
        if not YML.is_file():
            print(f"YAML not found: {YML}", file=sys.stderr)
            sys.exit(1)

        data = yaml.safe_load(YML.read_text(encoding="utf-8"))
        top_steps = (data or {}).get("steps") or []

        for step, _grp in iter_leaf_steps(top_steps):
            label = step.get("label") or ""
            if LABEL_SUBSTR and LABEL_SUBSTR not in label:
                continue
            if not label_matches_test_type(label, yaml_test_types):
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
            if "Perf Test" in label and key not in perf_job_keys:
                perf_job_keys.append(key)

    if not DRY_RUN:
        manifest_path = LOG_DIR / "jobs" / ".perf_job_keys"
        if perf_job_keys:
            manifest_path.write_text("\n".join(perf_job_keys) + "\n", encoding="utf-8")
        elif manifest_path.is_file():
            manifest_path.unlink()
        _write_job_timeouts_manifest(jobs_dir, job_timeouts)

    errs: list[str] = []
    if want_stability and matched_stability == 0:
        errs.append(
            f"No stability jobs matched TEST_TYPE={test_types!r} MODEL_TYPE={model_types!r} "
            f"LABEL_SUBSTR={LABEL_SUBSTR!r} under {REPO_ROOT}"
        )
    if want_local and matched_local == 0:
        if LABEL_SUBSTR:
            errs.append(
                f"No local jobs matched LABEL_SUBSTR={LABEL_SUBSTR!r} "
                f"(tests/**/test_*.py or tests/dfx/perf/tests/*.json) "
                f"under {REPO_ROOT}"
            )
        else:
            errs.append(
                f"Local mode produced no jobs TEST_TYPE={test_types!r} MODEL_TYPE={model_types!r}"
            )
    if needs_yaml and matched_yaml == 0:
        errs.append(
            f"No YAML steps matched TEST_TYPE={test_types!r} MODEL_TYPE={model_types!r} "
            f"LABEL_SUBSTR={LABEL_SUBSTR!r} in {YML}"
        )
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        sys.exit(2)


main()
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

# shellcheck source=tools/run_jobs_common.sh
source "${SCRIPT_DIR}/../run_jobs_common.sh"

PERF_RESULTS_SRC="${REPO_ROOT}/tests/dfx/perf/results"

# Copy perf benchmark JSON written under tests/dfx/perf/results/ during this run into LOG_DIR.
_collect_perf_result_jsons() {
  local since_epoch="${1:?}"
  local dest="${LOG_DIR}/perf_results"

  if [[ ! -d "${PERF_RESULTS_SRC}" ]]; then
    return 0
  fi

  COLLECT_SINCE_EPOCH="${since_epoch}" \
  python3 - <<'PY'
import os
import shutil
import sys
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"]).resolve()
src = repo / "tests" / "dfx" / "perf" / "results"
dest = Path(os.environ["LOG_DIR"]).resolve() / "perf_results"
since = float(os.environ["COLLECT_SINCE_EPOCH"])

if not src.is_dir():
    sys.exit(0)

copied = 0
for path in sorted(src.rglob("*.json")):
    if not path.is_file():
        continue
    try:
        if path.stat().st_mtime < since:
            continue
    except OSError:
        continue
    rel = path.relative_to(src)
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)
    copied += 1

if copied:
    print(f"Collected {copied} perf result JSON file(s) -> {dest}", file=sys.stderr)
PY
}

# Run jobs: YAML perf steps first (if manifest exists), then generate Excel (always when
# manifest non-empty, even if some perf jobs failed; Excel uses whatever JSON exists on disk),
# then remaining jobs. Paths align with nightly Buildkite (tests/dfx/perf/results).
run_generated_jobs_with_tee() {
  set -o pipefail
  local any_fail=0
  local _run_start
  local perf_list="${LOG_DIR}/jobs/.perf_job_keys"
  local -a perf_keys=()
  _run_jobs_reset_timing
  _run_start="$(_run_jobs_epoch_seconds)"
  if [[ -f "${perf_list}" ]] && [[ -s "${perf_list}" ]]; then
    mapfile -t perf_keys < "${perf_list}"
  fi

  local k
  for k in "${perf_keys[@]}"; do
    [[ -n "${k}" ]] || continue
    if [[ ! -f "${LOG_DIR}/jobs/${k}.sh" ]]; then
      echo "warning: perf job script missing, skip: ${LOG_DIR}/jobs/${k}.sh" >&2
      any_fail=1
      continue
    fi
    _run_one_job_with_timing "${LOG_DIR}/jobs/${k}.sh" || any_fail=1
  done

  if ((${#perf_keys[@]})); then
    local excel_out excel_log excel_status excel_repo_logs excel_start excel_end excel_elapsed
    excel_repo_logs="${REPO_ROOT}/logs"
    mkdir -p "${excel_repo_logs}"
    excel_out="${excel_repo_logs}/nightly_perf_$(date -u +%Y%m%d-%H%M%S).xlsx"
    excel_log="${excel_repo_logs}/generate_nightly_perf_excel.log"
    if [[ "${any_fail}" -ne 0 ]]; then
      echo "Note: one or more perf jobs failed; still running generate_nightly_perf_excel.py " \
        "(report reflects JSON already under tests/dfx/perf/results)." >&2
    fi
    echo "==> python3 tools/nightly/generate_nightly_perf_excel.py -> ${excel_out}  (tee ${excel_log})" >&2
    excel_start="$(_run_jobs_epoch_seconds)"
    (
      cd "${REPO_ROOT}" && python3 tools/nightly/generate_nightly_perf_excel.py \
        --input-dir "${REPO_ROOT}/tests/dfx/perf/results" \
        --diffusion-input-dir "${REPO_ROOT}/tests/dfx/perf/results" \
        --output-file "${excel_out}"
    ) 2>&1 | tee "${excel_log}"
    excel_status="${PIPESTATUS[0]}"
    excel_end="$(_run_jobs_epoch_seconds)"
    excel_elapsed=$((excel_end - excel_start))
    _run_jobs_record_timing "generate_nightly_perf_excel" "${excel_elapsed}" "${excel_status}"
    if [[ "${excel_status}" -eq 0 ]]; then
      echo "    finished in $(_run_jobs_format_duration "${excel_elapsed}")" >&2
    else
      echo "    failed after $(_run_jobs_format_duration "${excel_elapsed}") (exit ${excel_status})" >&2
      any_fail=1
      echo "generate_nightly_perf_excel.py failed (exit ${excel_status}). See ${excel_log}" >&2
    fi
  fi

  local -A _perf_done=()
  for k in "${perf_keys[@]}"; do
    [[ -n "${k}" ]] && _perf_done["${k}"]=1
  done

  local _job base
  shopt -s nullglob
  for _job in "${LOG_DIR}/jobs"/*.sh; do
    base="$(basename "${_job}" .sh)"
    [[ ${_perf_done["${base}"]+isset} ]] && continue
    _run_one_job_with_timing "${_job}" || any_fail=1
  done
  shopt -u nullglob

  _collect_perf_result_jsons "${_run_start}"

  _run_jobs_print_timing_summary "${_run_start}" "${any_fail}"
  if [[ "${any_fail}" -ne 0 ]]; then
    return 1
  fi
  return 0
}

run_generated_jobs_with_tee || exit 1
