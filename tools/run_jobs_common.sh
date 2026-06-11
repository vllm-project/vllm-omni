#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared timing helpers for tools/run_*_jobs.sh (source this file; do not execute).

if [[ -n "${_RUN_JOBS_COMMON_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
_RUN_JOBS_COMMON_LOADED=1

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
    echo "==> ${_job}  (tee ${out}, timeout ${timeout_min}m)" >&2
  else
    echo "==> ${_job}  (tee ${out})" >&2
  fi
  start="$(_run_jobs_epoch_seconds)"
  if [[ -n "${timeout_min}" ]]; then
    (cd "${REPO_ROOT}" && timeout "${timeout_min}m" bash "${_job}") 2>&1 | tee "${out}"
  else
    (cd "${REPO_ROOT}" && bash "${_job}") 2>&1 | tee "${out}"
  fi
  job_status="${PIPESTATUS[0]}"
  end="$(_run_jobs_epoch_seconds)"
  elapsed=$((end - start))
  _run_jobs_record_timing "${base}" "${elapsed}" "${job_status}"
  if [[ "${job_status}" -eq 0 ]]; then
    echo "    finished in $(_run_jobs_format_duration "${elapsed}")" >&2
  elif [[ "${job_status}" -eq 124 && -n "${timeout_min}" ]]; then
    echo "    timed out after ${timeout_min}m (Buildkite timeout_in_minutes)" >&2
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
