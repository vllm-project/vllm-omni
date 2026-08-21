#!/usr/bin/env bash
# Run a model's E2E tests once per entry mode as separate coverage runs and upload
# the reports, so per-mode attribution survives (see vllm-project/vllm-omni#5332).
#
# Usage: run_cov_split.sh --model-id <id> [--offline <paths>] [--online <paths>] \
#                         -- <pytest args...>
#
#   --model-id  Names the artifacts: coverage-<id>-{offline,online}-<step id>.xml.gz.
#               Use the model's directory name under vllm_omni/*/models/.
#   --offline   Pytest path(s) for the offline half. Repeatable.
#   --online    Pytest path(s) for the online half. Repeatable.
#   --          Everything after this is forwarded verbatim to every pytest run,
#               quoting preserved: the job's own -m expression, --run-level, and
#               anything else it needs (e.g. --test-config-file).
#
# At least one of --offline/--online is required, so a job with only one mode runs
# just that half. Total runtime is bounded by the step's timeout_in_minutes.
#
# Exits non-zero if any half or the upload failed.
set -uo pipefail

MODEL_ID=""
OFFLINE=()
ONLINE=()
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --offline)  OFFLINE+=("$2"); shift 2 ;;
        --online)   ONLINE+=("$2"); shift 2 ;;
        --)         shift; PYTEST_ARGS=("$@"); break ;;
        *) echo "run_cov_split.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL_ID}" ]]; then
    echo "run_cov_split.sh: missing --model-id" >&2
    exit 2
fi

if ((${#OFFLINE[@]} == 0 && ${#ONLINE[@]} == 0)); then
    echo "run_cov_split.sh: need at least one of --offline / --online" >&2
    exit 2
fi

# Two steps can test the same model, and artifacts are scoped to the build (a
# nightly one carries the ready, merge and nightly tiers at once), so unsuffixed
# reports would share a path -- a glob download then races to rename onto one file.
# Not BUILDKITE_JOB_ID: the step id survives retries, so attempts collapse under
# Buildkite's latest-attempt lookup instead of accumulating. A parallel step would
# need BUILDKITE_PARALLEL_JOB as well; no pilot is one.
REPORT_SUFFIX="${BUILDKITE_STEP_ID:-local}"
OFFLINE_REPORT="coverage-${MODEL_ID}-offline-${REPORT_SUFFIX}.xml"
ONLINE_REPORT="coverage-${MODEL_ID}-online-${REPORT_SUFFIX}.xml"

run_half() {
    local mode="$1"
    shift
    local report="coverage-${MODEL_ID}-${mode}-${REPORT_SUFFIX}.xml"
    echo "--- coverage: ${MODEL_ID} ${mode}"
    pytest -s -v "$@" "${PYTEST_ARGS[@]}" --cov=vllm_omni --cov-report="xml:${report}"
    local status=$?
    # Cobertura lists every statement in vllm_omni whether or not it ran, so the
    # report is ~7MB no matter how little the job covers. The repetition is what
    # dominates it, and gzip takes that to ~400KB.
    gzip -9 -f "${report}" || status=1
    return "${status}"
}

# Clear both modes, not just the ones being run: the upload glob matches both, so
# a report left by an earlier run would otherwise be published as if the mode it
# belongs to had run this time.
rm -f "${OFFLINE_REPORT}" "${ONLINE_REPORT}" "${OFFLINE_REPORT}.gz" "${ONLINE_REPORT}.gz"

EXIT=0
((${#OFFLINE[@]})) && { run_half offline "${OFFLINE[@]}" || EXIT=1; }
((${#ONLINE[@]})) && { run_half online "${ONLINE[@]}" || EXIT=1; }

buildkite-agent artifact upload "coverage-${MODEL_ID}-*-${REPORT_SUFFIX}.xml.gz" || EXIT=1

exit "${EXIT}"
