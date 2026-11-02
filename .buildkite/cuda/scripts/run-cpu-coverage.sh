#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Keep shell PID expansion outside Buildkite upload and Kubernetes EnvVar expansion.
set +e
if [ "${VLLM_OMNI_NATIVE_PHASE_PROBE:-0}" = "1" ]; then
  PROBE_OUTPUT="$(mktemp -d /workspace/build/native-cpu-phase-probe.XXXXXX)"
  if [ -n "${PROBE_OUTPUT}" ]; then
    export NATIVE_PHASE_PROBE_OUTPUT_DIR="${PROBE_OUTPUT}"
    export NATIVE_PHASE_PROBE_PARENT_PID=${BASHPID}
    export NATIVE_PHASE_PROBE_SLOW_SECONDS=1.0
    export NATIVE_PHASE_PROBE_MAX_EVENTS=2000
    export NATIVE_PHASE_PROBE_MAX_BYTES=2097152
    export NATIVE_PHASE_PROBE_CHECKPOINT_REPORTS=1000
    printf 'native-phase-probe: shell enabled=1 pid=%s pytest=%s output=%s\n' "${BASHPID}" "$(command -v pytest)" "${PROBE_OUTPUT}" >&2
  else
    echo "native-phase-probe: output unavailable; original pytest continues" >&2
  fi
else
  printf 'native-phase-probe: shell enabled=0 pid=%s\n' "${BASHPID}" >&2
fi
REPORT="coverage-core-model-cpu-${BUILDKITE_STEP_ID:-local}.xml"
pytest -sv tests/ -m 'core_model and cpu' -n 4 --dist loadgroup --cov=vllm_omni --cov-report=term-missing:skip-covered --cov-report=xml:"${REPORT}"
EXIT=$?
gzip -9 -f "${REPORT}"
exit "$EXIT"
