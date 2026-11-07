#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Workload preparation for the paired-checkout DFX tests. CI reporting and
# failure handling are shared with ordinary pytest jobs.
set -euo pipefail

if [[ "${1:-}" != "--ci-wrapped" ]]; then
    OUTPUT="${3:?artifact directory}"
    REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    exec bash "$REPOSITORY/.buildkite/common/scripts/run-pytest-ci.sh" "$OUTPUT" bash "$0" --ci-wrapped "$@"
fi
shift
STAGE="${1:?smoke, generate, reference, collect, aggregate, evaluate, or a reference collection stage}"
PROFILE="${2:?profile JSON}"
OUTPUT="${3:?artifact directory}"
REQUESTED_STAGE="$STAGE"
REFERENCE_ONLY_REQUEST="${LINGBOT_REFERENCE_ONLY-0}"
case "$REFERENCE_ONLY_REQUEST" in
    0) ;;
    1)
        case "$STAGE" in
            generate) STAGE="reference" ;;
            evaluate) STAGE="reference-quality" ;;
            collect) STAGE="reference-collect" ;;
            aggregate) STAGE="reference-aggregate" ;;
            smoke|reference|reference-quality|reference-collect|reference-aggregate) ;;
            *)
                echo "Unknown LingBot CI stage in reference collection mode: $STAGE" >&2
                exit 2
                ;;
        esac
        ;;
    *)
        echo "LINGBOT_REFERENCE_ONLY must be unset, 0, or 1" >&2
        exit 2
        ;;
esac
if [[ "$STAGE" == "reference" || "$STAGE" == "reference-quality" || "$STAGE" == "reference-collect" || "$STAGE" == "reference-aggregate" ]]; then
    export LINGBOT_CI_PHASE="reference-collection"
else
    export LINGBOT_CI_PHASE="acceptance"
fi
echo "+++ LingBot World phase=$LINGBOT_CI_PHASE stage=$STAGE (requested $REQUESTED_STAGE)"
if [[ "$LINGBOT_CI_PHASE" == "reference-collection" ]]; then
    echo "Reference-only calibration collection; these results do not establish candidate acceptance."
fi
python3 - "$OUTPUT" "$REQUESTED_STAGE" "$STAGE" "$PROFILE" "$REFERENCE_ONLY_REQUEST" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
(output / "ci-configuration.json").write_text(json.dumps({
    "requested_stage": sys.argv[2],
    "effective_stage": sys.argv[3],
    "phase": os.environ["LINGBOT_CI_PHASE"],
    "reference_only": os.environ["LINGBOT_CI_PHASE"] == "reference-collection",
    "reference_only_opt_in": sys.argv[5] == "1",
    "profile": sys.argv[4],
    "case_shard": os.environ.get("LINGBOT_CASE_SHARD"),
    "buildkite_job_id": os.environ.get("BUILDKITE_JOB_ID"),
    "build_message": os.environ.get("BUILDKITE_MESSAGE"),
}, indent=2) + "\n")
PY

if [[ -n "${LINGBOT_CASE_SHARD:-}" ]]; then
    PROFILE="$(python -m tests.dfx.lingbot_world.prepare shard --profile "$PROFILE" \
        --case-shard "$LINGBOT_CASE_SHARD" --root "$OUTPUT")"
fi

if [[ "$STAGE" == "aggregate" || "$STAGE" == "reference-aggregate" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    GENERATION="${4:?measurement artifact directory}"
    if [[ "${BUILDKITE:-}" == "true" ]]; then
        # Matrix jobs share logical step keys; their artifact prefixes are unique.
        buildkite-agent artifact download "$GENERATION/**" .
    fi
    TEST_FILE="tests/dfx/lingbot_world/test_aggregate.py"
    if [[ "$STAGE" == "reference-aggregate" ]]; then
        TEST_FILE="tests/dfx/lingbot_world/test_reference.py::test_aggregate_reference_measurements"
    fi
    python -m pytest -sv "$TEST_FILE" \
        --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" --lingbot-generation="$GENERATION" \
        -m 'full_model and cpu' --run-level full_model --junitxml="$OUTPUT/pytest.xml"
elif [[ "$STAGE" == "evaluate" || "$STAGE" == "reference-quality" ]]; then
    GENERATION="${4:?generation artifact directory}"
    if [[ "${BUILDKITE:-}" == "true" ]]; then
        buildkite-agent artifact download "$GENERATION/**" .
    fi
    EVALUATOR_LOCK="$(python -c 'from tests.dfx.lingbot_world.prepare import evaluator_cache_key; print(evaluator_cache_key())')"
    export LINGBOT_EVALUATOR_ROOT="${LINGBOT_EVALUATOR_ROOT:-${HF_HOME:-${TMPDIR:-/tmp}}/lingbot-evaluators-$EVALUATOR_LOCK}"
    python -m tests.dfx.lingbot_world.prepare evaluators --root "$LINGBOT_EVALUATOR_ROOT"
    export LINGBOT_EVALUATOR_PYTHON="$LINGBOT_EVALUATOR_ROOT/venv/bin/python"
    QUALITY_ARGS=()
    if [[ -n "${LINGBOT_QUALITY_CASE_INDEX:-}" ]]; then
        QUALITY_ARGS+=(--lingbot-quality-case-index="$LINGBOT_QUALITY_CASE_INDEX")
    fi
    TEST_FILE="tests/dfx/lingbot_world/test_quality.py"
    if [[ "$STAGE" == "reference-quality" ]]; then
        TEST_FILE="tests/dfx/lingbot_world/test_reference.py::test_collect_reference_quality"
    fi
    pytest -sv "$TEST_FILE" \
        --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" --lingbot-generation="$GENERATION" \
        -m 'full_model and diffusion and cuda and H100 and cards_1' --run-level full_model \
        --junitxml="$OUTPUT/pytest.xml" "${QUALITY_ARGS[@]}"
elif [[ "$STAGE" == "generate" || "$STAGE" == "reference" || "$STAGE" == "smoke" || "$STAGE" == "collect" || "$STAGE" == "reference-collect" ]]; then
    PLATFORM="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["platform"])' "$PROFILE")"
    if [[ "$STAGE" == "reference-collect" && "$PLATFORM" != "npu" ]]; then
        echo "Split reference collection requires an NPU profile" >&2
        exit 2
    fi
    if [[ ( "$STAGE" == "generate" || "$STAGE" == "reference" ) && "$PLATFORM" == "cuda" ]]; then
        CUDA_VISIBLE_DEVICES="$(python -m tests.dfx.lingbot_world.prepare devices --profile "$PROFILE" --root "$OUTPUT")"
        export CUDA_VISIBLE_DEVICES
        export LINGBOT_ALLOCATION_FILE="$OUTPUT/allocation.json"
    elif [[ "$PLATFORM" == "npu" && ( "$STAGE" == "generate" || "$STAGE" == "reference" || "$STAGE" == "collect" || "$STAGE" == "reference-collect" ) ]]; then
        python -m tests.dfx.lingbot_world.prepare devices --profile "$PROFILE" --root "$OUTPUT"
        ASCEND_RT_VISIBLE_DEVICES="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["visible_devices"])' "$OUTPUT/allocation.json")"
        export ASCEND_RT_VISIBLE_DEVICES
        export LINGBOT_ALLOCATION_FILE="$OUTPUT/allocation.json"
    fi
    REFERENCE="${TMPDIR:-/tmp}/lingbot-reference-${BUILDKITE_JOB_ID:-$$}"
    python -m tests.dfx.lingbot_world.prepare reference --profile "$PROFILE" \
        --root "$REFERENCE" --env-output "$OUTPUT/model.json"
    LINGBOT_MODEL_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$OUTPUT/model.json")"
    export LINGBOT_MODEL_PATH
    CARDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dies"])' "$PROFILE")"
    TEST_FILE="tests/dfx/lingbot_world/test_generate.py"
    EXTRA_ARGS=()
    if [[ "$STAGE" == "smoke" ]]; then
        TEST_FILE="tests/dfx/lingbot_world/test_smoke.py"
    elif [[ "$STAGE" == "reference" ]]; then
        TEST_FILE="tests/dfx/lingbot_world/test_reference.py::test_collect_${PLATFORM}_reference_cohort"
    elif [[ "$STAGE" == "collect" || "$STAGE" == "reference-collect" ]]; then
        if [[ "$STAGE" == "reference-collect" ]]; then
            TEST_FILE="tests/dfx/lingbot_world/test_reference.py::test_collect_npu_reference_measurement"
        else
            TEST_FILE="tests/dfx/lingbot_world/test_collect.py"
        fi
        EXTRA_ARGS+=(--lingbot-measurement-repeat="${LINGBOT_MEASUREMENT_REPEAT:?0, 1, or 2}")
    fi
    pytest -sv "$TEST_FILE" --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" \
        --lingbot-reference-root="$REFERENCE" -m "full_model and diffusion and $PLATFORM and cards_$CARDS" \
        --run-level full_model --junitxml="$OUTPUT/pytest.xml" "${EXTRA_ARGS[@]}"
else
    echo "Unknown LingBot CI stage: $STAGE" >&2
    exit 2
fi
