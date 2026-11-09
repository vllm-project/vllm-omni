#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail
STAGE="${1:?generate, collect, aggregate, precision, performance, wbench, vbench, stability}"
PROFILE="${2:?profile JSON}"
OUTPUT="${3:?output directory}"
mkdir -p "$OUTPUT"
ARGS=(--lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" --run-level full_model --junitxml="$OUTPUT/pytest.xml")
case "$STAGE" in
    precision|performance|wbench|vbench|aggregate)
        GENERATION="${4:?generation artifact directory}"
        if [[ "${BUILDKITE:-}" == "true" ]]; then
            buildkite-agent artifact download "$GENERATION/**" .
        fi
        ARGS+=(--lingbot-generation="$GENERATION")
        if [[ "$STAGE" == "wbench" || "$STAGE" == "vbench" ]]; then
            KEY="$(python -c 'from tests.dfx.lingbot_world.prepare import evaluator_cache_key; print(evaluator_cache_key())')"
            export LINGBOT_EVALUATOR_ROOT="${LINGBOT_EVALUATOR_ROOT:-${HF_HOME:-${TMPDIR:-/tmp}}/lingbot-evaluators-$KEY}"
            python -m tests.dfx.lingbot_world.prepare evaluators --root "$LINGBOT_EVALUATOR_ROOT"
            export LINGBOT_EVALUATOR_PYTHON="$LINGBOT_EVALUATOR_ROOT/venv/bin/python"
            TEST="tests/dfx/lingbot_world/test_$STAGE.py"
            MARK='full_model and diffusion and cuda and H100 and cards_1'
        else
            export CUDA_VISIBLE_DEVICES=""
            MARK='full_model and cpu'
            TEST="tests/dfx/lingbot_world/test_$STAGE.py"
            if [[ "$STAGE" == "aggregate" ]]; then
                TEST='tests/dfx/lingbot_world/test_collect.py::test_aggregate_npu_measurements'
            fi
        fi
        ;;
    generate|collect|stability)
        PLATFORM="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["platform"])' "$PROFILE")"
        python -m tests.dfx.lingbot_world.prepare devices --profile "$PROFILE" --root "$OUTPUT"
        VISIBLE="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["visible_devices"])' "$OUTPUT/allocation.json")"
        if [[ "$PLATFORM" == "cuda" ]]; then export CUDA_VISIBLE_DEVICES="$VISIBLE";
        else export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE"; fi
        REFERENCE="${TMPDIR:-/tmp}/lingbot-reference-${BUILDKITE_JOB_ID:-$$}"
        PREPARE=reference
        if [[ "$STAGE" == "stability" ]]; then PREPARE=model; fi
        python -m tests.dfx.lingbot_world.prepare "$PREPARE" --profile "$PROFILE" --root "$REFERENCE" --env-output "$OUTPUT/model.json"
        export LINGBOT_MODEL_PATH
        LINGBOT_MODEL_PATH="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["model_path"])' "$OUTPUT/model.json")"
        MARK="full_model and diffusion and $PLATFORM and cards_4"
        if [[ "$STAGE" == "stability" ]]; then
            TEST="tests/dfx/lingbot_world/test_stability.py::test_${PLATFORM}_long_stability"
        else
            TEST="tests/dfx/lingbot_world/test_collect.py::test_collect_${PLATFORM}_measurements"
            ARGS+=(--lingbot-reference-root="$REFERENCE")
            if [[ "$STAGE" == "collect" ]]; then
                [[ "$PLATFORM" == "npu" ]]
                ARGS+=(--lingbot-measurement-repeat="${LINGBOT_MEASUREMENT_REPEAT:?0, 1 or 2}")
            fi
        fi
        ;;
    *) echo "Unknown LingBot stage: $STAGE" >&2; exit 2 ;;
esac
python -m pytest -sv "$TEST" -m "$MARK" "${ARGS[@]}"
