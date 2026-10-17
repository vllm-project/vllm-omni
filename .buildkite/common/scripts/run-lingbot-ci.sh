#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

STAGE="${1:?unit, smoke, generate, collect, aggregate, or evaluate}"
PROFILE="${2:?profile JSON}"
OUTPUT="${3:?artifact directory}"
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/pytest.xml" "$OUTPUT/setup-failure.json"
# Retain a failed report even if the agent force-kills the entire process group.
# Pytest replaces this file when it completes; successful exit removes the marker.
cat > "$OUTPUT/pytest.xml" <<'XML'
<testsuite name="lingbot-launcher" tests="1" failures="1"><testcase name="incomplete"><failure message="LingBot CI did not finish; see run.log"/></testcase></testsuite>
XML
cat > "$OUTPUT/setup-failure.json" <<'JSON'
{"passed": false, "status": "incomplete"}
JSON
exec > >(tee "$OUTPUT/run.log") 2>&1

finish() {
    local status=$?
    if [[ "$status" -eq 0 ]]; then
        rm -f "$OUTPUT/setup-failure.json"
    else
        python - "$OUTPUT" "$status" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
output = Path(sys.argv[1])
message = "LingBot CI failed or was interrupted; see run.log (exit " + sys.argv[2] + ")"
suite = ET.Element("testsuite", name="lingbot-setup", tests="1", failures="1")
case = ET.SubElement(suite, "testcase", name="setup")
ET.SubElement(case, "failure", message=message).text = message
path = output / "pytest.xml"
try:
    incomplete = any(case.get("name") == "incomplete" for case in ET.parse(path).iter("testcase"))
except (OSError, ET.ParseError):
    incomplete = True
if incomplete:
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
(output / "setup-failure.json").write_text(json.dumps({"passed": False, "error": message}) + "\n")
PY
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "${LINGBOT_CASE_SHARD:-}" ]]; then
    PROFILE="$(python -m tests.dfx.lingbot_world.prepare shard --profile "$PROFILE" \
        --case-shard "$LINGBOT_CASE_SHARD" --root "$OUTPUT")"
fi

if [[ "$STAGE" == "unit" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    cp "$PROFILE" "$OUTPUT/policy.json"
    python - "$OUTPUT" <<'UNIT_ENV'
import importlib.metadata
import platform
import sys
from pathlib import Path
from benchmarks.lingbot_world.regression import write_json
from tests.dfx.lingbot_world.runner import ROOT, git_revision
write_json(Path(sys.argv[1]) / "environment.json", {
    "python": sys.version,
    "machine": platform.machine(),
    "source_revision": git_revision(ROOT),
    "accelerator_visibility": "",
    "packages": {name: importlib.metadata.version(name)
                 for name in ("pytest", "numpy", "torch", "vllm", "diffusers", "transformers")},
})
UNIT_ENV
    python -m pytest -sv -p tests.dfx.lingbot_world.conftest \
        tests/benchmarks/test_lingbot_world_*.py \
        tests/diffusion/models/lingbot_world/test_offload_lingbot_world.py \
        'tests/diffusion/offloader/test_backend_plan_contract.py::test_text_encoder_only_offload_keeps_early_dit_projections_resident[cpu]' \
        'tests/diffusion/offloader/test_backend_plan_contract.py::test_text_encoder_only_offload_handles_cached_dit_owner_references[cpu]' \
        --lingbot-output="$OUTPUT" -m 'core_model and cpu' --run-level core_model \
        --junitxml="$OUTPUT/pytest.xml"
elif [[ "$STAGE" == "aggregate" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    GENERATION="${4:?measurement artifact directory}"
    if [[ "${BUILDKITE:-}" == "true" ]]; then
        # Matrix jobs share logical step keys; their artifact prefixes are unique.
        buildkite-agent artifact download "$GENERATION/**" .
    fi
    python -m pytest -sv tests/dfx/lingbot_world/test_aggregate.py \
        --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" --lingbot-generation="$GENERATION" \
        -m 'full_model and cpu' --run-level full_model --junitxml="$OUTPUT/pytest.xml"
elif [[ "$STAGE" == "evaluate" ]]; then
    GENERATION="${4:?generation artifact directory}"
    if [[ "${BUILDKITE:-}" == "true" ]]; then
        buildkite-agent artifact download "$GENERATION/**" .
    fi
    EVALUATOR_LOCK="$(python -c 'from tests.dfx.lingbot_world.prepare import evaluator_cache_key; print(evaluator_cache_key())')"
    export LINGBOT_EVALUATOR_ROOT="${LINGBOT_EVALUATOR_ROOT:-${HF_HOME:-${TMPDIR:-/tmp}}/lingbot-evaluators-$EVALUATOR_LOCK}"
    python -m tests.dfx.lingbot_world.prepare evaluators --root "$LINGBOT_EVALUATOR_ROOT"
    export LINGBOT_EVALUATOR_PYTHON="$LINGBOT_EVALUATOR_ROOT/venv/bin/python"
    pytest -sv tests/dfx/lingbot_world/test_quality.py \
        --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" --lingbot-generation="$GENERATION" \
        -m 'full_model and diffusion and cuda and H100 and cards_1' --run-level full_model \
        --junitxml="$OUTPUT/pytest.xml"
elif [[ "$STAGE" == "generate" || "$STAGE" == "smoke" || "$STAGE" == "collect" ]]; then
    PLATFORM="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["platform"])' "$PROFILE")"
    if [[ "$STAGE" == "generate" && "$PLATFORM" == "cuda" ]]; then
        CUDA_VISIBLE_DEVICES="$(python -m tests.dfx.lingbot_world.prepare devices --profile "$PROFILE" --root "$OUTPUT")"
        export CUDA_VISIBLE_DEVICES
        export LINGBOT_ALLOCATION_FILE="$OUTPUT/allocation.json"
    elif [[ "$PLATFORM" == "npu" && ( "$STAGE" == "generate" || "$STAGE" == "collect" ) ]]; then
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
    elif [[ "$STAGE" == "collect" ]]; then
        TEST_FILE="tests/dfx/lingbot_world/test_collect.py"
        EXTRA_ARGS+=(--lingbot-measurement-repeat="${LINGBOT_MEASUREMENT_REPEAT:?0, 1, or 2}")
    fi
    pytest -sv "$TEST_FILE" --lingbot-profile="$PROFILE" --lingbot-output="$OUTPUT" \
        --lingbot-reference-root="$REFERENCE" -m "full_model and diffusion and $PLATFORM and cards_$CARDS" \
        --run-level full_model --junitxml="$OUTPUT/pytest.xml" "${EXTRA_ARGS[@]}"
else
    echo "Unknown LingBot CI stage: $STAGE" >&2
    exit 2
fi
