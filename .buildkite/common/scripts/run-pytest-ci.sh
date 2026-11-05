#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run a pytest job (including its setup command) with failure evidence even if
# setup exits before pytest or the agent interrupts the process group. The job
# supplies --junitxml=$OUTPUT/pytest.xml and its ordinary artifact_paths.
set -euo pipefail

OUTPUT="${1:?artifact directory}"
shift
if [[ "$#" -eq 0 ]]; then
    echo "run-pytest-ci.sh: command required" >&2
    exit 2
fi
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/pytest.xml" "$OUTPUT/setup-failure.json"
# A hard kill cannot run a trap. Initialize a failed report before any setup.
cat > "$OUTPUT/pytest.xml" <<'XML'
<testsuite name="ci-launcher" tests="1" failures="1"><testcase name="incomplete"><failure message="CI job did not finish; see Buildkite job output"/></testcase></testsuite>
XML
cat > "$OUTPUT/setup-failure.json" <<'JSON'
{"passed": false, "status": "incomplete"}
JSON

finish() {
    local status=$?
    if [[ "$status" -eq 0 ]]; then
        if python3 - "$OUTPUT/pytest.xml" <<'VERIFY'
import sys
import xml.etree.ElementTree as ET

try:
    root = ET.parse(sys.argv[1]).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(case.get("name") == "incomplete" for case in cases):
        raise ValueError("pytest did not replace the initial failure report")
    if all(case.find("skipped") is not None for case in cases):
        raise ValueError("all selected tests were skipped")
    if list(root.iter("failure")) or list(root.iter("error")):
        raise ValueError("JUnit reports failure despite a zero command exit")
except (OSError, ET.ParseError, ValueError) as error:
    print(f"CI report validation failed: {error}", file=sys.stderr)
    sys.exit(1)
VERIFY
        then
            rm -f "$OUTPUT/setup-failure.json"
        else
            status=1
        fi
    fi
    if [[ "$status" -ne 0 ]]; then
        python3 - "$OUTPUT" "$status" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

output = Path(sys.argv[1])
message = "CI job failed or was interrupted; see Buildkite job output (exit " + sys.argv[2] + ")"
suite = ET.Element("testsuite", name="ci-setup", tests="1", failures="1")
case = ET.SubElement(suite, "testcase", name="setup")
ET.SubElement(case, "failure", message=message).text = message
path = output / "pytest.xml"
try:
    cases = list(ET.parse(path).iter("testcase"))
    incomplete = (
        not cases
        or any(case.get("name") == "incomplete" for case in cases)
        or all(case.find("skipped") is not None for case in cases)
    )
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

if [[ -n "${CI_POLICY_FILE:-}" ]]; then
    cp "$CI_POLICY_FILE" "$OUTPUT/policy.json"
fi

python3 - "$OUTPUT" <<'PY'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

packages = {}
for name in ("pytest", "numpy", "torch", "vllm", "diffusers", "transformers"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
(Path(sys.argv[1]) / "environment.json").write_text(json.dumps({
    "python": sys.version,
    "machine": platform.machine(),
    "source_revision": revision.stdout.strip() if revision.returncode == 0 else os.environ.get("BUILDKITE_COMMIT"),
    "accelerator_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "packages": packages,
}, indent=2) + "\n")
PY
"$@"
