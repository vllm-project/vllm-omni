# SenseNova-U1 platform pixel references

The weekly SenseNova H100 and MI300 jobs run the same smoke and pixel-accuracy
contracts. All model tests use the `slow` mark and run from the platform weekly
pipelines, outside merge CI. The smoke invocation checks generation and output
dimensions and excludes `pixel_accuracy` tests. Each accuracy case then runs in
its own pytest process on the same worker and compares ten RGB samples per image
with a per-channel tolerance of ±10. The jobs preserve PNG, JSON and JUnit
artifacts and fail if the smoke invocation or either accuracy case fails.

The accuracy tests resolve `SenseNova/SenseNova-U1-8B-MoT` at the immutable
revision `bfa9b436503cb8aed4f2bc60e3236710cc77468d` before loading the model.
`sensenova_u1_reference.json` is the reviewed H100 reference and
`sensenova_u1_reference_mi300.json` is the reviewed MI300 reference. Each file
records the same input contract, sampled pixels, output PNG hash, GPU, driver,
software versions, source hashes and capture image. Platform references are not
interchangeable; a ROCm run must never compare against or refresh the H100 file.
Software and source versions are provenance, rather than equality requirements:
the gate must be able to test new code against an existing reference.

`SENSENOVA_U1_CI_IMAGE` records provenance; it does not select or pin the running
container. The H100 hardware preset selects the CI image for `BUILDKITE_COMMIT`.
A successful capture and replay in a recorded image does not establish a pass
for a newer checkout or CI image; the weekly accuracy job validates that
environment. Attach the capture PNG images and JSON, independent replay logs,
exit codes, JUnit results and source/image provenance to the PR so reviewers can
verify the evidence rather than infer it from metadata.

## Capture a platform candidate

Use a real H100 or MI300 and the corresponding recorded CI image, with the
repository's test dependencies installed. Run from the repository root. Set
`SENSENOVA_U1_CI_IMAGE` to the actual capture image reference, preferably
including its immutable digest. Confirm that the checkout and environment
changes are intentional before capturing. Choose exactly one selector:

```bash
# H100
export SENSENOVA_U1_MARKER='slow and diffusion and cuda and H100 and cards_1'
export SENSENOVA_U1_REFERENCE_FILE=tests/e2e/accuracy/sensenova_u1_reference.json

# MI300
# export SENSENOVA_U1_MARKER='slow and diffusion and rocm and MI300 and cards_1'
# export SENSENOVA_U1_REFERENCE_FILE=tests/e2e/accuracy/sensenova_u1_reference_mi300.json
```

```bash
: "${SENSENOVA_U1_CI_IMAGE:?Set the actual capture CI image reference}"
export SENSENOVA_U1_CI_IMAGE
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SENSENOVA_U1_ARTIFACT_DIR="$(mktemp -d /tmp/sensenova-u1-candidate.XXXXXX)"
export SENSENOVA_U1_CANDIDATE_DIR="$SENSENOVA_U1_ARTIFACT_DIR"

for mode in text2img img2img; do
  capture_status=0
  pytest -sv \
    "tests/e2e/offline_inference/test_sensenova_u1_${mode}_expansion.py::test_sensenova_u1_${mode}_pixel_accuracy" \
    -m "$SENSENOVA_U1_MARKER" \
    --run-level full_model \
    --junitxml="$SENSENOVA_U1_ARTIFACT_DIR/$mode.xml" || capture_status=$?
  printf '%s capture comparison exit: %s\n' "$mode" "$capture_status"
done
```

Each invocation is a separate process. The tests save `t2i.png`, `i2i.png` and
their JSON records before comparing pixels. A missing or mismatched reference
therefore fails while preserving these artifacts. A startup or generation
failure may not produce a record; resolve it before proceeding. Capturing
artifacts does not establish that the new reference is correct.

Inspect both PNG images and the JSON input contracts. Verify the full prompts, seed,
think mode, step count, output dimensions, input-image hash, model revision and
ten sampling positions. Review the recorded environment and source provenance
and explain why the reference should change. Do not replace a reference merely
to make a failing comparison pass.

After this review, explicitly assemble the candidate from the two records:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

candidate_dir = Path(os.environ["SENSENOVA_U1_CANDIDATE_DIR"])
cases = {
    case: json.loads((candidate_dir / f"{case}.json").read_text())
    for case in ("t2i", "i2i")
}
for case, record in cases.items():
    assert record["case"] == case
    assert record["environment"]["reference_hardware"] in {"H100", "MI300"}
    assert len(record["reference_pixels"]) == 10
candidate_path = candidate_dir / "sensenova_u1_reference.candidate.json"
with candidate_path.open("x") as handle:
    json.dump({"schema_version": 1, "cases": cases}, handle, indent=2)
    handle.write("\n")
print(candidate_path)
PY

cp "$SENSENOVA_U1_CANDIDATE_DIR/sensenova_u1_reference.candidate.json" \
  "$SENSENOVA_U1_REFERENCE_FILE"
```

The copy stages the candidate for validation; it does not accept the reference.
Keep the capture PNG images and JSON records available for review.

## Validate and accept

Run the normal tests twice, with each case in a new process and each round in a
separate artifact directory. Do not load a capture plugin or replace assertions.

```bash
verification_status=0
for round in 1 2; do
  export SENSENOVA_U1_ARTIFACT_DIR="$SENSENOVA_U1_CANDIDATE_DIR/verify-$round"
  mkdir -p "$SENSENOVA_U1_ARTIFACT_DIR"
  for mode in text2img img2img; do
    pytest -sv \
      "tests/e2e/offline_inference/test_sensenova_u1_${mode}_expansion.py::test_sensenova_u1_${mode}_pixel_accuracy" \
      -m "$SENSENOVA_U1_MARKER" \
      --run-level full_model \
      --junitxml="$SENSENOVA_U1_ARTIFACT_DIR/$mode.xml" || verification_status=1
  done
done
test "$verification_status" -eq 0
```

Accept the candidate only after all four comparisons pass and the reference
diff, images and metadata have been reviewed. Include the capture environment
and validation evidence in the change. If verification fails, preserve the
artifacts and investigate before accepting any update. Normal CI must remain
read-only with respect to the reference: never refresh it automatically, skip a
mismatch or widen the ±10 tolerance to accommodate a new output.
