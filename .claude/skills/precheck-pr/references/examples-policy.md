# Examples Python Policy

Prevent new model-specific Python examples without forcing cleanup of the
existing backlog. Run this check on every PR in quick and full modes.

## Scope

Set the merge base as in the main workflow, then list Python paths introduced
by the PR, including additions, copies, and renames:

```bash
BASE="$(git merge-base HEAD origin/main)"
git diff --name-status --diff-filter=ACR --find-renames --find-copies \
  "${BASE}"...HEAD -- 'examples/**/*.py'
```

Inspect destination paths for copies and renames. Existing model-specific files
that are only modified or deleted are grandfathered and must not be reported.

## Blocking classification

Mark ✗ when a newly introduced Python file does any of the following:

- Names or scopes a path to one model, checkpoint, vendor, or model family.
- Hard-codes a model identifier, model-specific prompt, request shape, output
  conversion, or launch configuration behind an otherwise generic filename.
- Duplicates an existing task/protocol runner to demonstrate one model.
- Adds a per-model helper below a generic task hub such as `text_to_speech/`.

Do not infer safety from a generic filename alone. For example, a file named
`text_to_audio.py` is still model-specific if it only implements one model's
contract.

## Acceptable additions

A new Python example may pass when it is a genuinely model-neutral,
user-facing task or protocol entrypoint. It must accept model selection through
configuration and keep model-specific behavior out of the script.

Use ⚠ when the file is model-neutral but reusable logic appears misplaced under
`examples/`. Route code by purpose:

| Code purpose | Preferred location |
|---|---|
| Model prompt, defaults, request/output adaptation | `vllm_omni/model_extras/` or production model modules |
| Runnable model command and validation evidence | Task documentation or `recipes/` |
| UI or interactive application | `apps/` |
| Benchmark or evaluator | `benchmarks/` |
| Regression reproducer | `tests/` |
| Download, conversion, or setup utility | `tools/` |

If a model cannot use an existing shared runner, report the missing generic
capability. Do not solve the gap by adding a model-specific Python example.

## Report format

Add one row to the pre-check report:

```text
Examples policy    ✓ no new Python example paths
Examples policy    ⚠ generic helper belongs in tools/
Examples policy    ✗ examples/offline_inference/new_model/end2end.py is model-specific
Examples policy    ✗ examples/offline_inference/x_to_y_model_name.py is model-specific
```

For a blocker, name the path, the model-specific behavior, and the appropriate
shared runner or destination. Do not propose deleting unrelated existing
examples as part of the PR.
