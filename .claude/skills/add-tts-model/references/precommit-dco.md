# Pre-commit and DCO

Every commit must pass `pre-commit` lint and carry a `Signed-off-by` line
that matches the commit author email.

## Pre-commit

Install hooks once:

```bash
pre-commit install
```

Run before every push on the files you changed:

```bash
pre-commit run --files \
  vllm_omni/model_executor/models/<model_name>/*.py \
  vllm_omni/entrypoints/openai/serving_speech.py \
  vllm_omni/model_executor/models/registry.py \
  tests/e2e/offline_inference/test_<model_name>.py \
  tests/e2e/online_serving/test_<model_name>.py
```

When pre-commit **modifies files** (ruff format auto-fix), it exits non-zero
but the changes are correct — stage the modified files and re-commit.

| Failure | Root cause | Fix |
|---------|-----------|-----|
| `ruff F841` | Variable extracted but never forwarded to model call | Remove the extraction or wire it through |
| `ruff E402` | Import added below function definitions | Move to top-level import block |
| `ruff format` | Line length, spacing, quote style | Accept auto-fix, stage, re-commit |

## DCO sign-off

Every commit must carry `Signed-off-by: Your Name <your@email.com>`. Use
`-s`:

```bash
git commit -s -m "feat(<model>): add <Model> TTS support"
```

Or set it permanently: `git config format.signOff true`.

The DCO check verifies that the commit author email matches the
`Signed-off-by` line. Confirm `git config user.email` matches your GitHub
account email before committing.

Fix a missing or mismatched sign-off on the latest commit:

```bash
git commit --amend -s --no-edit
git push origin <branch> --force-with-lease
```
