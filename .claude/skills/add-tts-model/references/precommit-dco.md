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
| `check-spdx-header` | Missing header, or copyright still says `vLLM project` | Two-line Omni header (`vLLM-Omni project`); restage the rewrite |
| `check-forbidden-imports` | Stdlib `re`/`base64`, pickle, Hugging Face Hub API, or direct Triton/TileLang | `import regex as re` and `pybase64`; use `vllm.transformers_utils.repo_utils`. Do **not** add the file to `allowed_files` without review |
| `check-torch-cuda-call` | New `torch.cuda.*` call site | Use `current_omni_platform` / `OmniPlatform`; do not grow `ALLOWED_FILES` without review |
| `check-tts-adapter-migration` | New `self._tts_model_type` branch in `serving_speech.py` | Put per-model logic in `tts_adapters/`. Lower `MAX_MODEL_TYPE_BRANCHES` when removing branches; do not raise it without review |
| `check-test-ci-coverage` | New `tests/**/test_*.py` missing level or hardware mark | Add `core_model`/`advanced_model`/… plus `cpu`/`cuda`/`hardware_test(` |
| `mypy-3.10` | Type error on changed `vllm_omni/` files | Fix the types; model trees are excluded. Extra versions: `pre-commit run --hook-stage manual mypy-3.12` |
| `markdownlint-cli2` | Docs/README markdown lint | Accept auto-fix under `docs/` / `recipes/` / root README |
| `check-buildkite` | Invalid `.buildkite/*.yml` | Fix the YAML; do not grow `SKIP_FILES` without review |
| `shellcheck` | No `shellcheck` on PATH, or script warning | `apt-get`/`dnf`/`brew install shellcheck`, or `shellcheck.exe` on Windows PATH |

Canonical policy, CI `SKIP` list, and allowlist locations:
[docs/contributing/README.md](../../../docs/contributing/README.md#linting).

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
