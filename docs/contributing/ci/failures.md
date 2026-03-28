# CI Failures

## Overview of CI Checks

When you open a PR against vLLM-Omni, several CI checks run automatically:

| Check | Platform | What it does |
| ----- | -------- | ------------ |
| **pre-commit** | GitHub Actions | Runs linting (Ruff), formatting, spell-checking (typos), and YAML validation. |
| **Build Wheel** | GitHub Actions | Builds Python wheels for Python 3.11 and 3.12 on Ubuntu. Skipped for docs-only or Markdown-only changes (controlled by `paths-ignore` in the workflow). |
| **DCO** | GitHub | Verifies every commit has a `Signed-off-by` line. |
| **docs/readthedocs.org:vllm-omni** | Read the Docs | Builds the MkDocs documentation site. |
| **buildkite/vllm-omni** | Buildkite | Runs GPU-based tests on NVIDIA CUDA hardware (L4, H100). |
| **buildkite/vllm-omni-amd** | Buildkite | Runs GPU-based tests on AMD ROCm hardware (MI325). |
| **buildkite/vllm-omni-intel** | Buildkite | Runs GPU-based tests on Intel XPU hardware (Intel Arc BMG). |

## Step 1: Identify the Failing Check

Click the **Details** link next to the failing check on your PR to open the build log. The most common failures fall into these categories:

### pre-commit failures

These are typically formatting or linting issues introduced by your PR. Fix them locally:

```bash
uv pip install pre-commit
pre-commit run --all-files
```

Then commit the fixes and push.

### DCO failures

Every commit must include a `Signed-off-by` line. If you forgot, amend your commits:

```bash
git commit --amend -s
git push --force-with-lease
```

For multiple commits, use an interactive rebase to add the sign-off to each one.

### Read the Docs failures

The documentation build uses MkDocs with `fail_on_warning: true`, so even a minor warning (not just errors) will cause the build to fail. To reproduce locally:

```bash
uv pip install -e ".[docs]"
mkdocs build --strict
```

Common causes include broken cross-references, invalid admonition syntax, or missing files referenced by `--8<--` includes.

### Buildkite failures

Buildkite runs GPU tests in Docker containers. These are the most complex checks and can fail for reasons unrelated to your PR (infrastructure issues, flaky tests, etc.). See the sections below for how to investigate.

## Step 2: Check if the Failure Is a Known Issue

Before spending time debugging, check whether the failure already exists on the `main` branch:

1. **Look at the Buildkite build log** — the test name and error message are usually enough to identify the issue.
2. **Check recent CI runs on `main`** — if the same test is failing there, the failure is not caused by your PR.
3. **Search existing issues** — look for open issues in the [vllm-omni issue tracker](https://github.com/vllm-project/vllm-omni/issues) with the test name or error message.

If the failure is already tracked, leave a comment on your PR noting that the failure is pre-existing and link the issue.

## Step 3: Investigate the Failure

If the failure appears to be new, investigate whether your changes caused it.

### Reading Buildkite Logs

1. Click **Details** next to the Buildkite check on your PR.
2. Find the failing step in the pipeline (e.g., "Diffusion Model Test", "Simple Unit Test").
3. Expand the step to see the full test output with the traceback.

### Running Tests Locally

For instructions on running tests locally (including specific test files, functions, and markers), see the [Running Tests](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/test_guide/#running-tests) section in the Test Guide.

## Step 4: Raise an Issue or Fix It

### If the failure is pre-existing (not caused by your PR)

1. **Raise a new issue** if one doesn't already exist, using the title format:
   `[CI Failure]: [job-name] - [test-path]`
2. Include the error message, relevant log excerpts, and the commit hash where the failure occurs (e.g., "Still failing on main as of commit `abc1234`").
3. Leave a comment on your PR linking to the issue and noting that the failure is unrelated to your changes.

### If the failure is caused by your PR

1. Fix the issue in your branch and push the update.
2. If the fix is non-trivial, consider adding a test to prevent regression.

## Common Failure Patterns

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| `ruff` or formatting errors | Code style violation | Run `pre-commit run --all-files` |
| `Signed-off-by` missing | DCO check | Amend commits with `git commit --amend -s` |
| MkDocs build warning | Broken docs reference | Run `mkdocs build --strict` locally |
| `OOM` or `CUDA out of memory` | Test exceeds GPU memory | Check if your changes increased memory usage; use `--vae_use_slicing` / `--vae_use_tiling` for diffusion tests |
| Import errors | Missing or changed dependency | Check `pyproject.toml` and make sure dependencies are correct |
| Timeout (step exceeded N minutes) | Test is too slow or hangs | Profile the test; check for infinite loops or deadlocks |
| `Agent lost` in Buildkite | Infrastructure issue (not your fault) | Re-trigger the build; comment on your PR |
