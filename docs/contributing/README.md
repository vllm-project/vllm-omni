# Contributing to vLLM-Omni

Thank you for your interest in contributing to vLLM-Omni! This document provides guidelines and instructions for contributing.

!!! note
    vLLM-Omni hosts developer-facing meetings for Chinese- and English-language audiences. See [Community Meetings](../community/meetings.md) for current schedules, access details, agendas, and past notes.

## Getting Started

vLLM-Omni uses `uv` as the environment manager, to create and manage Python environments. Please follow the documentation to install `uv`. After installing `uv`, you can create a new Python environment using the following commands:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Development Environment for vLLM and vLLM-Omni

vLLM-Omni is quickly evolving, please see the [installation guide](../getting_started/installation/README.md) for details. It's recommended to build from source to provide the latest development environment.

!!! tip
    vLLM-Omni is compatible with Python versions 3.10 to 3.12. However, we recommend developing with Python 3.12 to minimize the chance of your local environment clashing with our CI environment.

### Adding a new model to vLLM-Omni

Please check [model implementation](model/README.md) for how to add diffusion and omni-modality models to vLLM-Omni.

### Linting

vLLM-Omni uses `pre-commit` to lint and format the codebase. See [pre-commit documentation](https://pre-commit.com/#usage) if `pre-commit` is new to you. Setting up `pre-commit` is as easy as:

```bash
uv pip install pre-commit
pre-commit install
```

vLLM-Omni's `pre-commit` hooks will now run automatically every time you commit.

!!! tip
    You can manually run the `pre-commit` hooks using:

    ```bash
    pre-commit run     # runs on staged files
    pre-commit run --show-diff-on-failure --color=always --all-files  # runs on all files (short for --all-files)
    ```

!!! warning
    GitHub Actions `pre-commit` **skips** several local gates so a whole-tree
    `--all-files` run stays green while historical debt is cleaned up. The
    current `SKIP` list in
    [`.github/workflows/pre-commit.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.github/workflows/pre-commit.yml)
    is `check-test-ci-coverage`, `markdownlint-cli2`, `shellcheck`,
    `check-spdx-header`, and `mypy-3.10`. Those hooks still run on **your
    commit** for changed files. A passing GitHub pre-commit check does not mean
    they passed locally. Hooks **not** on that list (forbidden imports,
    `torch.cuda`, TTS adapter ratchet, Buildkite schema, Ruff, typos, …) run in
    both places.

The hooks below are new relative to the previous Omni config (Ruff, typos,
actionlint, YAML/whitespace, DCO sign-off, and the pickle-only checker were
already there). `check-pickle-imports` is gone: pickle is now one rule inside
`check-forbidden-imports`.

| Hook id | Enforces | GitHub Actions |
| ------- | -------- | -------------- |
| `markdownlint-cli2` | Markdown in `docs/`, `recipes/`, `README.md`, `CONTRIBUTING.md` (not `.claude/` / `.cursor/` / `CLAUDE.md`) | skipped |
| `mypy-3.10` | Type-check changed `vllm_omni/` files (model trees excluded). `tests/` uses `--follow-imports skip` | skipped |
| `mypy-3.11` / `3.12` / `3.13` | Same checker, extra Python versions | not installed; `pre-commit run --hook-stage manual mypy-3.12` |
| `check-test-ci-coverage` | Every `tests/**/test_*.py` has a CI level mark and a hardware mark/helper | skipped |
| `check-tts-adapter-migration` | `self._tts_model_type` branches in `serving_speech.py` must not increase | runs |
| `shellcheck` | `*.sh` quoting / undefined vars | skipped |
| `check-spdx-header` | Omni SPDX header on `.py` / `.pyi` / `.sh` / `.rs` / `.proto` | skipped |
| `check-forbidden-imports` | pickle, stdlib `re`/`base64`, Triton, TileLang, Hugging Face Hub download/API | runs |
| `check-torch-cuda-call` | Direct `torch.cuda.*` helpers outside platform adapters | runs |
| `check-buildkite` | `.buildkite/*.yml` against the Buildkite schema | runs |

Related config (not a new hook id): Ruff `TID251` also bans `librosa` and several
`torch.cuda.*` names; typos is pinned and ignores git-describe hashes (`+g` +
hex) and NumPy `writeable`. Keep the `suggestion` hook last in
`.pre-commit-config.yaml` so autofix output does not bury the SKIP tip.

#### SPDX headers

Source files must start with:

```text
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
```

Rust and proto files use `//` instead of `#`. Empty files are exempt. A stale
`Copyright contributors to the vLLM project` line is **rewritten in place** to
the Omni copyright; the hook then exits non-zero so you restage the rewrite.

#### Forbidden imports

Under `vllm_omni/`, do not use:

- stdlib `re` → `import regex as re`
- stdlib `base64` → `import pybase64` (or `import pybase64 as base64`)
- `pickle` / `cloudpickle` (also banned in most tests)
- `from huggingface_hub import snapshot_download` / `hf_hub_download` / `HfApi`
  / similar → `vllm.transformers_utils.repo_utils`
- direct `triton` / `tilelang` → `vllm.triton_utils` / `vllm.tilelang_utils`

`examples/`, `tests/`, `benchmarks/`, `apps/`, `docs/`, `tools/`, `scripts/`,
`.buildkite/`, and `.github/` may keep stdlib `re` and `base64`
(`_NON_LIBRARY_DIRS`). Pickle is stricter: tests are not exempt unless the
file is on the pickle allowlist.

#### `torch.cuda` call sites

Prefer `current_omni_platform` / `OmniPlatform`. Platform adapters under
`vllm_omni/platforms/` are allowed; other files must not add new
`torch.cuda.*` helpers (`device_count`, `synchronize`, `empty_cache`,
`device()`, `manual_seed`, and similar). `import torch` is fine.

#### shellcheck

The hook does **not** download a binary. Install `shellcheck` with a signed
package manager, then re-run:

- Debian/Ubuntu/WSL: `sudo apt-get install shellcheck`
- Fedora: `sudo dnf install ShellCheck`
- macOS: `brew install shellcheck`
- Git Bash (MINGW): `scoop install shellcheck` so `shellcheck.exe` is on PATH.
  WSL is Linux: use `apt-get`, not a Windows `.exe`.

If no native binary is found, the wrapper prints those commands (and
[the upstream install page](https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing))
and exits 1. AMD and other vendor scripts are **not** excluded.

#### mypy

`mypy-3.10` runs on commit for changed files. `[tool.mypy]` uses Python 3.12,
`ignore_missing_imports = true`, and `follow_imports = "silent"`. Wrapper
`tools/pre_commit/mypy.py` skips `vllm_omni/model_executor/models/` and
`vllm_omni/diffusion/models/`, and type-checks `tests/` with
`--follow-imports skip`. Extra versions:

```bash
pre-commit run --hook-stage manual mypy-3.12
```

#### Test CI marks

`check-test-ci-coverage` requires each collected test module under `tests/` to
have at least one CI **level** mark (`core_model`, `advanced_model`,
`full_model`, `local_model`, or `slow`) and a **hardware** mark (`cpu`, `cuda`,
`H100`, …) or helper (`hardware_test(` / `hardware_marks(`). GitHub Actions
skips this hook; local commit does not. See the
[test writing guide](./ci/test_writing_guide.md).

#### TTS adapter ratchet

`check-tts-adapter-migration` counts `self._tts_model_type` comparisons in
`vllm_omni/entrypoints/openai/serving_speech.py`. The count must stay at or
below `MAX_MODEL_TYPE_BRANCHES` in
[`tools/pre_commit/check_tts_adapter.py`](https://github.com/vllm-project/vllm-omni/blob/main/tools/pre_commit/check_tts_adapter.py).
New per-model behavior belongs in
`vllm_omni/entrypoints/openai/tts_adapters/`. Removing branches **must** lower
the budget in the same change; raising it is a reviewable policy edit.

#### Markdownlint and Buildkite

`markdownlint-cli2` auto-fixes `docs/`, `recipes/`, and the root README /
CONTRIBUTING (rules in `.markdownlint.yaml`). Skill files under `.claude/` and
`.cursor/` are excluded.

`check-buildkite` validates `.buildkite/*.yml` against the official schema
after the same expansion `upload_pipeline.py` uses. Omni-only keys such as
`mirror_hardwares` are stripped or rendered first. A few files are skipped in
`SKIP_FILES` inside `tools/pre_commit/check_buildkite.py`.

#### Extending allowlists and budgets

Prefer fixing the call site. Growing an allowlist or raising a budget is a
policy change and needs review — do not edit these just to make the hook pass.

| Gate | File | What to edit |
| ---- | ---- | ------------ |
| pickle, `re`, `base64`, Hugging Face Hub, Triton/TileLang | [`tools/pre_commit/check_forbidden_imports.py`](https://github.com/vllm-project/vllm-omni/blob/main/tools/pre_commit/check_forbidden_imports.py) | `CHECK_IMPORTS[<rule>].allowed_files`. Use `allowed_dirs` only when a whole tree should stay exempt. |
| `torch.cuda` | [`tools/pre_commit/check_torch_cuda.py`](https://github.com/vllm-project/vllm-omni/blob/main/tools/pre_commit/check_torch_cuda.py) | `ALLOWED_FILES`, or `ALLOWED_PREFIXES` for a platform adapter tree. |
| TTS `_tts_model_type` branches | [`tools/pre_commit/check_tts_adapter.py`](https://github.com/vllm-project/vllm-omni/blob/main/tools/pre_commit/check_tts_adapter.py) | `MAX_MODEL_TYPE_BRANCHES` (down only, unless the PR justifies a raise). |
| Buildkite skip list | [`tools/pre_commit/check_buildkite.py`](https://github.com/vllm-project/vllm-omni/blob/main/tools/pre_commit/check_buildkite.py) | `SKIP_FILES` for pipelines that cannot be expanded. |

To skip one hook for a single commit (discouraged): `SKIP=<hook-id> git commit`.
To bypass every hook: `git commit --no-verify` (also discouraged).

### Documentation

MkDocs is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file, `mkdocs.yml`.

Get started with:

```bash
uv pip install -e ".[docs]"
```

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it. From the root of the repository, run:

```bash
mkdocs serve                           # with API ref (~10 minutes)
API_AUTONAV_EXCLUDE=vllm_omni mkdocs serve  # API ref off (~15 seconds)
```

Once you see `Serving on http://127.0.0.1:8000/` in the logs, the live preview is ready! Open <http://127.0.0.1:8000/> in your browser to see it.

For additional features and advanced configurations, refer to the:

- [MkDocs documentation](https://www.mkdocs.org/)
- [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/) (the MkDocs theme we use)

### Testing

vLLM-Omni uses `pytest` to test the codebase.
Please refer to the [test instructions](./ci/test_execution_guide.md) for detailed testing information.

!!! warning
    Currently, not all unit tests pass when run on CPU platforms. If you don't have access to a GPU platform to run unit tests locally, rely on the continuous integration system to run the tests for now.

### Using repository skills with coding agents

vLLM-Omni maintains [repository-scale skills](https://github.com/vllm-project/vllm-omni/tree/main/.claude/skills) for common coding, testing, performance, and review workflows. These skills capture repository-specific structure, validation requirements, and evidence standards that a general coding agent may not know.

If your coding agent discovers repository skills automatically, ask it to use the relevant skill by name. Otherwise, point it to the linked `SKILL.md` and ask it to read the file before changing code. Skills can be combined: use a domain skill for the implementation, `vllm-omni-test` for test and CI decisions, and `precheck-pr` before opening the pull request.

| Task | Repository skill |
| --- | --- |
| Add or extend a diffusion model | [`add-diffusion-model`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/add-diffusion-model/SKILL.md) |
| Add or extend a text-to-speech model | [`add-tts-model`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/add-tts-model/SKILL.md) |
| Add, select, or debug quantization | [`quantization`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/quantization/SKILL.md) |
| Diagnose or optimize diffusion performance | [`diffusion-perf-opt`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/diffusion-perf-opt/SKILL.md) |
| Upgrade an NPU model runner | [`vllm-omni-npu-model-runner-upgrade`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/vllm-omni-npu-upgrade/SKILL.md) |
| Add a regression test, choose L1-L4 coverage, or wire Buildkite | [`vllm-omni-test`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/vllm-omni-test/SKILL.md) |
| Self-check a branch before opening a pull request | [`precheck-pr`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/precheck-pr/SKILL.md) |
| Perform a maintainer or reviewer review of an existing pull request or local branch | [`review-pr`](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/review-pr/SKILL.md) |

Use the following workflow when collaborating with a coding agent:

1. **Ground the task.** Provide the issue, RFC, or design document that defines the expected behavior. Ask the agent to state its assumptions and identify the affected module or feature contracts before editing code.
2. **Select the narrowest applicable skills.** Start with the domain skill that owns the change. Add `vllm-omni-test` whenever production behavior or tests change. For a bug fix, ask for the smallest reproducer and a regression test tied to the original symptom.
3. **Require executable evidence.** Ask the agent to run the narrowest relevant checks and report the exact commands and results. It must identify tests it could not run because of unavailable hardware, model weights, credentials, or dependencies; an unrun check is not a pass.
4. **Inspect the resulting diff.** Confirm that the implementation stays within the issue or RFC, does not overwrite unrelated work, and includes the required tests, documentation, and benchmark or accuracy evidence.
5. **Run the pre-submit workflow.** Use `precheck-pr`, then run the applicable local tests and pre-commit hooks before opening the pull request.

For example:

```text
Use add-diffusion-model and vllm-omni-test to implement the linked issue.
Read the relevant design documents first, keep the change within the issue,
and report every test command, result, and hardware validation gap.
```

```text
Use vllm-omni-test to turn this bug reproducer into the smallest deterministic
regression test. Prefer L1 CPU coverage; if the failure requires real weights
or serving, explain the required L2/L3 environment and provide the exact command.
```

Repository skills guide the agent, but they do not replace contributor judgment, the accepted issue or RFC, required test evidence, or maintainer review. Review all generated changes before submitting them.

## Issues

If you encounter a bug or have a feature request, please search existing issues first to see if it has already been reported. If not, please file a new issue, providing as much relevant information as possible.

!!! important
    Do not report suspected security vulnerabilities through a public issue, pull request, discussion, or Slack channel. Follow the [security disclosure instructions](../community/contact_us.md#security-disclosures) to arrange a private report.

## Pull Requests & Code Reviews

Thank you for your contribution to vLLM-Omni! Before submitting the pull request, please ensure the PR meets the following criteria. This helps vLLM-Omni maintain the code quality and improve the efficiency of the review process.

### DCO and Signed-off-by

When contributing changes to this project, you must agree to the [DCO](https://developercertificate.org/). Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

!!! tip
    You can enable automatic sign-off via your IDE:

    - **PyCharm**: Click on the `Show Commit Options` icon to the right of the `Commit and Push...` button in the `Commit` window. It will bring up a `git` window where you can modify the `Author` and enable `Sign-off commit`.
    - **VSCode**: Open the Settings editor and enable the `Git: Always Sign Off` (`git.alwaysSignOff`) field.

### PR Title and Classification

Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

- `[Bugfix]` for bug fixes.
- `[CI/Build]` for build or continuous integration improvements.
- `[Doc]` for documentation fixes and improvements.
- `[Model]` for adding a new model or improving an existing model. Model name should appear in the title.
- `[Frontend]` For changes on the vLLM-Omni frontend (e.g., OpenAI API server, `Omni`/`AsyncOmni`, etc.)
- `[Kernel]` for changes affecting CUDA kernels or other compute kernels.
- `[Core]` for changes in the core vLLM-Omni logic (e.g., `OmniProcessor`, `OmniARScheduler`, etc.)
- `[Hardware][Vendor]` for hardware-specific changes. Vendor name should appear in the prefix, such as [Ascend] for Ascend NPUs.
- `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

!!! note
    If the PR spans more than one category, please include all relevant prefixes.

### Pre-Check Before Submitting

Before submitting a PR, run the [precheck-pr skill](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/precheck-pr/SKILL.md) with the code agent for a self-review against project conventions:

The skill offers two modes:

- **Quick (~3 min):** catches showstoppers — PR title format, missing benchmark claims, rebase status
- **Full (~10 min):** thorough maintainer-grade review — dead code scan, copy-paste detection, import hygiene

The precheck covers five PR types: Bug Fix, Performance, New Model, Diffusion Model, and General. Each type has a tailored checklist that validates evidence quality (repro steps, A/B benchmarks, registry entries, etc.). See the [precheck-pr skill](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/precheck-pr/SKILL.md) for the full checklist.

### Local Test

Please run the L1 and L2 test cases locally first and attach the results before contacting us to add the "ready" label. Please refer to the [test instructions](./ci/test_execution_guide.md) for running the test cases.

### Automatic skip-ci (docs and pytest skip marks)

On pull requests and `main` pushes, the bootstrap step in [`.buildkite/cuda/pipeline.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/pipeline.yml) runs [`.buildkite/common/scripts/upload_pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py) against the git diff. When every changed file qualifies, **L2 (`ready`) and L3 (`merge-test`) pipelines are not uploaded**, so the default GPU CI jobs are skipped.

| Change per file | Examples |
| --- | --- |
| Documentation | `docs/**`, any `*.md`, `mkdocs.yml` |
| Pytest skip marks only (under `tests/`) | Add/remove/edit `@pytest.mark.skip`, `@pytest.mark.skipif`, or `pytest.skip(...)`; reformat `pytestmark` only to add a skip/skipif alongside existing `pytest.mark.*` entries |
| New skipped test module | New `tests/**/*.py` whose `pytestmark` includes unconditional `pytest.mark.skip` |

These PR shapes all trigger skip-ci:

- Documentation only
- Qualifying skip-mark edits in `tests/**/*.py` only
- **A mix of documentation and qualifying skip-mark test edits**

Skip-ci does **not** apply when the diff also touches product code (for example `vllm_omni/`), or when test files change assertions, imports, fixtures, or other non-skip logic. If the diff cannot be resolved (non-PR branches outside `main`), CI runs as usual.

!!! note
    Skipping L2/L3 does **not** disable the Docker image build step. Nightly (L4) upload can still run when the PR has a `nightly-test` label or on scheduled `main` builds with `NIGHTLY=1`. Weekly (L5) upload can still run with `weekly-test`, `WEEKLY=1`, or `NON_CRITICAL=1`. Bootstrap child steps live in `bootstrap-upload-steps.yml`; the hook entry step runs `upload_pipeline.py --upload <platform>/bootstrap-upload-steps.yml`, which injects `if` by step `key` from skip-ci before upload. See [CI Settings — Diff-aware CI](./ci/ci_settings.md#diff-aware-ci) and [Test System Overview](./ci/test_system_overview.md).

### Code Quality

The PR needs to meet the following code quality standards:

- We adhere to Google Python style guide and Google C++ style guide.
- Pass all linter checks.
- The code needs to be well-documented to ensure future contributors can easily understand the code.
- Include sufficient tests to ensure the project stays correct and robust. This includes both unit tests and integration tests.
- Please add documentation to `docs/` if the PR modifies the user-facing behaviors of vLLM-Omni. It helps vLLM-Omni users understand and utilize the new features or changes.

### Notes for Large Changes

Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with `rfc-required` and might not go through the PR.

### What to Expect for the Reviews

The goal of the vLLM-Omni team is to be a _transparent reviewing machine_. We would like to make the review process transparent and efficient and make sure no contributor feels confused or frustrated. However, the vLLM-Omni team is small, so we need to prioritize some PRs over others. Here is what you can expect from the review process:

- After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.
- After the PR is assigned, the reviewer will provide status updates every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM-Omni team.
- After the review, the reviewer will put an `action-required` label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.
- Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.

## Additional Resources

- [Design Documents](../design/index.md) - Architecture and design documentation

## Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM-Omni. All of your contributions help make vLLM-Omni a great tool and community for everyone!
