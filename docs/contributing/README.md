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
    Skipping L2/L3 does **not** disable the Docker image build step. Nightly (L4) upload can still run when the PR has a `nightly-test` label or on scheduled `main` builds with `NIGHTLY=1`. Bootstrap child steps live in `bootstrap-upload-steps.yml`; the hook entry step runs `upload_pipeline.py --upload <platform>/bootstrap-upload-steps.yml`, which injects `if` by step `key` from skip-ci before upload. See [CI Settings — Diff-aware CI](./ci/ci_settings.md#diff-aware-ci) and [Test System Overview](./ci/test_system_overview.md).

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
