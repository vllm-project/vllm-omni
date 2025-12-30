# Contributing to vLLM-Omni

Thank you for your interest in contributing to vLLM-Omni! This document provides guidelines and instructions for contributing.

!!! note
    We host weekly developer-facing online meetings to discuss milestones and updates **every Tuesday at 19:30 PDT**. Meeting link as well as the past meeting notes can be found [here](https://tinyurl.com/vllm-omni-meeting).

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

```bash
# Run all tests
pytest tests/

# Run tests for a single test file with detailed output
pytest -s -v tests/test_omni_llm.py
```

!!! warning
    Currently, not all unit tests pass when run on CPU platforms. If you don't have access to a GPU platform to run unit tests locally, rely on the continuous integration system to run the tests for now.

## Issues

If you encounter a bug or have a feature request, please search existing issues first to see if it has already been reported. If not, please file a new issue, providing as much relevant information as possible.

!!! important
    If you discover a security vulnerability, please report it by creating a GitHub issue with the `security` label.

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
- `[Frontend]` For changes on the vLLM-Omni frontend (e.g., OpenAI API server, `OmniLLM` class, etc.)
- `[Kernel]` for changes affecting CUDA kernels or other compute kernels.
- `[Core]` for changes in the core vLLM-Omni logic (e.g., `OmniProcessor`, `OmniARScheduler`, etc.)
- `[Hardware][Vendor]` for hardware-specific changes. Vendor name should appear in the prefix, such as [Ascend] for Ascend NPUs.
- `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

!!! note
    If the PR spans more than one category, please include all relevant prefixes.

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
