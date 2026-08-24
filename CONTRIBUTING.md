# Contributing to vLLM-Omni

You may find information about contributing to vLLM-Omni on [Contributing](https://vllm-omni.readthedocs.io/en/latest/contributing/).

Local `pre-commit` gates (SPDX, forbidden imports, `torch.cuda`, shellcheck,
mypy, test marks, TTS adapter ratchet, markdownlint, Buildkite schema, and
how to extend allowlists) are documented in the
[Linting](https://vllm-omni.readthedocs.io/en/latest/contributing/#linting)
section of that guide.

Before submitting a PR, run the [precheck-pr skill](https://github.com/vllm-project/vllm-omni/blob/main/.claude/skills/precheck-pr/SKILL.md) with the code agent for a self-check against project conventions.
