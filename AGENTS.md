# Agent guide for vLLM-Omni

Read [CONTRIBUTING.md](CONTRIBUTING.md) before contributing.

## Subsystem design

Before changing or reviewing a documented subsystem, find the owning module
and any affected feature pages in the [design index](docs/design/index.md) at
the revision in scope. Trace the affected entry point through the code selected
by the relevant configuration and platform. Verify the behavior against the
source and tests linked from those pages.

Use the same design pages when preparing or reviewing a PR. When the change
alters an active contract, update its design page in the same PR. In the PR
description, explain the before-and-after behavior. Include the validation
command, result, revision, and backend where applicable. See the
[design contract rules](.claude/skills/review-pr/references/process/design-contracts.md)
and the [worked PR handoff](docs/contributing/ci/pr_merge_loop.md#worked-pr-handoff).

## Code standards

Follow the project's [code quality standards](docs/contributing/README.md#code-quality)
and [local linting instructions](docs/contributing/README.md#linting) when
writing or reviewing code.

## PRs and issue triage

Use [precheck-pr](.claude/skills/precheck-pr/SKILL.md) for author self-checks
and [review-pr](.claude/skills/review-pr/SKILL.md) for independent reviews.
See the [PR merge loop](docs/contributing/ci/pr_merge_loop.md) for maintainer
triage, CI admission, and merge decisions.

## Lessons

Before code changes, issue triage or PR reviews, read relevant
[review and issue lessons](docs/contributing/ci/lessons.md) and verify their
evidence against the current revision.
