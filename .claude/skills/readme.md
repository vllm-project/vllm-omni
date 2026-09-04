# Repository Skills for vLLM-Omni

This directory contains repository-scale skills maintained for `vllm-omni`.
They capture repeatable workflows for common contributor and maintainer tasks
such as model integration, CI-aligned testing, performance optimization, and
pull request review.

## Directory Structure

Each skill lives in its own directory under `.claude/skills/`. A skill may
include:

- `SKILL.md`: the main workflow and operating instructions
- `references/`: focused reference material used by the skill
- `scripts/`: small helper scripts used by the skill

## Using the Skills

Coding agents that discover repository skills can invoke the relevant skill by
name. For other agents, point them to the skill's `SKILL.md` and ask them to
read it before changing code. Combine a domain skill with `vllm-omni-test` when
tests or CI coverage change, then use `precheck-pr` for the contributor's final
self-check.

## Available Skills

- [`add-diffusion-model`](add-diffusion-model/SKILL.md): guides integration of
  a new diffusion model into `vllm-omni`
- [`add-tts-model`](add-tts-model/SKILL.md): covers integration of new TTS
  models and related serving workflows
- [`diffusion-perf-opt`](diffusion-perf-opt/SKILL.md): guides diffusion model
  performance optimization, including profiling traces, parallel strategies,
  stage timing analysis, and benchmark-driven tuning
- [`find-simplifications`](find-simplifications/SKILL.md): finds evidence-backed
  opportunities to remove or merge dead, duplicated, speculative,
  over-generalized, or unnecessarily defensive vLLM-Omni code
- [`precheck-pr`](precheck-pr/SKILL.md): self-checks a branch before creating a
  PR by validating title format, dead code, simplification opportunities,
  accuracy and performance claims, and merge readiness
- [`production-add-diffusion-model`](production-add-diffusion-model/SKILL.md):
  takes a working Day-0 diffusion integration through API/limit parity,
  feature-combination evidence, quantization/offload, performance, production
  serving, hardware recipes, and CI readiness
- [`quantization`](quantization/SKILL.md): guides quantization method selection,
  model integration, checkpoint loading, and quality/performance validation
  for vLLM-Omni
- [`review-pr`](review-pr/SKILL.md): provides a frozen-snapshot, contract-aware
  workflow for maintainers and reviewers; contributors should use
  `precheck-pr` for self-review
- [`vllm-omni-npu-model-runner-upgrade`](vllm-omni-npu-upgrade/SKILL.md):
  upgrades NPU model runners to align with the latest vllm-ascend
  `NPUModelRunner`
- [`vllm-omni-test`](vllm-omni-test/SKILL.md): guides generation and execution
  of CI-aligned tests (L1–L4), pytest marker selection (`core_model` /
  `advanced_model` / `full_model`, `omni` / `tts` / `diffusion`), Buildkite
  wiring (`test-ready.yml`, `test-merge.yml`, `test-nightly.yml`,
  `test-weekly.yml`), and copy-paste local plus CI-like `pytest` commands; see
  `references/test-routing.md` for level-to-command mapping

## Maintenance Guidelines

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
