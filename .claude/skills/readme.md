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
- [`add-omni-model`](add-omni-model/SKILL.md): integrates a new multi-stage
  thinker/talker/code2wav omni model, from the HF reference through pipeline,
  stage hand-off processors, deploy YAML, parity ladder, tests, and recipe
- [`add-recipe`](add-recipe/SKILL.md): creates or updates in-repository model
  recipes with verified task/input/output contracts, concrete hardware,
  shared feature-guide links, exact commands, and qualification evidence
- [`add-tts-model`](add-tts-model/SKILL.md): covers integration of new TTS
  models and related serving workflows
- [`diffusion-perf-opt`](diffusion-perf-opt/SKILL.md): guides diffusion model
  performance optimization, including profiling traces, parallel strategies,
  stage timing analysis, and benchmark-driven tuning
- [`diffusion-accuracy-test`](diffusion-accuracy-test/SKILL.md): adds or extends
  image/video numerical accuracy tests with aligned references, scoped
  tolerances, and explicit Buildkite Accuracy coverage
- [`diffusion-perf-test`](diffusion-perf-test/SKILL.md): adds or extends
  image/video performance-benchmark cases and their nightly/weekly Buildkite
  Perf coverage, using the `run_benchmark.py` config schema and a fixed
  measurement contract
- [`find-simplifications`](find-simplifications/SKILL.md): finds evidence-backed
  opportunities to remove or merge dead, duplicated, speculative,
  over-generalized, or unnecessarily defensive vLLM-Omni code
- [`omni-perf-opt`](omni-perf-opt/SKILL.md): guides multi-stage omni serving
  optimization, including stage attribution, TTFT / AUDIO_TTFP / AUDIO_RTF
  tuning, async_chunk codec chunking, CUDA graphs, MRv2, replicas, and
  quality-gated A/B evidence
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
- [`vllm-omni`](vllm-omni/SKILL.md): entry skill for multi-stage omni models;
  covers deploy YAML, serving, debugging, benchmarking, and the code map, and
  routes model integration and performance work to the dedicated skills
- [`vllm-omni-npu-model-runner-upgrade`](vllm-omni-npu-upgrade/SKILL.md):
  upgrades NPU model runners to align with the latest vllm-ascend
  `NPUModelRunner`
- [`vllm-omni-test`](vllm-omni-test/SKILL.md): guides generation and execution
  of CI-aligned tests (L1–L4), pytest marker selection (`core_model` /
  `advanced_model` / `full_model`, `omni` / `tts` / `diffusion`), Buildkite
  wiring (`test-ready.yml`, `test-merge.yml`, `test-nightly.yml`,
  `test-weekly.yml`), and copy-paste local plus CI-like `pytest` commands; see
  `references/test-routing.md` for level-to-command mapping

## MiniMax H3 Creative Skills

The [portable H3 collection](../../.agents/skills/README.md) contains the prompt
writing skill and eight creative workflows for product ads, 3D shorts, paper art,
brand promos, music videos, co-op intros, and hand-drawn/live-action fusion.
Same-named directories here link to the canonical `.agents/skills/` copies, so
Claude Code and Codex use the same instructions and templates.

## Maintenance Guidelines

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
