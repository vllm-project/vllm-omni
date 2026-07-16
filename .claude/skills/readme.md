# Claude Skills for vLLM-Omni

This directory contains Claude Code skills maintained for the `vllm-omni`
repository. These skills capture repeatable workflows for common contributor
tasks such as model integration, pull request review, and release note
generation.

## Directory Structure

Each skill lives in its own directory under `.claude/skills/`. A skill may
include:

- `SKILL.md`: the main workflow and operating instructions
- `references/`: focused reference material used by the skill
- `scripts/`: small helper scripts used by the skill

## Available Skills

- `add-diffusion-model`: guides integration of a new diffusion model into
  `vllm-omni`
- `diffusion-perf-opt`: guides diffusion model performance optimization,
  including profiling traces, parallel strategies, stage timing analysis, and
  benchmark-driven tuning
- `quantization`: guides quantization method selection, model integration,
  checkpoint loading, and quality/performance validation for vLLM-Omni
- `add-omni-model`: covers addition of new omni-modality model support
- `add-tts-model`: covers integration of new TTS models and related serving
  workflows
- `generate-release-note`: helps prepare release notes for repository changes
- `precheck-pr`: self-check a branch before creating a PR — validates PR title
  format, catches dead code, verifies accuracy/perf claims, and confirms merge
  readiness
- `vllm-omni-create-testcase`: guides generation and execution of CI-aligned tests (L1–L4),
  pytest marker selection (`core_model` / `advanced_model` / `full_model`,
  `omni` / `tts` / `diffusion`), Buildkite wiring (`test-ready.yml`,
  `test-merge.yml`, `test-nightly.yml`, `test-weekly.yml`), and copy-paste
  local plus CI-like `pytest` commands; see `references/test-routing.md` for
  level-to-command mapping
- `vllm-omni-local-test`: runs nightly jobs on **H200** or **H800** cluster
  machines — confirm `REPO_ROOT`, `HF_HOME`, and `CUDA_VISIBLE_DEVICES` with the
  user, optional `git pull`, then `tools/nightly/run_nightly_jobs.sh` (full
  nightly, `--test-type local` / `--test-type stability` / `--test-type local,stability`,
  or `--label-substr`); sync logs per scope (`local` → `nightly_jobs_local_*` only;
  `stability` → `nightly_jobs_stability_*` only; `default` → `nightly_jobs_YYYYMMDD-*` only;
  `all` → local + stability + general nightly; auto-detected from the run's
  `--test-type` when the user started by running tests first) to `logs/nightly_jobs`
  on the laptop for reporting; see `references/nightly-local-h200.md` /
  `nightly-local-h800.md` and `nightly-local-log-fetch.md`
- `vllm-omni-test-report`: generates **HTML test reports** (Markdown only when
  explicitly requested) — **release** via `scripts/compose_full_report.py`
  (Buildkite metrics, test matrix, issue tracking) or **nightly** via
  `scripts/nightly_local_log_report.py` from synced `nightly_jobs` (logs from
  `vllm-omni-local-test`) plus optional Buildkite scheduled nightly; optional
  archive to `vllm-omni-kanban` via `push_report_to_kanban.py` / `push_kanban_report.py`
- `review-pr`: provides a structured workflow for reviewing pull requests
- `vllm-omni-npu-model-runner-upgrade`: upgrades NPU model runners to align with the
  latest vllm-ascend NPUModelRunner

## Maintenance Guidelines

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
