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
- `add-omni-model`: covers addition of new omni-modality model support
- `add-tts-model`: covers integration of new TTS models and related serving
  workflows
- `generate-release-note`: helps prepare release notes for repository changes
- `review-pr`: provides a structured workflow for reviewing pull requests

## Maintenance Guidelines

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
