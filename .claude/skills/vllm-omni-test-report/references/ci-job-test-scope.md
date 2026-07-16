# CI job test scope (Scheduled nightly)

This reference holds **Scope / intent** text keyed by **Buildkite job name**. Use it when maintaining **optional** **Test content (job scope)** sections (e.g. hand-authored reports or `scripts/patch_report_scope_local.py` on Markdown); **`compose_full_report.py` does not emit that section.** If a new job appears in the pipeline, add a row here so patches stay accurate.

Steps that only upload artifacts—such as `Upload * Pipeline`—are not listed in nightly **reportable** job tables (same rule as the report Summary).

## Diffusion / video generation

| Typical job name | Scope / intent |
|------------------|----------------|
| **Diffusion Model Test with H100** | Diffusion pytest on **H100** (multimodal diffusion inference, configuration, and regressions; see `tests/` paths in the log). |
| **Diffusion Model Test with L4** | Diffusion pytest on **L4** (different hardware / VRAM tier vs the H100 row). |
| **Diffusion Model Wan22 completed Test with H100** | **Wan 2.2** diffusion / video flows (completed path on **H100**). |
| **Qwen-Image Diffusion Perf Test with H100** | **Qwen-Image** diffusion **performance** and related regressions on **H100**. |

## Omni multimodal

| Typical job name | Scope / intent |
|------------------|----------------|
| **Omni Model Test** | **Qwen-Omni** end-to-end capability (multimodal inference, API/engine integration). |
| **Omni Model Test with H100** | Same as above, pinned to **H100**. |
| **Omni Model Perf Test & Test Case Statistics** | Omni **performance**, test-case statistics, and related regressions (perf / stats–oriented). |

## Speech / TTS

| Typical job name | Scope / intent |
|------------------|----------------|
| **Qwen3-TTS Non-Async-Chunk E2E Test** | **Qwen3-TTS** **end-to-end** non–async chunk scenarios (service/inference path and audio output regressions). |

## Benchmarks & docs

| Typical job name | Scope / intent |
|------------------|----------------|
| **GEBench Accuracy Test with H100** | **GEBench** image-generation **accuracy/quality** benchmark (**H100**). |
| **GEdit-Bench Accuracy Test with H100** | **GEdit-Bench** editing **accuracy** benchmark (**H100**). |
| **Documentation Example Code Test with H100** | Runnable **documentation examples** (examples tied to docs, executed on **H100**). |

## Unit tests & coverage (Metrics **ut**)

| Source | Scope / intent |
|--------|----------------|
| **Simple Unit Test** (step on the **latest non–Scheduled nightly** build on [`main`](https://buildkite.com/vllm/vllm-omni/builds?branch=main)) | Repo **unit / fast pytest** and **coverage** (e.g. **TOTAL** row). **Metrics overview** rows **ut** / **ut (exclude models)** come from `buildkite_build_stats.py` parsing that step’s log (independent of the Metrics date window; see `references/buildkite-api.md`). |

## Maintenance

- **Release HTML / Markdown from `compose_full_report.py`:** does **not** include **Test content (job scope)**. To refresh that block in an existing `.md` report, use `scripts/patch_report_scope_local.py` (Buildkite + this file). Keep **first column** job labels aligned with Buildkite **exact** `job.name` strings.
- Job names change with `.buildkite` / pipeline YAML; add or rename rows here when labels change.
- For failure details, rely on each step’s **raw log** and the **Per-job test execution (pytest)** table.
