---
name: diffusion-accuracy-ci
description: Add or extend numerical accuracy tests and Buildkite CI coverage for existing vLLM-Omni diffusion image and video models, using aligned reference outputs and scoped tolerances. Use for native/reference parity or accuracy checks for acceleration features; generation smoke tests and performance benchmarking are separate workflows.
---

# Add diffusion accuracy CI

Add reproducible accuracy coverage for a working diffusion model or acceleration
feature, using existing helpers, reference runners, and Buildkite conventions.
Default to one representative real-weight comparison, including the reference
and candidate runs it needs. Expand the workload/hardware matrix only when the
request or an uncovered failure requires it. Tiny random weights do not prove
model accuracy.

## Workflow

1. **Choose extension or first accuracy case.** For an existing case, locate
   its reference, metrics/threshold evidence, and CI step; reuse them within
   their validated scope and inspect only the requested change and affected
   dependencies. Reopen reference research or calibration when that scope
   changes. For a first case, inspect the model's registry/pipeline, recipe,
   task/output contract, and native vs Diffusers backend, then choose a
   reference via [reference-contract.md](references/reference-contract.md).
   First accuracy coverage does not imply porting a new model.
2. **Preflight before model loading.** Use the
   [preflight entry](references/reference-contract.md#preflight-before-model-loading)
   in the actual reference environment, then check required weights, inputs,
   and comparison tools. Resolve failures before starting either model/server.
   Reuse a successful preflight only while its environment and assets match.
3. **Check effective comparison conditions.** Use the short
   [comparison checklist](references/reference-contract.md#comparison-checklist)
   for this case's relevant fields. Inspect how arguments reach the pipeline
   and what settings actually take effect; matching names alone are insufficient.
   For acceleration, keep unrelated settings fixed against a trusted baseline
   with the feature disabled. Online/offline agreement proves path consistency.
4. **Implement the modality case.** For image generation, editing, or layered
   outputs, read [image-accuracy.md](references/image-accuracy.md). For T2V/I2V
   and video with audio, read [video-accuracy.md](references/video-accuracy.md).
   Follow the [model directory rule](references/ci-wiring.md#model-directory-and-migration)
   for tests, reference runners, and model-specific support files. Reuse
   [accuracy helpers](../../../tests/e2e/accuracy/helpers.py) and
   [fixtures](../../../tests/e2e/accuracy/conftest.py). Add a shared helper only
   for a missing reusable operation. Generate the required reference/candidate
   artifacts within a self-contained test or explicit fixture dependency, without relying on
   another test running first or an old output directory.
5. **Recommend an evidenced gate.** Follow
   [reference-contract.md](references/reference-contract.md#calibrate-a-scoped-gate).
   Reuse existing thresholds only within their evidenced scope; calibrate a
   first case or changed numerical contract. Check output structure before
   metrics, and save both outputs and metric summaries before assertions.
   Deliver a threshold recommendation with observed variability and headroom;
   the user selects new or adjusted thresholds after reviewing that evidence.
6. **Wire execution into CI.** Read [ci-wiring.md](references/ci-wiring.md).
   Match hardware marks, real-checkpoint runtime, explicit file/node selection,
   source dependencies, tools/reference assets, and artifact retention in the
   appropriate image/video Accuracy step. Choose nightly or weekly based on
   the case's resource cost and intended coverage. A marker alone does not
   make Buildkite execute the test.
7. **Validate, diagnose, and stop.** Collect the intended nodes with the exact
   CI selector, then run the representative comparison on suitable hardware.
   On failure, follow the [diagnostic order](references/reference-contract.md#diagnose-before-rerunning)
   before rerunning. Use the [completion and rerun rules](references/ci-wiring.md#completion-and-rerun-scope)
   to finish once evidence is sufficient and to rerun only affected checks.
   Report passed, failed, skipped, deselected, and unvalidated separately.
   Without suitable hardware, deliver feasible edits and exact commands with
   GPU execution/calibration explicitly unvalidated; collection is not a pass.

Read [vllm-omni-test](../vllm-omni-test/SKILL.md) and its
[test routing reference](../vllm-omni-test/references/test-routing.md) when
fixture, marker, runtime, or CI mechanics need more detail. For this accuracy
workflow, use the specific Accuracy lane and actual files described here
rather than applying a functional `*_expansion.py` recipe automatically.

## Deliver when invoked for a model

- The minimal accuracy pytest case/reference adapter and CI edits requested.
- Reference provenance, the aligned request contract, and a
  [threshold recommendation with headroom evidence](references/reference-contract.md#threshold-recommendation-and-user-selection),
  including its calibration limits and user-selection status.
- Copy-paste local single-case and CI-equivalent commands, with hardware,
  checkpoint cache/access, reference environment, and media-tool prerequisites.
- Locations of reference/candidate outputs, metrics and logs, plus execution
  results and the precise scope of the accuracy claim.
