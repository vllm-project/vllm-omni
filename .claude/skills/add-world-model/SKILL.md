---
name: add-world-model
description: Integrate causal or stateful diffusion world models into vLLM-Omni, including native checkpoint discovery, action or camera conditioning, AR-Diffusion state, and requested continuation, streaming, or audio/video delivery. Use for models similar to Hy-Nomas, LingBot-World 2.0, ABot-World, and TaoMate-H3. Use add-diffusion-model for stateless diffusion ports and production-add-diffusion-model for broader production qualification.
---

# Adding a World Model to vLLM-Omni

## Scope

Port the model's causal generation behavior into the existing Omni execution
path. Complete the capabilities requested by the issue or user, with evidence
for each advertised entry point. A basic integration does not automatically
include public realtime serving, audio transport, or every acceleration mode.

Use [add-diffusion-model](../add-diffusion-model/SKILL.md) for component,
registry, loader, and parallelism patterns. Apply its native-port contract:
the reference implementation is a pinned correctness oracle; importing it at
runtime is an external adapter and requires an explicitly permitted scope.
The Hy-Nomas adapter case documents that choice, rather than making it the
default for subsequent ports.

Read references when their workflow applies:

| Reference | Read when |
| --- | --- |
| [Repository map](references/repository-map.md) | Locating current discovery, model, runner, output, and test interfaces. |
| [State and output contracts](references/state-and-output.md) | Implementing causal state, continuation, interaction, or incremental decoding. |
| [Validation](references/validation.md) | Designing reference comparisons, lifecycle tests, long rollouts, or measurements. |
| [Integration cases](references/integration-cases.md) | Selecting a prior integration or checking the evidence behind a specific lesson. |

Use [vllm-omni-test](../vllm-omni-test/SKILL.md) for test levels and CI,
[diffusion-perf-opt](../diffusion-perf-opt/SKILL.md) for profiling and tuning,
[add-recipe](../add-recipe/SKILL.md) for deployment instructions, and
[precheck-pr](../precheck-pr/SKILL.md) before opening a PR.
Production API, soak, and feature-combination qualification belongs to
[production-add-diffusion-model](../production-add-diffusion-model/SKILL.md).

## Prerequisites

Read the target checkout's contribution and model-integration documentation.
Record the Omni base/head, official source commit, checkpoint and adapter
revisions, actual dependency imports, and available hardware. Inspect the
released configuration and weight index before choosing a similar backbone.

Create a bounded capability table with the requested entry points: offline,
`/v1/videos`, typed continuation, one-request streaming, realtime interaction,
and audio delivery. Record the engine, execution flags, output representation,
and validation status for each relevant row. Use `validated`, `limited`,
`unsupported`, and `not tested` with the evidence meanings in the production
skill; an untested path starts as `not tested`.

## Step 1: Freeze the causal generation contract

Trace the official entry point through:

- Prompt/image/audio preparation, action units, camera coordinates, resize,
  pose interpolation, and conditioning layout.
- Pixel and latent geometry, first/subsequent block lengths, and the meaning
  of chunk, segment, window, or phase for this model.
- Timestep/sigma lookup, DMD/re-noising, clean-context forwards, CFG, and
  precision at each conversion.
- KV retention, condition-encoder and decoder history, RNG, RoPE positions,
  adapter selection, and audio resets.

Compare native, Diffusers, or modular releases by their actual variant,
precision, components, scheduler, decoder, and dependencies. Matching weight
names establishes backbone compatibility only. The exit condition is an
explicit component/state/dataflow contract and a pinned reference workload.

## Step 2: Integrate discovery and strict loading

Reuse the target revision's index resolver and registry. For native layouts,
follow its existing lightweight native-discovery path and preserve explicit
pipeline/config precedence. Keep discovery free of model-runtime imports.

| Concern | Owner |
| --- | --- |
| Component configuration, checkpoint layout, remapping, reference constraints | Model package and loader. |
| Architecture aliases and capability discovery | Existing registry and lightweight metadata. |
| Public fields, sampling defaults, duration, and controls | Model extras and preprocessing. |
| Session scheduling, KV storage, cancellation, and retirement | Shared AR-Diffusion runtime. |
| Launch profiles and qualified usage | Deploy configuration, recipe, and indexes. |

Construct components from the released configuration. Reuse component weight
sources, shard iteration, `WeightsMapper`, and existing parameter loaders.
Track the actual loaded set, including all source shards of fused parameters;
reject unexpected, duplicate, missing, or incompatible required weights.
Audit camera/action/adapter parameters separately when a training loader
permits them to remain randomly initialized.

Test remapping with small real weight files, inspect the released index, and
check every constructed parameter's device and dtype. Constructor arguments
alone do not prove that nested modules received them.

For an explicitly permitted external adapter, pin its runtime dependency and
retain Omni component ownership, scheduling, and output handling. If it mutates
process-global SDPA or distributed state, validate an existing isolated-worker
configuration before imports and expensive loading. Context save/restore does
not provide concurrent process isolation.

Exit with working registry discovery and strict component loading. A shared
change needs a framework-level failing contract and regression coverage for
other consumers; a checkpoint-specific workaround stays model-local.

## Step 3: Preserve model math and attach AR state

Establish unoptimized reference correctness first. Match preprocessing,
noise generator, draw order/shapes, dtype, latent scaling, timestep/sigma,
conditioning, and decode. Capture intermediate tensors at the first mismatch;
keep a modified-noise reference run separate from the unmodified oracle.

Reuse the current AR capability, runner, allocator, session, and interaction
interfaces. Apply [state and output contracts](references/state-and-output.md)
to define ownership, pending/committed state, reset, close, eviction, failure,
and abort. Share the model math between request and stepwise execution.

Trace every accepted public parameter into computation. Cover shared video
fields as well as model-specific controls, and validate unsupported combinations
before loading or submission. Enforce a mode required for correct output;
documenting a known quality failure is insufficient execution validation.

Exit with a correct short causal rollout and failure cleanup. Add long-rollout
checks that cross the actual state boundaries in the next step.

## Step 4: Wire requested entry points and features

Reuse shared category examples, typed request protocols, deploy schemas, and
existing output transport. Keep checkpoint names and model algorithms out of
shared runners. For every requested mode, verify its actual engine,
`step_execution`, `streaming_output`, and returned payload.

Keep per-session VAE temporal state and honor stable-frame/lookahead rules.
Check normal final flush and cancellation separately. If audio is requested,
verify samples, tracks, timestamps, buffering, and complete encoded output,
as described in [state and output contracts](references/state-and-output.md).

Add only requested or justified accelerations. Reuse worker process groups,
rank-local KV geometry, shared device APIs, and component offload discovery.
Qualify each TP/SP/CFG/offload combination against its working baseline.
Treat approximate decoders, quantization, Cache-DiT, compile, and CUDA Graph
as separate quality/performance paths. For teacher/student models, reuse the
shared LoRA manager and verify adapter layout, alpha/r scaling, graph addresses,
weight-bank switching, and exception restoration.

Exit with evidence for the requested entry points, including long-rollout
boundaries, cancellation, and successful generation after release or failure.

## Step 5: Validate and prepare the contribution

Select the smallest relevant matrix from [validation](references/validation.md):
CPU contracts and real small-file loading, independent official parity,
offline/serving parity, same-latent decoder parity, and lifecycle recovery.
Inspect full temporal output; finite tensors and correct shapes are smoke
evidence rather than quality qualification.

Follow current task authorization and checkout rules for device experiments.
Complete available CPU/static checks and prepare exact commands, assets,
output locations, and thresholds when a device run cannot proceed.
Preserve the concrete blocker and mark the corresponding row `not tested`.

Run changed-file pre-commit checks. Update the recipe, its index, supported
models, and relevant feature guides. Keep raw experiments outside the source
tree according to the project's artifact policy. Write PR evidence against
the tested revision, including numerical metrics, generated clips, per-GPU
memory, timing boundaries, repetitions, and baseline variation. Revalidate
affected paths after a fix; keep earlier-revision results explicitly scoped.

## Definition of Done

- Discovery, strict loading, request defaults, and required controls work
  through the requested Omni entry points.
- Causal state has explicit owners, commit boundaries, memory bounds, and
  release behavior, with long-boundary and recovery evidence where applicable.
- Independent reference quality, internal serving consistency, decoder quality,
  and performance are reported as separate results.
- Tests use repository helpers and markers; docs and launch commands match
  the final implementation and list unverified capabilities.
- [precheck-pr](../precheck-pr/SKILL.md) and applicable local checks are complete
  before PR creation. Publishing or updating external services follows the
  current task's authorization.
