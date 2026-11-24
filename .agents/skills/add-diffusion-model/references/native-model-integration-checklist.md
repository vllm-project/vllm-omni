# Native Diffusion Model Integration Checklist

Use this checklist for native non-Diffusers models, hybrid pipelines, and
model-specific optimization work.

## 1. Freeze the contract before coding

Record:

- Reference repository and immutable revision.
- Official checkpoint ID, local layout, license, and required auxiliary files.
- Pipeline stages: prompt/image encoding, denoising, video/image/audio decoding,
  and muxing.
- Token packing, RoPE coordinates, attention sinks/LSE behavior, MoE layout,
  scheduler equations, CFG formulation, dtypes, and checkpoint transforms.
- Output geometry, frame count/FPS, audio rate/channels, supported tasks, and
  default generation steps.
- Available local hardware versus the official qualification hardware.
- Intended default deployment and optional optimization paths.

State whether the deliverable is a native implementation or an external
adapter. For native support, use the reference as an oracle only.

## 2. Use phase gates

| Phase | Required exit evidence |
|---|---|
| Architecture | Component/weight inventory; unsupported reference dependencies identified; ownership decisions documented |
| Native correctness | Strict checkpoint load; tiny forward or one-step golden vs pinned reference; output contract validated |
| Framework integration | Registry/discovery, shared request types, existing category example, serving route, docs, and recipe work without model-name branching in shared scripts |
| Lossless distribution | One-rank oracle vs each supported TP/SP/CFG/HSDP/VAE layout; released-checkpoint smoke for the recommended layout |
| Memory optimization | Ordinary/layerwise/DLO behavior, peak HBM, full-tree host PSS, and stage timeline; shared backend unchanged unless a general contract is proven |
| Lossy optimization | Cache hit demonstrated; quality delta measured; speed claim made only when warmup permits hits |
| Full qualification | Default-resolution E2E at official step count for every supported task; media metadata and hashes recorded |
| Review readiness | Focused tests, Ruff/format, strict docs, DCO, current-main rebase, issue/RFC for large changes, concise PR evidence and limitations |

Do not start performance tuning before native one-step parity. Do not claim a
feature from topology validation alone when released weights can exercise it.

## 3. Keep ownership narrow

Default placement:

| Concern | Owner |
|---|---|
| Architecture, checkpoint remap, packed attention math, nested block alias, auxiliary lifecycle | Model package |
| Kernel/version dispatch already useful to multiple models | Shared attention utility |
| Request-dependent example defaults | Typed model-extra entry consumed through a registry |
| Exact-ID/local-file detection for repositories without standard metadata | Generic native-checkpoint signature table |
| General offload/cache/HSDP bug | Shared layer plus one focused shared regression |

Before changing shared code, answer:

1. Is this a framework contract or only this checkpoint's layout?
2. Can the model expose an existing declarative hook instead?
3. Is there a second consumer or a framework-level failing test?
4. Can the shared change avoid importing or naming the model?

If the answers do not justify a shared change, keep it model-local.

## 4. Reuse examples without leaking model logic

- Reuse the existing `x_to_y.py` category script.
- Resolve the pipeline class through standard metadata or native signature
  discovery, not path substrings.
- Put dynamic model defaults and validation in `model_extras`; let shared
  examples consume only the generic typed API.
- Do not create image/speech/default abstractions without a real consumer.
- Add a dedicated example only when the shared request/output protocol cannot
  express the model.

## 5. Build a bounded evidence matrix

Start with the recommended resident layout. Add only supported, useful points:

- Single request: TP, SP, or hybrid TP/SP.
- Throughput: DLO/DP layouts when concurrent requests are valid.
- CFG: compare CFG-parallel output with the packed-CFG oracle only if the model
  actually uses CFG.
- HSDP: record loading and materialization peaks; it can use more HBM than a
  resident SP layout.
- VAE patch parallelism: compare decoded tensors or media exactly where the
  algorithm should be lossless.
- Cache: separate cache integration from acceleration evidence.

With four devices, qualify four-device layouts such as DP4, DP2SP2, SP4, or
CFG2xSP2. Do not present CFG2xSP4 or an official eight-device runtime as locally
validated.

For each measured layout record:

- Cold E2E and warm user latency.
- Steady-state wave time and throughput per device.
- Prompt, denoise, video decode, audio decode, and mux times.
- Peak allocated/reserved HBM and full-process-tree host PSS.
- Step count, seed, output geometry, frame/FPS/audio metadata, and parity metric.

Plot latency versus throughput/device. Keep one-step, four-step, and full-step
results separate because they are not quality-equivalent.

## 6. Keep tests proportional

Prefer a small set with distinct failure ownership:

1. Strict weight-key/dtype/load contract.
2. Tiny native golden or one-step reference parity.
3. One parametrized distributed-parity test covering supported layouts.
4. Pipeline/request/output geometry and serving smoke.
5. Shared-example/default test when the category script changes.
6. One shared regression only for a genuine shared-layer fix.

Avoid duplicating the same shape or configuration assertion across
model-specific files. Keep expensive released-checkpoint evidence in the PR or
recipe unless the repository's CI tier explicitly requires it.

## 7. Write evidence, not marketing

In the PR description include:

- A compact architecture/data-flow explanation.
- Default deployment and why it is recommended.
- Code-change counts by component.
- Accuracy and distributed-parity evidence tied to commands/scripts.
- Separate Pareto figures for different step counts.
- Supported versus validated versus planned optimizations.
- Hardware gaps and untested official-runtime claims.

Call a result a bounded local qualification unless it includes repeated timing
and variance. State whether latency includes text/image encoding and all
decoders. Preserve the reference revision and exact commands needed to
reproduce the evidence.
