---
name: diffusion-perf-opt
description: Diagnose and optimize vLLM Omni diffusion workloads, especially Wan/Qwen/Flux-style image and video generation. Use when Codex is asked to analyze profiling traces, choose parallel strategies, inspect torch profiler trace.json or trace.json.gz timelines, estimate optimization ROI, investigate GPU idle/free bubbles, compare USP/CFG/HSDP/VAE parallelism, or design operator/host/quantization optimizations for vLLM Omni.
---

# vLLM Omni Performance Optimization

Use this skill to run a disciplined optimization loop for vLLM Omni diffusion workloads. Keep two ideas separate: real performance baselines are collected with low overhead, while torch profiler traces are diagnostic artifacts and may distort latency.

## First Questions

Before proposing changes, ask for the optimization scene if it is not already known:

- GPU model, card count, topology, and whether NVLink is present.
- Model and pipeline, for example Wan2.2 I2V A14B.
- User workload: resolution, frames, steps, batch/concurrency, CFG scales, prompt/image inputs.
- Three runnable commands for each model/workload, or scripts that generate them:
  - **Server startup command**: exact `vllm serve` command, environment variables, model path, port, parallelism flags, profiler flags, and precision/compile settings.
  - **User request command**: exact single-request client command (for example curl or Python client) with fixed prompt/media/seed/size/steps. Use this to validate correctness and collect per-request stage timings.
  - **User benchmark command**: exact repeatable benchmark command or script with warmup count, measured iteration count, concurrency/batch policy, output directory, and summary format.
- Current enabled strategies: USP/SP, CFG parallel, HSDP/FSDP, VAE patch parallel, torch.compile, profiler options.
- Optimization target: latency, throughput, memory, cost, or quality-preserving speed.
- Precision/quality tolerance: bf16/fp8/quantization/sparsity/approximate attention allowed or not.

## Workflow

1. **Freeze the measurement protocol and commands.**
   - Before analyzing parallel strategies or traces, establish the three commands for the target model/workload: server startup, single user request, and user benchmark.
   - Prefer checked-in scripts or one-off shell scripts over free-form commands in chat. The scripts should make workload variables explicit, including model path, port/base URL, prompt/media inputs, resolution, frames, steps, seed, warmup, measured iterations, concurrency, and output path.
   - Measurement reports must preserve the concrete process, not just the final numbers. For each tested configuration, write down the server startup command, the user request command, the user benchmark command or polling loop, the final server response metrics, and the client-side timing/HTTP result.
   - Present all duration and latency values in **milliseconds (ms)** in tables and summaries. If an API returns seconds, convert to ms before reporting. Keep raw units only inside quoted raw logs or source snippets.
   - Keep profiler modes separate in the command set:
     - Baseline/benchmark commands should avoid torch profiler and stack collection.
     - Diagnostic commands may enable torch profiler after a baseline identifies a bottleneck.
   - If any of the three commands is missing, ask for it or create a proposed script before proceeding.
   - Make clear which command is authoritative for correctness validation and which command is authoritative for performance numbers.

2. **Collect a real baseline.**
   - Disable PyTorch profiler and stack collection.
   - Disable or fix torch.compile state so A/B is fair.
   - Avoid `--enforce-eager` for production-speed baselines unless eager is the target.
   - Prefer `--log-stats` and `--enable-diffusion-pipeline-profiler` for low-overhead stage timing. PR 3069 has the relevant metrics/log-stats changes; if local code does not include them, fetch or cherry-pick the minimal metrics changes rather than merging unrelated PR drift.
   - Run warmup requests and exclude first-request lazy init.

3. **Model the parallel strategy before testing.**
   - Estimate compute and communication for self-attention, cross-attention, FFN, CFG, VAE encode/decode, and HSDP.
   - Select a small candidate matrix rather than exhaustively testing everything.
   - Typical 2-card candidates: `USP=2`, `CFG=2`, `USP=1/HSDP on-off`, VAE parallel on-off if memory allows.
   - Typical 4-card candidates: `USP=4`, `CFG=2 x USP=2`, `USP=2 x HSDP`, VAE parallel on-off.
   - Typical 8-card candidates for official CFG-enabled video diffusion models:
     - Primary candidate: `CFG=2 x USP=4` with VAE patch parallel across all 8 ranks.
     - Compare against: `CFG=1 x USP=8` to test whether larger sequence parallel groups beat CFG branch parallelism.
     - Isolate VAE parallelism: keep the DiT strategy fixed and compare VAE patch parallel on/off or different VAE patch sizes.
     - If long-video `diffuse` remains dominant and Ulysses all-to-all is suspected, test a hybrid sequence strategy such as `CFG=2 x USP=2 x Ring=2`.
     - Test HSDP off only if the model fits without it; treat HSDP primarily as a memory strategy until A/B proves otherwise.
   - Prefer USP/SP for long video token sequences; prefer CFG parallel when CFG doubles transformer forwards and sequence length is modest.
   - Convert workload shape to latent/patch token counts before choosing candidates. For video models, record frames, latent frames, latent HxW, patch size, approximate token count, and which stages should scale with the token count.

4. **Search for the best parallel configuration.**
   - Start with a small matrix that answers one question per comparison:
     - `CFG parallel vs larger USP`: compare `CFG=2 x USP=world/2` against `CFG=1 x USP=world` for CFG-enabled workloads.
     - `VAE patch parallel value`: compare the best DiT strategy with VAE patch parallel enabled and disabled.
     - `HSDP cost`: compare HSDP on/off only when both configurations fit memory.
     - `Ulysses vs Ring`: test a Ring/Ulysses hybrid only after long-sequence `diffuse` is confirmed dominant.
   - For each configuration, create or record a stable config id, for example `A_cfg2_usp4_vaepp8_hsdp_tiling`.
   - For each config id, capture:
     - exact server command and environment variables,
     - observed distributed setup from logs, such as SP groups, CFG groups, HSDP shard/replicate sizes, VAE patch size,
     - exact request command for every scenario,
     - final server response metrics and client-side elapsed time,
     - output artifact paths and any failed/empty responses.
   - Use one warmup request per scenario, then at least three measured repeats for the shortlisted config. For exploratory matrix pruning, one measured request is acceptable only if the margin is large; label it as one-shot.
   - Compare configurations by stage, not only end-to-end latency. A configuration can improve `diffuse` while hurting `vae.decode`; record both effects.
   - Select the best config only after checking the target metric, dominant stages, memory headroom, and output correctness.

5. **Run targeted A/B tests.**
   - Change one variable per test.
   - Keep model, input, seed, request parameters, GPU placement, and warmup policy fixed.
   - Record latency, stage timings, memory, output quality, and logs.
   - Report comparison tables in ms. Include at least end-to-end client time, server `inference_time`, server stage generation time if available, `vae.encode`, `diffuse`, `vae.decode`, and peak memory.

6. **Collect diagnostic trace only after narrowing hypotheses.**
   - Use torch profiler for a small number of requests.
   - Run two separate diagnostic traces instead of mixing concerns:
     - **Operator/shape trace**: enable `torch_profiler_record_shapes=True` and keep stack collection disabled. Use this to rank CUDA kernels, NCCL collectives, attention/MLP/norm/RoPE work, and shape-specific hot operators.
     - **Host-stack trace**: enable `torch_profiler_with_stack=True` and normally keep shape collection disabled. Use this to map CPU/Python host gaps, synchronization points, scheduler paths, and request handling overhead.
   - Keep both trace commands and reports separate from baseline/benchmark commands. Torch profiler latency is diagnostic only and must not be used as the final latency claim.
   - Prefer profiling only the narrowed dominant scenario, for example the highest-resolution/video-length case where `diffuse` dominates.
   - If profiler endpoints are available, run one warmup request first, then call `/start_profile`, run one profiled request, and call `/stop_profile`. This keeps model initialization and warmup out of the diagnostic trace.
   - Start by analyzing rank 0 only. Expand to more ranks only if rank 0 suggests imbalance, unclear GPU idle/free bubbles, high NCCL wait, server timing mismatch, CFG branch imbalance, or USP group stragglers.
   - Diagnostic reports must be written to disk and preserve: server command, profiler config, warmup/request/polling commands, trace artifact paths, rank analyzed, analyzer output or summary, and the decision about whether additional ranks are necessary.
   - Analyze both rank-level balance and device-level free bubbles when additional ranks are opened.

7. **Analyze host, communication, and operators.**
   - Find GPU idle/free intervals and map each large gap to the enclosing CPU/Python code.
   - Separate real GPU idle from profiler overhead such as CUPTI `Command Buffer Full`.
   - Compare NCCL kernel time to user annotations; annotations can overcount nested intervals.
   - Rank operator work by total CUDA time and by repeated small-kernel launch count.

8. **Produce an optimization plan.**
   - Classify candidates as P0/P1/P2.
   - For each candidate, state necessity, expected benefit, implementation path, validation plan, and quality risk.
   - Do not implement high-risk operator rewrites before proving the operator is a bottleneck for the target shapes.
   - End the plan with a user-facing candidate selection table. The assistant
     should not automatically choose a risky optimization just because it is
     technically possible. Present the options clearly and let the user decide
     which item is worth implementing based on latency target, engineering
     budget, memory headroom, and quality tolerance.
   - Organize the plan by optimization layer, not by a flat list of ideas:
     - host/runtime optimization,
     - measurement/benchmark reliability,
     - parallelism and communication,
     - VAE encode/decode and media pre/post processing,
     - operator fusion and layout cleanup,
     - attention main-path optimization,
     - algorithmic, precision, or approximation changes.
   - For each layer, tie every candidate back to evidence from baseline
     metrics, diagnostic trace, source code, or output quality requirements.
     Do not list generic optimizations without a trace or workload reason.

## Priority Rules

- **P0:** low risk, likely useful, or required for trustworthy measurement. Examples: real baseline, warmup, targeted parallel A/B, disabling avoidable `empty_cache`, scheduler coefficient caching.
- **P1:** meaningful code changes with contained risk. Examples: cross-attention KV caching, VAE gather/broadcast reduction, AdaLayerNorm/RMSNorm/RoPE fusion after trace evidence.
- **P2:** high implementation or quality risk. Examples: FA to LA replacement, custom Triton/CUDA fused kernels, FP8/quantization, sparsity/Rainfusion-style acceleration.

Every implemented optimization needs A/B validation and quality regression. A/B means same workload and hardware before/after. Quality regression means checking generated image/video stability, artifacts, temporal flicker, and seed behavior when precision or approximate kernels change.

## Optimization Layers

After baseline, parallel-search, and diagnostic traces, summarize optimization
opportunities by layer. This is the core of the performance analysis: the goal
is to connect evidence to a scoped implementation and a validation plan.

### Host and Runtime Optimization

Purpose: remove CPU/Python stalls, synchronization points, allocator overhead,
and request-path overhead that leave GPU lanes empty.

Evidence to look for:

- High `idle_pct` or large `GAP` blocks in `trace_analyzer.py`.
- Host-stack trace lines such as `torch.cuda.empty_cache`,
  `cudaStreamSynchronize`, `cudaDeviceSynchronize`, Python locks, scheduler
  waits, image/video preprocessing, or repeated small allocation paths.
- Difference between client wall-clock time and server `inference_time_s`.

Typical candidates:

- Make avoidable `torch.cuda.empty_cache()` optional or guard it by memory
  headroom.
- Cache scheduler coefficients, timesteps, masks, or other tiny repeated CPU
  computations when the request shape/steps are fixed.
- Remove avoidable host-device synchronizations and blocking logging/stat calls.
- Move expensive preprocessing out of the critical path or cache fixed prompt,
  image, and transform work for benchmark scenarios.
- Ensure benchmark scripts record client-side elapsed time, HTTP status, output
  path, and server response metrics.

Priority guidance:

- Usually P0 when the change is measurement reliability or an obvious removable
  synchronization.
- Usually P1 when it changes scheduling, memory lifetime, or request execution
  order.

Validation:

- Re-run non-profiler baseline with same workload and seed.
- Confirm peak memory headroom if disabling cache cleanup.
- Confirm generated output exists and quality/seed behavior is unchanged.

### Parallelism and Communication Optimization

Purpose: choose the right decomposition for CFG branches, sequence tokens, model
weights, VAE tiles, and rank topology.

Evidence to look for:

- Baseline A/B across `CFG`, `USP/SP`, `Ring`, `HSDP/FSDP`, and VAE patch
  parallelism.
- Stage timing shifts: `diffuse`, `vae.encode`, `vae.decode`, and server
  end-to-end.
- NCCL kernel time from trace, not only `user_annotation` time.
- Rank imbalance across SP group ranks or CFG branch ranks.
- Memory headroom and OOM risk.

Typical candidates:

- `CFG=2 x USP=world/2` versus `CFG=1 x USP=world` for CFG-enabled models.
- VAE patch parallel on/off or patch size tuning.
- HSDP/FSDP on/off only if both configurations fit memory.
- Ulysses versus Ulysses+Ring only after long-sequence `diffuse` is confirmed
  dominant and all-to-all is suspected.
- Rank mapping/topology changes if all-rank traces show stragglers or NCCL wait.
- Buffer reuse or preallocation for FSDP/HSDP all-gather paths.

Priority guidance:

- P0/P1 for configuration-only changes with strong measured wins.
- P1 for buffer reuse or rank mapping changes.
- P2 for invasive distributed algorithm changes.

Validation:

- Measure by stage and memory, not only end-to-end.
- Use one-variable A/B with identical prompt/media/seed/shape/steps.
- When communication is suspected, compare rank0-3 in one USP group and rank0
  versus rank4 across CFG branches for `CFG=2 x USP=4`.

### VAE and Media Pipeline Optimization

Purpose: reduce encode/decode, tiling, split/gather, and media conversion time.

Evidence to look for:

- Large `vae.encode` or `vae.decode` in low-overhead stage timings.
- Host-stack gaps in VAE tile split, gather, merge, broadcast, or image/video
  transforms.
- VAE kernels or cuDNN convolution in operator trace.
- Whether every rank needs the final decoded tensor.

Typical candidates:

- Keep VAE patch parallel enabled when it has clear measured benefit.
- Reduce VAE gather/broadcast to only ranks that need the final media output.
- Reuse tile metadata, split buffers, or gather buffers.
- Evaluate bf16/autocast behavior for VAE only with visual quality checks.
- Avoid redundant image conversion, resize, or tensor construction in repeated
  benchmark runs.

Priority guidance:

- P0/P1 if VAE is a large share of the target workload or if a host gap is
  obvious and low risk.
- Lower priority when `diffuse` dominates and VAE is already patch-parallelized.

Validation:

- Compare `vae.encode`, `vae.decode`, server end-to-end, and peak memory.
- Check output video integrity, artifacts, flicker, and seed stability.

### Operator Fusion and Layout Cleanup

Purpose: reduce high-frequency small kernels, memory bandwidth pressure, layout
conversions, and launch overhead in transformer and VAE blocks.

Evidence to look for:

- Top operator tables showing many `aten::copy_`, `aten::cat`,
  `split_with_sizes_copy`, `aten::add`, `aten::mul`, `aten::div`, norm,
  activation, RoPE, or reshape/layout kernels.
- `ops_rankN.xlsx` `by_shape` sheet showing repeated small shapes inside the
  same block path.
- Trace lanes showing many short kernels between larger GEMM/attention kernels.
- Source code patterns with repeated elementwise chains or layout conversions.

Typical fusion targets:

- AdaLayerNorm / RMSNorm / LayerNorm plus scale/shift fusion.
- RoPE fusion with Q/K layout preparation when shapes are stable.
- Residual add, scale, gate, and elementwise chains.
- MLP gate/up/down cleanup, such as fusing activation and multiply around
  GEMM outputs when feasible.
- QKV projection and reshape/split/cat path cleanup.
- Attention pre/post layout cleanup to avoid unnecessary copies, cats, and
  splits around sequence parallel all-to-all.

Priority guidance:

- P1 when implemented with existing PyTorch/Triton/local helper patterns and
  validated against exact outputs or tolerances.
- P2 when it requires custom CUDA/Triton kernels, changes numerics, or touches
  attention math directly.

Validation:

- First prove the operator family is material for the target shape.
- Use non-profiler A/B for latency and stage timing.
- Use quality regression checks for generated video stability.
- Check compile behavior and graph breaks if using `torch.compile`.

### Attention Main-Path Optimization

Purpose: address the dominant self-attention cost when FlashAttention or other
attention kernels dominate CUDA time.

Evidence to look for:

- Operator trace where FlashAttention/SDPA kernels dominate total CUDA time.
- Attention shape from `ops_rankN.xlsx` `by_shape`, model code, or trace
  metadata.
- Whether attention cost scales with latent frames, latent H/W, patch size, or
  CFG duplication.
- Layout/copy/all-to-all work around attention.

Typical candidates:

- Verify the attention backend and shape are on the intended fast kernel path.
- Compare supported attention backends only with identical workload and quality
  settings.
- Reduce attention input size by safe model/config choices when allowed:
  latent resolution, frame count, patching, boundary ratio, or windowing.
- Remove avoidable layout conversions before/after attention.
- Reuse condition-side KV or other static inputs if the model structure allows.
- Consider custom kernels, sparse/window/linear attention, or approximation only
  after quality risk is accepted.

Priority guidance:

- P1 for backend/config/layout changes with preserved math.
- P2 for approximate attention, sparsity, custom kernels, or any change that can
  alter quality/temporal consistency.

Validation:

- Always include output quality and seed behavior checks.
- Compare `diffuse`, server end-to-end, peak memory, and attention kernel time
  in diagnostic traces if needed.

### Algorithmic, Precision, and Approximation Optimization

Purpose: reduce mathematical work or precision cost beyond local code cleanup.

Evidence to look for:

- A single operator family dominates even after low-risk runtime, parallel, and
  fusion work.
- Memory bandwidth or compute utilization suggests precision or quantization
  could matter.
- The user explicitly allows quality-preserving or approximate methods.

Typical candidates:

- FP8/quantization for transformer or selected projections.
- Sparsity or Rainfusion-style acceleration.
- Reduced steps, scheduler changes, distillation, or caching across frames.
- Approximate attention or linear attention.

Priority guidance:

- Usually P2 because quality, numerics, and implementation risk are high.

Validation:

- Requires strict A/B, visual quality review, temporal flicker checks, seed
  stability, and possibly human evaluation.

### Interpolation, Super-Resolution, and E2E Pipeline Optimization

Purpose: optimize the whole user-visible video product, not only the base
diffusion invocation. Some deployments trade base-model latency against
post-processing, interpolation, or super-resolution stages.

Evidence to look for:

- E2E latency breakdown across base generation, interpolation, super-resolution,
  encoding, storage, and response streaming.
- Fast/slow GPU or fast/slow stage analysis across multiple cards and pipeline
  stages.
- User quality target: resolution, FPS, temporal smoothness, and acceptable
  post-processing artifacts.

Typical candidates:

- Add or optimize a frame interpolation stage when it reduces required base
  model frames for the same perceived FPS.
- Add or optimize a super-resolution model when generating lower base
  resolution plus SR is faster for the target quality.
- Analyze E2E2 pipeline behavior: client request, service scheduling, diffusion,
  VAE/media, post-process, file write, and response.
- Identify fast/slow cards or stages and rebalance pipeline placement.

Priority guidance:

- P1 when using proven interpolation/SR components without changing diffusion
  math.
- P2 when quality risk is high or the pipeline adds significant operational
  complexity.

Validation:

- Measure E2E wall-clock, per-stage server timings, output FPS/resolution,
  artifacts, flicker, and user-visible quality.

### Optimization Candidate Library

Use this table as a compact menu, not as automatic recommendations. Pick items
only when baseline metrics, trace evidence, source inspection, or quality
tolerance supports them.

| Layer | Candidate | Evidence | Priority | Validation focus |
|---|---|---|---|---|
| Measurement | Freeze server/request/benchmark commands | Missing or drifting commands | P0 | Repeatable non-profiler A/B |
| Measurement | Separate baseline and diagnostic profiler runs | Profiler used for latency claims | P0 | Low-overhead stage timings |
| Host/runtime | Guard or remove avoidable `empty_cache` | Host-stack gaps or sync stalls | P0/P1 | Latency, peak memory, OOM safety |
| Host/runtime | Cache scheduler coefficients/timesteps | Repeated tiny CPU/GPU work | P0/P1 | Same seed/output, stage timing |
| Host/runtime | Reduce framework scheduling overhead | Client time exceeds server time | P1 | E2E latency, throughput |
| Parallel | `CFG=2 x USP=world/2` vs `USP=world` | CFG doubles forward work | P0/P1 | `diffuse`, NCCL, memory |
| Parallel | Tune VAE patch parallelism | VAE encode/decode is material | P0/P1 | VAE time, output correctness |
| Parallel | HSDP on/off or buffer reuse | HSDP affects memory/all-gather | P1 | Memory, latency, OOM risk |
| Parallel | Ulysses vs Ulysses+Ring | Long sequence all-to-all suspected | P1/P2 | Rank balance, NCCL kernels |
| Cross-attn | Disable SP for short condition tokens | Cross-attn comm exceeds compute | P1 | `diffuse`, correctness |
| VAE/media | Reduce VAE gather/broadcast | Rank traces show VAE wait | P1 | Rank balance, output file |
| VAE/media | Reuse tile metadata/buffers | Tile split/merge host gaps | P1 | `vae.encode/decode`, memory |
| VAE/media | VAE bf16/autocast | VAE float kernels are slow | P1/P2 | Artifacts, flicker, seed stability |
| Operator fusion | AdaLayerNorm/LayerNorm fusion | Norm plus scale/shift kernels | P1 | Numeric tolerance, latency |
| Operator fusion | RMSNorm fusion | Many small RMSNorm kernels | P1 | Numeric tolerance, latency |
| Operator fusion | RoPE cache/fuse/layout cleanup | RoPE copy/reshape kernels | P1/P2 | Kernel count, correctness |
| Operator layout | QKV or attention layout cleanup | Copy/cat/split around attention | P1 | Copy kernels, compile behavior |
| Attention | Verify backend fast path | FA/SDPA dominates trace | P1 | `diffuse`, attention kernels |
| Attention | FA to LA or selected-head LA | Attention remains dominant | P2 | Quality, temporal stability |
| Precision | Transformer FP8/quantization | Compute/bandwidth bound and allowed | P2 | Quality, speed, stability |
| Sparsity | Rainfusion-style acceleration | DiT compute remains dominant | P2 | Prompt diversity, quality |
| Pipeline | Frame interpolation | Fewer base frames can meet FPS | P1/P2 | E2E latency, motion artifacts |
| Pipeline | Super-resolution | Lower base res plus SR may win | P1/P2 | Detail quality, artifacts |
| E2E | Fast/slow-card analysis | Multi-card stragglers | P0/P1 | Per-rank/stage wall-clock |

### Optimization Plan Template

Use this table shape when reporting the next work items:

| Priority | Layer | Candidate | Evidence | Expected benefit | Implementation path | Validation | Quality risk |
|---|---|---|---|---|---|---|---|
| P0 | Host/runtime | Guard `empty_cache` | Host-stack gap points to `torch.cuda.empty_cache` | Small latency reduction, less idle | Add config/env guard | Non-profiler A/B, memory check | Low |
| P1 | Operator fusion | RMSNorm/AdaLayerNorm fusion | High-frequency norm/elementwise kernels | Lower launch/bandwidth overhead | Use existing fusion helper or targeted Triton | A/B + output check | Medium |
| P1/P2 | Attention | Attention layout/backend investigation | FA kernel dominates CUDA time | Potentially large | Inspect shapes/backend and remove layout copies | A/B + trace + quality | Medium/high |

Then present a short selection prompt using the same rows:

```text
Which candidate should we implement next?

1. P0 Host/runtime: guard empty_cache
   - Expected benefit: small but low-risk latency reduction.
   - Risk: possible memory increase/OOM if memory headroom is insufficient.

2. P1 Operator fusion: inspect by_shape and implement first norm/RoPE/layout fusion
   - Expected benefit: medium if high-frequency small kernels are confirmed.
   - Risk: numerical/compile/quality validation needed.

3. P1/P2 Attention: FA/LA/backend/layout investigation
   - Expected benefit: potentially large.
   - Risk: high quality and implementation risk.
```

If the user has not chosen an item, default to explaining tradeoffs and asking
which candidate to execute. Only proceed autonomously on low-risk P0 measurement
or instrumentation fixes.

## Analysis Helpers

PyTorch profiler traces are Chrome/Perfetto-compatible JSON files, usually
`trace_rankN.json` or `trace_rankN.json.gz`. They normally contain a top-level
`traceEvents` list, though some exporters emit the raw event list directly.

Use the checked-in analyzer from the repository root:

```bash
python3 .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  vllm_profile/.../trace_rank0.json.gz \
  --min-gap-ms 5 \
  --topn 20
```

For rank imbalance or communication questions, pass all relevant rank traces in
one command. For host gaps, lower `--min-gap-ms` to `1` and use a host-stack
trace. Read `gpu_span_s`, `busy_union_s`, `idle_union_s`, `idle_pct`, `GAP`
blocks, `Top GPU/operator events by total duration`, and `Top NCCL-like events
by category`. Treat `cat=user_annotation` NCCL ranges as enclosing annotations;
prefer `cat=kernel` or `cat=gpu_user_annotation` for real device work.

The analyzer summarizes timing only. It does not parse tensor shapes, attribute
overlap to individual streams, prove quality, or provide final latency claims.
Use `ops_rankN.xlsx` or PyTorch key averages for shape analysis, and re-test any
optimization with non-profiler baseline commands.

Read `references/optimization-playbook.md` when drafting the optimization table or comparing candidate techniques.

## vLLM Omni Heuristics

- Cross-attention usually should not use USP/SP when text/image condition token count is much smaller than latent video tokens. Confirm via trace; in Wan2.2 I2V, self-attention dominates cross-attention.
- VAE bf16/autocast is often worthwhile but requires visual quality checks.
- VAE patch parallel can help decode/encode but may add gather/merge/broadcast overhead. Check whether all ranks need the final decoded tensor.
- HSDP/FSDP is primarily a memory strategy. If the model fits without it, run an on/off latency comparison.
- Scheduler work can create small host/device gaps; cache tiny solve coefficients when timesteps/order are known.
- `torch.cuda.empty_cache()` can prevent OOM but creates synchronization/idle. Make it optional if memory headroom is sufficient.
- `Command Buffer Full` in profiler output is profiler overhead, not a model optimization target.
