# Hunyuan AR VAE request RNG

[Issue #9362](https://github.com/vllm-project/vllm-omni/issues/9362) exposed that
an encoder batch used its first request's VAE seed. This fix preserves
[#4145](https://github.com/vllm-project/vllm-omni/pull/4145)'s request generator:
for fixed inputs, reference order, configuration, seed and backend/dtype,
each reference occurrence samples as if its request encoded every image in order.
Batching, reordering, cache skips and recomputation must preserve that stream.
This establishes conditional VAE sampling consistency, not full-model KV or image parity.

## Identity and transport

Hunyuan's `_get_mm_hashes` chains each seeded image's upstream hash with the
preceding reference hashes and a policy version. The seed is already in the
upstream hash. Both processor and encoder caches use the chain, which also feeds
AR KV block hashing alongside its existing key components. Identical seeded
prefixes can reuse results across requests; `[A, A]` has two distinct identities.
Processor-cache reuse across positions decreases: UUID-only inputs need a warm
entry for that prefix, otherwise the caller must resend pixels.

The processor validates signed 64-bit seeds and CPU int64 seed/grid tensors
(presence, shapes, matching seeds and positive dimensions) before cache writes.
Pre-rendered `MultiModalInput` objects bypass it and remain trusted internal inputs.
Seed and grid fields use `keep_on_cpu=True`, so Hunyuan's M-RoPE prefix stripping
retains them. Existing LRU/SHM receivers restore items before scheduling/execution;
the worker reads the seed directly. No additional request wire field is needed.

## Worker stream

The model declares `SupportsMMRequestRNG`, supplies a shape/dtype/device draw plan
and implements skipped draws. The runner owns device-local generators/checkpoints,
separate from AR token sampling, and builds a plan on first encoding. It attaches
`(request_handle, mm_index)` to a shallow item copy, associating scheduled items by
request ID and contextual identifier through upstream filtering and reordering.

For image `i`, the provider restores its logical draw position, replaying missing
predecessors with the posterior's exact `randn_tensor` shape/dtype/device, without
VAE forwards. Device aliases are normalized; failed draws do not commit checkpoints.
Processor hits restore inputs; encoder/KV-prefix hits can skip sampling and need replay.
Preemption retains state; completion/abort/shutdown release it. Reused request IDs
get new handles; live updates must preserve references, seed and draw plan.
All-prefix hits need no generator until encoding becomes necessary.

GPU/NPU `load_model` binds the provider. MRv2 rejects this unsupported capability
at startup; use `model_runner: v1`. Execution is serialized eager encoding;
encoder CUDA graph capture is outside scope. Seed zero enables request sampling.
Seedless AR retains global-RNG sampling and image-based cache reuse.

## Separate DiT questions

The DiT posterior path draws conditional images in reference order, then initial
noise and later draws. CFG copies conditioning without extra posterior draws.
Its generator is independent of AR's. Skipping conditional encoding for AR-KV
imports or native prefix hits currently shifts the noise stream; this PR does
not fix that. A separate replay/checkpoint fix must match the actual DiT posterior
shape/dtype/device, including autocast; tensor-returning VAE backends differ.
Suggested validation compares initial noise and final RNG state across miss,
hit and import paths, including unequal/repeated references and supported CFG modes.

DiT identity uses an explicit generator's state ahead of a numeric seed, falling
back to request ID without either. AR's seedless reuse policy differs. A shared
seedless policy, cross-stage identity alignment and missing-only transfer remain
separate integration work, per the [original scope](https://github.com/vllm-project/vllm-omni/issues/9192#issuecomment-6407797456).
These are open questions, not an agreed export protocol or import-rejection policy.
Matching RNG consumption alone does not establish compatible conditioning or KV.

## Validation

Tests cover processor/cache paths, scheduler/runner batching, replay and lifecycle.
Load-hook tests replace hardware dependencies; they are not NPU inference evidence.
Real random-weight VAE tests use smaller CPU widths/images and the
[complete official encoder config](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct/blob/2ec2c78bee7d4b94157341fba86c4c2c7b1858b2/config.json)
on CUDA at 1024x1024, 768x1024 and 1024x768 with native 384-pixel spatial tiling.
They check posterior specs, samples and RNG state; CUDA request tests also cover FP16.
No pretrained quality, distributed TP, NPU hardware or AR-to-DiT E2E claim is made.

```bash
python -m pytest tests/inputs/test_mm_rng.py tests/model_executor/models/test_hunyuan_request_rng.py tests/worker/test_mm_request_rng_wiring.py -m 'core_model and cpu' -q
python -m pytest tests/model_executor/models/test_hunyuan_request_rng.py -m 'core_model and cuda and L4' --run-level=core_model -q
```
