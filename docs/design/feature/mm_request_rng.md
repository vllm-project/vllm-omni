# Request-owned multimodal random streams

HunyuanImage3 conditional VAE sampling uses one random stream per seeded AR
request. Reference images consume that stream in their original request order,
as intended by [#4145](https://github.com/vllm-project/vllm-omni/pull/4145).
[Issue #9362](https://github.com/vllm-project/vllm-omni/issues/9362) exposed that
the previous implementation instead used the first seed for an entire encoder
batch. Its result depended on unrelated requests and which images were cached.

## Ownership and interfaces

```mermaid
flowchart TD
    P[Hunyuan processor: resolved seed] --> I[Omni InputProcessor: request context and encoder identity]
    I --> E[EngineCore: scheduler request]
    E --> S[Scheduler: work and compact image metadata]
    S --> R[Omni Runner: request lifecycle]
    R --> G[RequestRNGProvider: generators and checkpoints]
    R --> B[Existing encoder batching: request handle and image index]
    B --> H[Hunyuan: draw specifications]
    H --> G
    G --> V[VAE posterior sampling]
    V --> G
```

The optional `SupportsMMRequestRNG` contract binds a model to a Runner-owned
provider. The model supplies the ordered random draw specifications and the
operation that consumes a skipped draw. The provider owns generators and
checkpoints; it never uses the AR token sampler's generator. Each executing
worker owns device-local objects. Generator objects are not serialized or
transferred between AR and DiT stages.

`MMRNGContext` carries the policy version, the processor's actual seed, and
the names of compact metadata fields needed by the model. Hunyuan emits this
context on every processor invocation, including all-cache-hit invocations.
`OmniInputProcessor` preserves it for both raw prompts and rendered engine
inputs. It delegates tokenization and multimodal preprocessing to the existing
renderer and delegates request construction to upstream vLLM.
For raw prompts, parameter, LoRA and DP-rank checks run before rendering can
mutate the sender cache. A request rejected by these early checks leaves no
unsent image entries that could break the next request's cache lookup.

Before encoder batching, the Runner attaches a CPU `(request_handle, mm_index)`
field to a shallow copy of each scheduled image item. Upstream grouping and
encoding remain responsible for batching and output order. Shared processor
cache items never contain runtime handles. `embed_multimodal(**kwargs)` keeps
its upstream signature.

The draw plan and image-to-draw positions are built once per request. Each
encoder dispatch maps only its scheduled items to original image indices.
Checkpoints cover a contiguous prefix of draw boundaries, allowing direct
lookup without scanning prior images or checkpoints. Request updates revalidate
the cached plan at the lifecycle boundary, outside the per-image encoding loop.

## Cache behavior and random-stream restoration

Processor `mm_hash` remains unchanged. Encoder `identifier` is derived from
the policy, seed and original identities of the reference prefix through the
current image. Including the original identities preserves existing adapter
scope. The transformation happens before the scheduler constructs KV block
hashes. Request IDs are excluded, so identical reference prefixes can reuse
encoder outputs across requests. Different preceding images conservatively
prevent reuse even if their random draw shapes happen to match.

For image `i`, the provider restores the closest checkpoint at or before `i`.
It replays missing predecessor draws using the same shape, dtype, device and
`randn_tensor` operation as the posterior sampler, discarding those temporary
noise tensors. It does not execute the predecessor VAE forwards. A successful
sample records the next checkpoint; an exception in sampling does not commit
that image. A repeated or reordered encoder call restores its logical position
instead of consuming the current tail of the stream.

KV prefix hits can remove image pixels before dispatch. The Omni scheduler
adapter preserves `vae_token_grid_hw` and `vae_generator_seed` from the full
request, independently of M-RoPE configuration. SHM handles stay intact until
the worker's normal receiver resolves them. Hunyuan checks its inferred draw
specification against the actual posterior before sampling.

Pausing or preempting a request preserves its RNG state. Completion and abort
remove it, including steps with zero scheduled tokens. Reusing a request ID
creates a new handle. Worker shutdown clears the remaining state.

## Compatibility and validation boundary

The implementation binds request RNG in the Omni GPU/AR Runner and follows the
existing seed-forwarding paths. A delivered seed, including zero, enables the
request stream. Absent seeds retain `generator=None` and global RNG behavior;
DiT and token-sampling RNG are unchanged.

Seeded engine inputs must retain the processor-produced context. The reference
list and sampling plan remain fixed within a request; incompatible updates are
rejected before reusing its cached plan.

Regression tests use production processing, cache, scheduling, batching and
posterior sampling with tiny network substitutes. Comparisons are exact within
one device and dtype; they do not assert equality between CPU/CUDA or different
precisions. These tests establish RNG and lifecycle behavior, not pretrained
Hunyuan image quality, distributed performance, or AR-to-DiT integration.
CPU cases cover request lifecycle, cache transport and plan reuse. CUDA cases
focus on mixed-request sampling and missing-predecessor replay, each in FP32,
FP16 and BF16.

Run from the repository root with the supported vLLM and Omni dependencies:

```bash
python -m pytest tests/inputs/test_mm_rng.py tests/model_executor/models/test_hunyuan_request_rng.py -m 'core_model and cpu' -q
python -m pytest tests/model_executor/models/test_hunyuan_request_rng.py -m 'core_model and cuda and L4' --run-level=core_model -q
```

The CUDA tests require one CUDA GPU and no model weights; the L4 mark selects
the existing CI runner tier. Both test paths are already collected by the
CPU and model-executor CUDA sweeps in `test-ready.yml`.
