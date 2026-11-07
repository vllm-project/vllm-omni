# Hunyuan AR VAE request RNG

HunyuanImage3 conditional VAE sampling uses one random stream per seeded AR
request. Reference images consume it in input order, as intended by
[#4145](https://github.com/vllm-project/vllm-omni/pull/4145).
[Issue #9362](https://github.com/vllm-project/vllm-omni/issues/9362) exposed that
one encoder batch instead shared its first seed, making conditioning depend on
unrelated requests and cache hits.

## Scope and sampling contract

For fixed model/preprocessing configuration, reference inputs, their order,
seed and execution backend/dtype, each AR reference-image occurrence must
sample as if its request ran independently and encoded every reference in
order. Encoder batching, execution reordering,
cache-skipped predecessors and recomputation must preserve that logical stream.
This contract concerns conditional VAE samples and their cache identities; it
does not establish bitwise equality of all AR KV tensors or final images.

The bug was found while investigating the combination of AR KV import with
local DiT prefix hits in [#9192](https://github.com/vllm-project/vllm-omni/issues/9192).
As described in the [original scope statement](https://github.com/vllm-project/vllm-omni/issues/9192#issuecomment-6407797456),
[#9382](https://github.com/vllm-project/vllm-omni/pull/9382) fixes the AR producer.
Missing-only transfer, cross-stage identity alignment and a shared seedless
policy remain separate integration work. The AR bug also affects requests
without KV transfer, so its fix has an independent correctness target.

## Processor identities and existing transport

For seeded AR inputs, Hunyuan's `_get_mm_hashes` extends each image's upstream
hash with a policy version and the preceding reference hashes. The upstream hash already includes
`vae_generator_seed` and the processor options. The resulting prefix chain
scopes both processor and encoder caches before scheduling and KV block hashing.
Upstream encoder LoRA scoping remains unchanged. Request IDs are excluded:
identical seeded reference prefixes can reuse encoder outputs across requests
when the other existing cache-key components match, while `[A, A]` has distinct
identities for its two sampling positions. The image identity contributes to AR
KV block hashes; the other existing KV-cache key components still apply.

This deliberately reduces processor-cache reuse across reference positions.
Changing an earlier reference can require later images to be processed again.
UUID-only inputs work when the corresponding prefix-scoped entries are cached;
if a new prefix misses, the caller must provide the image data again. UUIDs are
not permanent media storage: upstream also requires a cache hit when pixels
are omitted. No measured processing or memory overhead is claimed.

Processor-cache hits restore preprocessed inputs and do not by themselves skip
VAE sampling. Encoder-cache hits reuse sampled image embeddings; AR KV-prefix
hits reuse prompt-prefix state. Either can omit an encoder call, requiring
replay of its logical draw when a later reference still needs encoding.

The processor validates the signed 64-bit seed before mutating its cache and
stores it in each image's existing `vae_generator_seed` field. Before field
splitting or cache writes, it validates the seed and grid tensors' presence,
CPU placement, integer dtype and batch shapes, matching seeds and positive
grid dimensions. Invalid RNG metadata raises a request validation error;
it cannot populate the LRU/SHM cache. Cache hits resolve previously validated
items. Manually constructed or mutated pre-rendered `MultiModalInput` objects
bypass this processor and remain a trusted internal-input boundary.
Both metadata fields use `keep_on_cpu=True`. Hunyuan uses M-RoPE, so upstream
prefix stripping retains these fields after dropping covered image pixels.
LRU payloads are restored before scheduling; SHM payloads are restored before
worker execution. The worker reads the seed from the resolved image item.
There is no additional request field or custom input-processing prelude.

## Worker ownership and replay

The optional `SupportsMMRequestRNG` interface binds the model to a runner-owned
`RequestRNGProvider`. The model declares its seed field, supplies an ordered
shape/dtype/device draw plan, and implements the exact skipped-draw operation.
Each worker owns its device-local generator and checkpoints. They are separate
from AR token-sampling state and are not transferred between stages.
This covers serialized eager multimodal encoder execution. Capturing the VAE's
request-owned generators in encoder CUDA graphs is outside this change.

The runner builds the plan once when a seeded request first needs encoding.
It attaches `(request_handle, mm_index)` to a shallow copy of each scheduled
item, preserving shared processor-cache entries. Request ID plus the contextual
encoder identifier associates results with original image indices through
upstream filtering and reordering. `embed_multimodal(**kwargs)` is unchanged.

For image `i`, the provider restores a checkpoint at or before its logical draw
position. Missing predecessors are replayed with the same `randn_tensor` shape,
dtype and device as posterior sampling; their noise is discarded without VAE
forwards. Reordering and recomputation restore the same position. Failed draws
do not commit a checkpoint. Device aliases are resolved before comparing the
planned and actual posterior specifications.

All-prefix hits need no generator until encoding becomes necessary. Preemption
preserves existing RNG state; completion, abort and shutdown release it.
Reusing a finished request ID creates a new handle. Updates to an existing request must
preserve its references, seed and draw plan.

Legacy GPU and NPU runners initialize the provider during model loading.
MRv2 lacks the required encoder metadata/lifecycle integration and rejects
models declaring this capability at startup. Use `model_runner: v1` for this
Hunyuan AR path. Missing seeds retain global-RNG sampling and the original
image-based cache identities; seed zero enables request-owned sampling.

## Separate DiT behavior and integration questions

On the DiT posterior-sampling path, the uncached draw order is: conditional VAE
draws in reference order, then initial generated-image noise, then subsequent
draws. CFG copies sampled conditioning without another posterior draw for each
branch. These draws use the DiT request generator, independently of the
worker-local AR generator fixed here.

The current DiT path skips conditional draws when AR-KV import or native KV
covers those images, shifting the initial noise and later RNG state. This
pre-existing issue is not fixed by the AR changes. Exact skipped-draw replay or
an equivalent checkpoint is a possible separate fix, as suggested in the
[PR review](https://github.com/vllm-project/vllm-omni/pull/9382#pullrequestreview-5729773061).
Such replay needs the actual DiT posterior shape/dtype/device, including
autocast. A VAE backend returning tensors directly does not make the same
posterior sampling call, so the AR draw specification cannot simply be reused.

Suggested validation for that separate fix compares initial noise and later
generator state across misses, native prefix hits and AR-KV imports, starting
from the same random state and compatible conditioning. Repeated and unequal
reference sizes, partial coverage and supported CFG/VAE configurations matter.
Matching RNG consumption alone does not prove that AR and DiT conditioning or
their KV layouts are interchangeable; that needs cross-stage validation.

Seedless behavior is unchanged here. AR uses global-RNG sampling with
image-based identities, allowing reuse of a previously sampled encoder result.
DiT uses an explicit generator's device/state in its reference identity when
provided (ahead of a numeric seed); without either, it scopes that identity to
the request ID. These are existing behaviors, not a newly unified policy.

Before composing the caches, the integration needs to resolve whether seedless
requests should reuse previously sampled conditioning, how compatible AR and
DiT identities are established, and what happens when compatibility cannot be
established. This document records those open questions without choosing a new
seedless default, an export metadata protocol or an import-rejection policy.

## Validation

Regressions exercise production processing, caching, scheduling, batching and
posterior sampling with tiny network substitutes. They cover mixed seeds,
zero/absent seeds, duplicate references, reorder/recompute, partial encoder and
prefix hits, LRU/SHM transport, request lifecycle and one-time plan construction.
GPU/NPU load-hook tests verify provider wiring on CPU; MRv2 tests verify startup
rejection. CPU FP32/BF16 tests use a random-weight `AutoencoderKLConv3D` with
smaller hidden widths and image sizes. Additional CUDA FP32/BF16 tests use the
complete official encoder configuration and the production 1024x1024,
768x1024 and 1024x768 resolution buckets. They execute real encoding with the
VAE's native spatial tiling (384-pixel tiles), compare posterior shape/dtype/device
with the draw plan, and check replay samples and final RNG state bitwise.
The configuration is pinned to the
[HunyuanImage-3.0-Instruct revision](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct/blob/2ec2c78bee7d4b94157341fba86c4c2c7b1858b2/config.json).

```bash
python -m pytest tests/inputs/test_mm_rng.py tests/model_executor/models/test_hunyuan_request_rng.py -m 'core_model and cpu' -q
python -m pytest tests/model_executor/models/test_hunyuan_request_rng.py -m 'core_model and cuda and L4' --run-level=core_model -q
```

CUDA cases cover FP32/FP16/BF16 mixed-request sampling and predecessor replay.
Equality is checked within one device and dtype. These checks do not establish
pretrained image quality, NPU hardware inference, distributed performance or
AR-to-DiT parity.
