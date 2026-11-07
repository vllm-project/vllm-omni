# Hunyuan request-owned multimodal random streams

HunyuanImage3 conditional VAE sampling uses one random stream per seeded AR
request. Reference images consume it in input order, as intended by
[#4145](https://github.com/vllm-project/vllm-omni/pull/4145).
[Issue #9362](https://github.com/vllm-project/vllm-omni/issues/9362) exposed that
one encoder batch instead shared its first seed, making conditioning depend on
unrelated requests and cache hits.

## Processor identities and existing transport

Hunyuan's `_get_mm_hashes` extends each image's upstream hash with a policy
version and the preceding reference hashes. The upstream hash already includes
`vae_generator_seed` and the processor options. The resulting prefix chain
scopes both processor and encoder caches before scheduling and KV block hashing.
Upstream encoder LoRA scoping remains unchanged. Request IDs are excluded:
identical seeded reference prefixes can reuse results across requests, while
`[A, A]` has distinct identities for its two sampling positions.

This deliberately reduces processor-cache reuse across reference positions.
Changing an earlier reference can require later images to be processed again.
UUID-only inputs work when the corresponding prefix-scoped entries are cached;
if a new prefix misses, the caller must provide the image data again. UUIDs are
not permanent media storage: upstream also requires a cache hit when pixels
are omitted. No measured processing or memory overhead is claimed.

The processor validates the signed 64-bit seed before mutating its cache and
stores it in each image's existing `vae_generator_seed` field. Before field
splitting or cache writes, it validates the seed and grid tensors' presence,
CPU placement, integer dtype and batch shapes, matching seeds and positive
grid dimensions. Invalid processor output raises a request validation error;
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
preserves existing state; completion, abort and shutdown release it. Reusing a
finished request ID creates a new handle. Updates to an existing request must
preserve its references, seed and draw plan.

Legacy GPU and NPU runners initialize the provider during model loading.
MRv2 lacks the required encoder metadata/lifecycle integration and rejects
models declaring this capability at startup. Use `model_runner: v1` for this
Hunyuan AR path. Missing seeds retain global-RNG sampling and the original
image-based cache identities; seed zero enables request-owned sampling.

## DiT integration boundary

The canonical uncached DiT order is: conditional VAE draws in reference order,
then initial generated-image noise, then subsequent draws. CFG copies sampled
conditioning without another posterior draw for each branch.

The current DiT path skips conditional draws when AR-KV import or native KV
covers those images, shifting the initial noise and later RNG state. This
pre-existing limitation is not fixed by the AR changes. Integration in
[#9192](https://github.com/vllm-project/vllm-omni/issues/9192#issuecomment-6571334678) must replay skipped
draws or restore an equivalent validated state. Replay must match the actual
posterior shape/dtype/device, including autocast. VAE backends returning tensors
directly require their own sampling contract; blindly replaying the ordinary
posterior draw is incorrect for that branch.

That integration must compare initial noise and subsequent RNG state across
local misses, native prefix hits and AR-KV imports, including repeated and
unequal-sized references and supported CFG/VAE configurations. Imported
conditioning must also have the matching reference and sampling identity.

The common policy for composed AR/DiT caches is reuse of the **same realized
conditioning**, including ordered reference occurrences and compatible
preprocessing, model/VAE and sampling provenance. An explicit generator takes
precedence over a seed: its device and state before conditional sampling matter,
not just its initial seed. Matching image bytes or numeric seeds alone does not
establish equivalence across stages.

Without an explicit generator or seed, conditioning is **request-local by
default**. A new request lifetime selects a new realization, even when a request
ID is reused. Retries and recomputation retain that identity only if the original
conditioning can be recovered or reproduced. Explicit import of an identified
existing realization is the exception: cross-request or cross-source reuse
requires the same immutable realization and compatible provenance. Otherwise,
disable that hit/import and use local conditioning. If the original realization
cannot be reproduced, a fallback selects a new identity before execution; it
cannot mix old cached pages with newly sampled conditioning under the old key.

AR-KV exports must carry the source conditioning identity and required sampling
provenance; DiT must retain that source identity in imported cache entries.
The receiver's request ID or an image-only hash cannot manufacture equivalence.
Imports must match the receiving request's selected conditioning and cannot
override its explicit generator or seed. Conditioning compatibility and DiT
stream position are separate requirements: a valid import still needs exact
skipped-draw replay or an equivalent checkpoint before initial noise.

These are the required integration rules for #9192. This AR change preserves
standalone seedless behavior: AR may reuse a global-RNG sample through its
image-based cache, while DiT otherwise uses request-local reference identities.
Legacy AR image-only keys alone do not qualify for composed-cache reuse.

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
