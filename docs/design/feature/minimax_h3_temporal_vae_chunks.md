# MiniMax H3 temporal VAE chunk callback

!!! warning "Experimental internal capability"

    This feature is an opt-in model integration seam. The current MiniMax H3
    pipeline still uses complete-output VAE decode, and no CLI option, serving
    path, transport, or MP4 sink enables temporal chunks by default.

## Status and scope

MiniMax H3 decodes video latents in temporal windows, but the released VAE
normally exposes only the complete reconstructed video. The temporal callback
publishes frames after the released decoder has completed overlap blending and
padding removal. This gives a future runtime consumer a stable point at which
it can schedule representation conversion, D2H transfer, and encoding while
the VAE continues decoding later windows.

The implementation remains model-local:

- `vae.py` owns VAE topology and the thin
  `decode_latent_with_chunks()` facade;
- `chunked_decode.py` owns the normalized callback contract, metadata
  validation, and fixed VAE-group rendezvous points; and
- `temporal_chunks.py` owns the source-gated compatibility planner and the
  released temporal assembly control flow.

Crop policy, typed media, transport, backpressure, encoding, and public output
remain runtime responsibilities. The model code does not schedule requests or
select a presentation format.

## Execution flow

The default and opt-in paths are intentionally separate:

```text
Default:
latent -> MiniMaxH3VideoVAE.decode_latent() -> complete RGB video

Opt-in producer seam:
latent
  -> MiniMaxH3VideoVAE.decode_latent_with_chunks()
  -> group-local preflight and callback-owner validation
  -> released temporal decode / overlap blending / padding removal
  -> borrowed decoder-domain chunk
  -> normalize + clone into owned RGB float32 BCTHW [0, 1]
  -> synchronous callback
  -> complete RGB video is still returned
```

Calling `decode_latent_with_chunks()` is the opt-in. Supplying no callback on a
VAE peer does not select the default decoder; every rank in a VAE patch group
must enter the chunked method, and only the output rank receives the callback.

## Producer contract

The normalized callback has this structural signature:

```python
def callback(
    frames: torch.Tensor,
    /,
    *,
    chunk_index: int,
    total_chunks: int,
    frame_start: int,
    is_final: bool,
) -> None:
    ...
```

For a successful decode:

- `frames` is owned, contiguous RGB float32 in `BCTHW` layout and `[0, 1]`;
- callback storage does not alias the returned complete video;
- the consumer may retain or mutate a chunk;
- `chunk_index` is zero-based, contiguous, and strictly increasing;
- `total_chunks` is positive and stable for the decode;
- `frame_start` equals the end of the previous published range;
- chunks are nonempty and contain only temporally committed frames;
- exactly one chunk has `is_final=True`; and
- concatenating chunks along time exactly reconstructs the returned video.

The callback runs synchronously on the producer's current device stream. A
consumer that reads the tensor on another stream must establish ordering, for
example with an event, and retain the allocation for that stream's lifetime.
A slow callback also blocks VAE production. An overlap-oriented runtime
consumer should therefore acquire bounded capacity, enqueue side-stream work,
and return promptly instead of encoding frames inside the callback.

Chunks retain the decoded VAE canvas dimensions. A future runtime integration
must apply the same requested-height and requested-width crop as the complete
MiniMax H3 pipeline before declaring the media transport-ready.

The raw compatibility callback is deliberately narrower. It receives borrowed
decoder-domain tensors and must not retain or mutate them. The coordinator is
the ownership boundary that normalizes and clones raw chunks before invoking
the public callback.

## Temporal planning and terminal publication

The released configuration advances five latent tokens per temporal window and
includes a two-token overlap, so each `_adaptive_decode` input normally contains
seven tokens. After temporal scaling, pre-padding removal, and five-frame
overlap blending, the normal committed unit is 17 output frames. The final
remainder depends on the requested frame plan.

Representative released frame plans include:

| Output frames | Published chunks |
| ---: | --- |
| 124 | `7 x 17 + 5` |
| 209 | `12 x 17 + 5` |
| 243 | `14 x 17 + 5` |
| 362 | `21 x 17 + 5` |

The terminal callback is held until the complete frame plan passes its logical
frame, padding, written-frame, chunk-count, and finality checks. Earlier chunks
are semantically stable, but they do not imply that the overall request has
succeeded; a consumer must discard or abort partial output if a later decoder
failure terminates the request.

## Source and configuration gate

The released VAE has no public temporal-callback API. vLLM-Omni therefore owns
a compatibility implementation for one exact released code contract. The
reference revision identifies its provenance, but runtime acceptance is based
on source content, configuration, and structure rather than repository or
weight metadata:

- reference Hugging Face revision, not independently checked at runtime:
  `42ed227ee7df40d41602854ae760620d6eb651fe`;
- sorted 15-file top-level `video_vae/*.py` manifest (excluding
  `__init__.py`) SHA-256:
  `30e64afbaa940696bf8bfe2c86003a4b1666c84b22fa3aa12403a3e6a03f705d`;
- clip length `17`;
- token chunk, overlap, and drop `5`, `2`, and `3`;
- temporal ratio `4`;
- frame pre-padding and overlap `3` and `5`; and
- no isolated first or last frame.

The adapter does not verify Hugging Face revision metadata, repository identity,
or model weights. The manifest pin detects Python compatibility drift; it is
not a security boundary for remote-code execution. The source manifest, listed
temporal attributes, structural contract, plan, and dtype are validated before
the first `_adaptive_decode` call. A mismatch disables only the opt-in chunk path.
`decode_latent()` remains available and performs no source fingerprint or chunk
rendezvous.

## Supported models

Support in this table means the internal producer contract above, not public
serving or end-to-end transport support.

| Model or component | Source contract | Status | Validation |
| --- | --- | --- | --- |
| MiniMax H3 FL2VA video VAE | Released manifest/listed temporal attributes above (reference `42ed227e`) | Supported, opt-in | Same-environment full-output parity and L20X/B300 feasibility measured on the implementation through PR commit `60e9bbad` |
| MiniMax H3 Ref2VA video VAE | Same released manifest/listed temporal attributes | Source-compatible; not separately benchmarked | Ref2VA and FL2VA use byte-identical video-VAE source and configuration in the reference snapshot |
| MiniMax H3 with a different Python manifest or listed temporal attribute | Different enforced contract | Unsupported for chunk mode | Fails closed before decoder collectives; complete decode remains available |
| Same manifest/temporal attributes with different weights or other component metadata | Not checked by this gate | Not independently qualified | The adapter does not inspect weights, repository identity, revision metadata, or arbitrary component configuration |
| Other video models | Any | Unsupported | No generic temporal VAE producer ABI is declared by this feature |

## Feature compatibility

| Dimension or combination | Status | Constraint |
| --- | --- | --- |
| Existing complete-output decode | Supported and default | No source check, callback object, or callback rendezvous |
| Explicit chunk decode | Supported internally | Caller invokes `decode_latent_with_chunks()` directly |
| VAE patch parallel size 1 | Supported by the model seam | Caller supplies the callback only on the request output owner; no WORLD rendezvous is added |
| VAE patch parallel size equal to the full DiT group | Supported when data parallel size is 1 | Every VAE-group rank calls; exactly group rank 0 owns the callback |
| Partial VAE patch subgroup | Unsupported | MiniMax H3 currently accepts size 1 or the full DiT group only |
| Data parallel size greater than 1 with VAE patch parallel size 1 | Model seam only | Per-replica callback ownership is possible, but no serving/runtime consumer is wired |
| Data parallel size greater than 1 with VAE patch parallel size greater than 1 | Unsupported | Fails before decoder collectives because no replica-local VAE patch group is defined |
| Too few spatial tiles for the VAE group | Correctness fallback | Decodes rank-locally and restores native tiling state; slower than patch parallel decode |
| Device memory and full-output assembly | Not reduced or bounded | The compatibility path still assembles and returns the full decode while allocating owned callback chunks; no peak-HBM reduction is claimed |
| DiT cache, attention, TP, SP, HSDP, LoRA, or quantization features | No direct callback dependency | This seam runs after denoising and does not independently qualify combinations beyond the MiniMax H3 model's own matrix |
| Layerwise or distributed component offload | Not separately validated | Component residency must place the video VAE before decode |
| Step execution or request batching | Not integrated | No pipeline caller binds a runtime chunk consumer |
| Requested-size crop and device uint8 preparation | Not implemented here | Runtime-owned follow-up |
| Pinned D2H, SHM/ZMQ, or cross-process transport | Not implemented here | Runtime-owned follow-up |
| Complete MP4 or fragmented MP4 sink | Not implemented here | Existing serving behavior still waits for complete output |
| Audio VAE chunks and incremental A/V composition | Not supported | Video chunks do not imply audio or request finality |
| Cancellation and multi-output lease ownership | Not implemented here | Requires a bounded runtime lifecycle |
| NVIDIA CUDA | Feasibility evidence | L20X and B300 measurements apply to the implementation through `60e9bbad`; the current coordination refactor was not reprofiled |
| AMD ROCm | Not validated for this feature | Base MiniMax H3 support is documented separately and does not imply temporal-chunk validation |
| Ascend, Intel, or other accelerators | No feature support claim | Consult the model support table independently; this design adds no platform qualification |

## Distributed and failure semantics

When VAE patch parallelism is active, all VAE-group ranks run the same
preflight before entering checkpoint collectives. Source/planner errors and the
resolved temporal concat dtype are reconciled at fixed group rendezvous points.

Callback ownership is also a group decision. Rank `r` contributes `r + 1` when
it owns a callback and zero otherwise. A group SUM equals one only when rank 0
is the sole owner; every invalid owner configuration therefore fails
identically on all ranks before decode.

Callback exceptions are retained on the output rank. Further publication stops,
but every rank completes the same remaining native decoder collective schedule.
The original exception is then raised on the owner and
`MiniMaxH3ChunkCallbackPeerError` on peers.

Exceptions raised inside the checkpoint's native distributed decode cannot in
general be reconciled by the adapter: a peer may already be blocked inside a
checkpoint-owned collective. Resolving such failures requires native collective
handling or supervisor-level timeout and worker-group teardown. The process
group must not be assumed reusable after such a failure.

## Validation evidence

The contract suite covers ordinary and padded frame plans, blend boundaries,
owned storage and mutation isolation, source-manifest/listed-temporal-attribute
drift, callback failure, and healthy subsequent requests. Real Gloo processes cover
rank-asymmetric preflight failures, dtype disagreement, two-rank callback
failure propagation, three-rank wrong-owner rejection, insufficient-tile
fallback, and group recovery for failures that occur at adapter-owned
rendezvous points.

Four-device L20X and independent four- and eight-device B300 experiments on the
implementation through PR commit `60e9bbad` showed exact serial-versus-chunked
frame/MP4 parity within each environment and lower VAE-to-complete-MP4 latency
when an experimental consumer was attached. Hashes are not claimed to match
across different hardware/software environments. The later coordination and
module-boundary refactors were revalidated with CPU and real Gloo tests but were
not reprofiled on GPUs. These are feasibility measurements, not a promise for
default serving, because this change does not contain the transport or encoder
consumer. See [RFC #6872](https://github.com/vllm-project/vllm-omni/issues/6872)
and [PR #6885](https://github.com/vllm-project/vllm-omni/pull/6885) for the
measurement provenance and limitations.

## Generalization boundary and follow-ups

This feature does not define a repository-wide `SupportsVideoChunkDecode`
interface. H3 temporal blending, Wan causal caches, LTX temporal/spatial halos,
and pipeline-level generation chunks have different commit and failure
boundaries. A generic producer contract should be extracted only after a second
model proves the same semantic event without `Any` arguments or model-name
dispatch.

The reusable future boundary is the emitted, committed video-media chunk, not a
universal latent-to-VAE method. It should extend the typed media work in
[RFC #6541](https://github.com/vllm-project/vllm-omni/issues/6541) and
[PR #6615](https://github.com/vllm-project/vllm-omni/pull/6615), while keeping
model-specific halo, cache, padding, blending, distributed, and source-gating
logic inside each model adapter.

Runtime follow-ups own:

- typed video-plus-audio media and request/output identity;
- requested-size crop and device preparation;
- bounded leases, backpressure, cancellation, and cleanup;
- D2H/IPC transport and observability; and
- complete-MP4 and fragmented-MP4 consumers.

## Related documentation

- [Diffusion model integration](../module/diffusion/diffusion_model_integration.md)
- [VAE patch parallelism](vae_parallel.md)
- [Async diffusion output](async_diffusion_output.md)
- [MiniMax H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
