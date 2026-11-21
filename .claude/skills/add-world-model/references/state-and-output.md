# Causal State and Output Contracts

## Execution modes

Inspect current capabilities and runner code before selecting a mode.

| Mode | Call scope | State retirement | Required evidence |
| --- | --- | --- | --- |
| Ordinary offline/full video request | `forward()` returns a complete result. | Completion or failure. | Public parameters, complete output, request isolation. |
| Typed session/tick | One tick advances a model block. | Reset, close, eviction, or failure. | Continuity, metadata, commit, cross-tick state. |
| Stepwise streaming | One request produces multiple chunks. | Completion, abort, or failure. | Runner binding, step scheduling, incremental decode, release. |

An internal chunk loop does not establish stepwise scheduling. Verify engine,
execution flags, output representation, and transport independently.

## Ownership and commit

| State | Owner |
| --- | --- |
| Self/cross-attention KV, page tables, pending scratch | Shared allocator and runner, keyed by session. |
| KV semantics and ordered retained segments | Model policy over runtime-owned storage. |
| Image condition, encoder/decoder history, camera, RNG, audio history | Model session state, retired by lifecycle hooks. |
| Pending interactions, accepted/applied events, chunk indices | Existing session and interaction contracts. |
| Graphs, LoRA wrappers, and weight banks | Existing worker/model facilities; invalidate graphs before releasing captured storage. |

Expose bound AR state only during a runner invocation and clear the binding
on exit. Cross-step scratch remains on the runner-owned session. A generic
execution path cleans its own state without accessing uninitialized AR fields.

For each chunk:

1. Prepare inputs from committed state and reserve pending/scratch storage.
2. Run denoising probes with the model's intended history visibility.
3. Produce persistent KV at its clean-context forward or equivalent boundary.
4. Validate results and typed metadata before committing chunk/event state.
5. On failure, release and fail the session; recover through the protocol's
   reset or new-request behavior.

Check safe cancellation points and whether waits can be interrupted. A prompt
update may invalidate text cross-attention while retaining causal history;
camera and audio resets follow their own model rules.

## Memory and rollout bounds

For equal-geometry layers, estimate per-rank self-attention KV tensor bytes as:

`2 * num_layers * local_kv_heads * head_size * element_bytes * allocated_tokens * local_branch_slots`

Add cross-attention, scratch/staging, metadata, model-owned persistent state,
and concurrent work. Apply the actual allocator's bounds; logical CFG branches
may share physical rank-local slots. For segmented KV, account for each
segment's partial page rather than rounding total tokens once, and pass exact
valid lengths independently of page padding.

Include condition-encoder caches, VAE convolution feature maps, withheld
latents, pending histories, and in-flight copies in per-session admission.
Inspect multiple live sessions when that capability is supported. Record
loading peaks, steady residency, and offload host memory separately.

Derive boundaries from the model: sink/recent window, condition horizon,
segment capacity, RoPE table, audio reset, graph bucket, maximum token
position, and output buffering. Cross the affected boundaries in validation.
Raising one limit does not prove the remaining limits are removed. Advance
causal condition encoding according to the reference rather than repeating an
old encoded tail block.

## Image, camera, and interaction inputs

Verify world-to-camera versus camera-to-world transforms, relative poses,
axes, intrinsics scaling, Plucker/RoPE ordering, and pixel/latent event timing.
Derive image resize/crop and intrinsics together. Reuse driver materialization
and validated input snapshots instead of re-reading source files per block.

Check accepted/applied metadata and actual content or camera-state changes.
Include short press/release sequences that can land on the same generation
boundary. Use existing scheduler wait/pacing facilities; abort must release a
session held while waiting for an interaction or deadline.

## Stable video output

Preserve decoder temporal state per session. Reinitializing a causal decoder
for each block repeats first-frame expansion rules and can alter frame counts
and continuity.

For lookahead decoders, distinguish generated latents, available pixels,
stable emitted frames, and withheld tail frames. Flush normal completion
according to the decoder contract; validate cancelled partial output against
its actual termination state. Equal input turn lengths need not produce equal
stable-frame counts per turn.

Validate batch/channel/time layout, RGB order, value range, and final frame
count. Compare complete float RGB before uint8/codec conversion for numerical
quality. Same-latent one-shot versus streaming decode isolates decoder drift
from generation drift.

## Audio/video delivery

When A/V is in scope, verify precodec samples, chunk payloads, encoded tracks,
and client playback as separate boundaries.

- Validate sample rate, channel layout, missing/empty audio, explicit rate
  changes, and late tracks before advancing the stream.
- Preserve metadata/binary correspondence for buffered calls. A protocol may
  permit `byte_length=0`; buffering is different from completion.
- Distinguish submitted frame/sample/event counts from emitted bytes. Buffered
  content can arrive in a later fragment or the final trailer.
- Check AAC priming/edit lists, first audio content, monotonic DTS/PTS,
  close-time padding, and both tracks when remuxing saved output.
- Progressively decode real fMP4 and inspect final frames, samples, and time
  ranges. Codec checks and browser playback are separate validations.
- Reject ambiguous combinations such as multiple videos sharing one audio
  payload before emitting part of the result, according to the current API.

Changes to shared transport also retain existing video-only behavior.
