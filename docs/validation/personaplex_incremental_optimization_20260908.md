# PersonaPlex incremental transport optimization (2026-09-08)

## Scope and design choice

Optimize retained-stream bookkeeping and consume-side batching without adding
an intentional wait for future input. This follows the
[checked stream-drain implementation](personaplex_stream_drain_20260907.md).
No change to the KV Append frequency, model sampling, accepted-input contract,
or explicit one-frame acoustic delay is intended.

The selected approach keeps the producer free of partially filled output
batches. Its existing delivery barrier remains the close/flush boundary:
accepted appends complete, every generated acoustic frame is delivered, then
the response completes. There is no new timer thread or producer flush RPC.
This avoids reintroducing the stranded-tail state removed in the previous fix.

## Implemented

- **Bounded Stage0 history:** retain only the current and two preceding user
  codec frames. Preserve a separate total-frame counter and independent
  prepared snapshots. The temporal model's delayed inputs and absolute frame
  clock are unchanged.
- **Explicit codec offsets:** indexed chunks carry `meta.codec_frame_offset`
  and a codebook-major `[K, F]` tensor. The decoder retains only the most recent
  chunk for replay validation. Equal audio at a new offset is new input, not
  a replay. Gaps/conflicts/stale offsets fail before advancing the codec.
  Mixing indexed and unindexed input mid-request is rejected. The unindexed
  offline/cumulative interface remains supported for existing consumers.
- **Opportunistic batching:** an explicitly opted-in `codec_coalesce` stream
  can combine already-available contiguous blocks for the same request, up
  to five frames and a 1ms probe-loop budget. The first frame is never probed
  ahead. Terminal/unknown metadata, gaps and incompatible tensors are not
  merged; one lookahead result is retained for the next receiver registration.
  Cleanup fences both lookahead and commit, so cancellation cannot resurrect
  receiver state. The budget does not claim a hard OS scheduling deadline.
- **Actual non-waiting transport:** new optional `get_nowait` defaults to no
  lookahead on unsupported connectors. Shared memory uses nonblocking file
  locking. Busy/missing data is not consumed. Ordinary transport `get` may
  block and is not used for opportunistic probes.
- **Safe Mimi batching:** preserve two-position acoustic-frame steps through
  the transformer ring. Whole-batch ring writes could evict history needed by
  the earliest queries after wrap. Quantizer/convolutions still process
  multi-frame tensors. A deterministic regression fails with the previous
  whole-batch ring implementation and passes with per-frame transformer steps.

The offset/opt-in metadata and bounded receiver mechanism are shared transport
contracts. Stage0's three-frame window and Mimi's clock belong to PersonaPlex.
Other models/connectors do not opt into batching implicitly.

## Measurements

Microbenchmarks run in the existing H200 container and vLLM 0.28.0 environment.
Raw JSON and executable benchmark scripts are in workspace
`personaplex_opt_20260908/`. The CPU comparison uses the previously validated
`source-personaplex-drain-worktree` as baseline, not upstream GitHub main.
The initial asynchronous copy named `before` raced source editing and is not
used as a baseline or cited as pre-change evidence.

### CPU bookkeeping, synthetic 3000-frame history

Five repeats, single Torch CPU thread; each timing averages the last 128
operations. This is not inference throughput or a claim that a real session
can exceed its configured context limit.

| Measurement | Previous | Optimized |
| --- | ---: | ---: |
| Codec increment identification and commit, median | 15.39 microseconds | 3.21 microseconds |
| Stage0 retained user-code tensor | 192000 bytes | 192 bytes |
| Codec retained tensor, single-frame input | 192000 bytes | 64 bytes |
| Codec retained tensor, at maximum merged batch | History-dependent | 320 bytes |

Only tensor contents are counted, not Python objects, model KV, Mimi state,
or frontend replay logs. Stage0 preparation timings fluctuate around
84–96 microseconds in repeated runs; there is no robust Stage0 CPU-speedup
claim. The demonstrated benefit there is bounded storage/copy size.

### Real Mimi CUDA decode

150 frames (12 seconds of audio; enough transformer positions to wrap the
250-position ring), one warmup and five measured repeats. End-to-end wall
time is synchronized with CUDA for each measurement.

| Frames per call | Calls | Median time | Relative to single-frame time |
| --- | ---: | ---: | ---: |
| 1 | 150 | 1.233 s | 100% |
| 2 | 75 | 1.095 s | 88.8% |
| 5 | 30 | 0.977 s | 79.3% |

This is about 21% less codec decode time when five frames are already
available. It is not a 21% service-throughput gain. The initially faster
whole-batch-transformer measurement is not the selected implementation.
Batching convolution/quantizer arithmetic changes floating-point results:
maximum absolute PCM error is about 0.004 and relative RMS error about 0.12%
on the synthetic code input. This is neither bit parity nor perceptual-quality
certification. Do not silently change global TF32 settings to hide this.

Real-time light load may have no backlog to merge. The initial traced E2E
candidate sent 518/1043/524 audio packets for 524/1049/524 output frames,
compared with one packet per frame previously: a small packet reduction,
not the theoretical five-to-one reduction for a fully queued workload.
Waiting to fill five future frames would instead add up to 320ms of batching
delay and was not enabled.

## Verification and reproduction

CPU regression: 468 passed across the shared connector, SHM, PersonaPlex,
stage-input, and close/handler suites; another 37 payload-schema tests passed.
Coverage includes repeated identical content at new offsets, replay conflicts,
invalid offsets, bounded history with correct delay/clock, first-frame
bypass, five-frame batch bounds, terminal lookahead, cancellation during
lookahead, writer-lock contention, and transformer ring wrap.
The existing L1 CPU pipeline collects these files; no new CI job was added.

Prerequisites: matching Linux vLLM 0.28.0/Omni environment with pytest dev
dependencies; CPU tests require no model weights or GPU. From the repo root:

```bash
# Local targeted checks.
python -m pytest -q tests/model_executor/models/personaplex/test_streaming_code2wav.py

# CI-like regression.
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  tests/distributed/omni_connectors/test_shm_connector.py \
  tests/model_executor/models/personaplex/duplex \
  tests/model_executor/stage_input_processors/test_personaplex.py \
  tests/entrypoints/openai/test_continuous_duplex.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/engine/test_data_entry_keys.py \
  -m 'core_model and cpu' --run-level=core_model
```

H200 uses the existing ownership-contained launcher and verified ModelScope
assets. Candidate source:
`/root/personaplex-opt-20260908.mlf6cg/verified`.
No environment/container changes or unrelated process cleanup are performed.
Runtime Python fingerprint, excluding unrelated profiler work:

```text
7d5ac8c3e8640893f605a4400f1a3120d4b2bd41792b42b28d97beae78b4ed4e
```

Final H200 run `personaplex-opt-verified-20260908` passes the unchanged strict
close gates without extra receipt logging. Primary/survivor/replacement
receive 524/1049/524 audio frames from 525/1050/525 inputs; all responses finish
as `completed/stream_drained`, with explicit model_delay_frames=1.
Packet counts are 519/1044/524: ten fewer packets across 2097 generated frames
(about 0.48%), consistent with little backlog at real-time light load.
Overflow admission and slot reuse pass; the surviving response identity is
unchanged. Before/after fingerprints match the local code, and
`remaining_owned=[]`. Earlier candidates also passed their E2E scenarios,
but only this final run includes nonblocking SHM lookahead and the ring fix.

No global optimum, arbitrary-concurrency fairness, throughput ceiling,
cross-hardware sign-off, live KV migration, or official-model bit parity is
claimed. This is a bounded no-intentional-wait implementation, not a fixed
five-frame batching policy for all duplex models.
