# PersonaPlex generated-audio drain (2026-09-07)

## Contract

The subsequent [incremental transport optimization](personaplex_incremental_optimization_20260908.md)
adds bounded history, explicit codec offsets and no-wait backlog coalescing
while retaining this generated-audio drain contract.

PersonaPlex now uses the shared `continuous_stream` close path. Close waits
for accepted append receipts, pads a partial input frame through the existing
input buffer, and waits for every fully generated acoustic frame to reach
the output delivery fence. Success is `completed/stream_drained`, followed
by `session.end` and `session.closed`. Missing output times out as a failure;
explicit cancellation is still cancellation.

This supersedes the cancelled-close/four-frame-tail result in the
[ModelScope acquisition report](personaplex_modelscope_20260907.md).
It is generated-audio drain, **not** a claim that N input frames produce N
model-generated output frames. One acoustic frame needs the next step's
cb1..7 predictions. For N accepted frames, this staged implementation has
`max(N - 1, 0)` fully generated acoustic frames. Close explicitly reports
`model_delay_frames`, `expected_audio_frames`, and actual `audio_frames`.
No generated frame may remain unsent at successful close.

## Official behavior checked

NVIDIA's [pinned offline implementation](https://github.com/NVIDIA/personaplex/blob/3428dfd95309a7f3c84fd93259ded0f810d1ff91/moshi/moshi/offline.py)
feeds the input to `LMGen.step` and collects generated frames. At EOF it
trims or zero-pads the exported WAV to the input duration (lines 301–309).
It does not perform additional EOF model sampling in that loop. Output WAV
duration alone therefore does not prove that every sample was generated.

We do not append fake output PCM or silently inject more user audio to make
frame counts equal. The explicit one-frame delay describes the existing
staged de-delay algorithm; no byte-identical official-model parity is claimed.

## Implementation

- At a resumable PersonaPlex segment boundary, send each complete acoustic
  frame immediately instead of retaining it until the next five-frame batch.
  The one raw successor row remains for codebook alignment. Non-resumable
  offline requests retain configured batching.
- Opt the PersonaPlex adapter into the already-shared continuous-stream
  lifecycle rather than adding a second close state machine.
- Track engine sequence acknowledgements with `max` so retries cannot
  double-count input. Track output delivery separately from projection.
- Reject non-frame-aligned or excess output accounting. Audio encoding that
  returns no payload cannot be counted as successful delivery.
- Tighten the real-model driver from a four-frame allowance to one explicit
  model-delay frame. Require completed/drained close and check the server's
  delivery accounting against actual received PCM. Save events after close.

This fixes a model-specific buffering policy and connects it to a shared
architecture contract. MiniCPM and Nemotron lifecycle policies are unchanged.

## Validation

CPU regression: **332 passed, 1 deselected**, including shared handler,
PersonaPlex, Nemotron and stage-input tests. New tests cover varying segment
counts, duplicate receipts, delivery-before-completion, partial/excess output,
encoding failure, and close while a full/partial append is pending. Deliberately
missing output fails close instead of being reported as completed.
Running the new segment-boundary regression against the old runtime produces
four expected failures (2, 3, 5, and 7 input frames); all seven boundary
lengths pass with the fix. Chunk-aligned old cases pass, demonstrating why a
single convenient input length was insufficient coverage.

First H200 run, with public receipt tracing:

| Session | Accepted inputs | Delivered audio frames | Explicit model delay | Close |
| --- | ---: | ---: | ---: | --- |
| Primary | 525 | 524 | 1 | completed/stream_drained |
| Survivor across slot reuse | 1050 | 1049 | 1 | completed/stream_drained |
| Replacement | 525 | 524 | 1 | completed/stream_drained |

The old run delivered 521/1046/521 frames and cancelled on close. Each session
now delivers three additional generated frames (240 ms); the model-delay
frame is not fabricated. Overflow admission remains rejected; slot reuse does
not change the survivor's response identity. This is not a throughput claim.

Tradeoff: resumable output packet cadence rises from roughly five-frame
batches to individual 80 ms frames. This removes batching wait/tail retention
but increases codec calls and packet overhead. No CPU-overhead or maximum
concurrency benchmark is claimed in this correctness follow-up.

Artifacts are under workspace `duplex_official_alignment_20260907/`; the first
run is `results/personaplex-drain-v1/`. The final candidate adds an explicit
audio-encoding failure guard. Its runtime fingerprint, excluding unrelated
profiler work, is:

```text
50a936cf5bfba8674983bdd898fb529adfcda4d4ba2ffbeb309af1d5ac324812
```

The isolated candidate also passed the tightened driver without receipt tracing
in `results/personaplex-drain-final-gpu4/`. All three sessions completed with
the same 524/1049/524 generated-frame counts and checked delivery accounting.
That isolated run's fingerprint is
`ae2dd2e9d73554b6236c59d57c4bb590767e551f4c0bbec2539e6283db8d9845`;
the local worktree fingerprint above additionally includes existing changes
in `session_attachment.py`, `session_runner.py`, and `websocket.py` that were
not modified by this tail-drain patch.
The complete current-worktree snapshot was subsequently tested separately in
`results/personaplex-drain-worktree/`: CPU **332 passed, 1 deselected**, and
H200 dual-session/slot-reuse/checked-close passed without receipt tracing.
Its before/after runtime fingerprint matches the local value above.
Every task-owned process was reclaimed (`remaining_owned=[]`).
The earlier GPU 0 and GPU 2 launch attempts were refused by the occupancy
guard before starting a model; those are not test failures or GPU runs.

## Reproduction

Prerequisites: matching Linux vLLM 0.28.0/Omni environment and pytest development
dependencies. CPU tests need no weights or GPU. From the repository root:

```bash
# Local close regression.
python -m pytest -q tests/entrypoints/openai/test_continuous_duplex.py

# CI-like focused sweep; already collected by the existing L1 pipeline.
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python -m pytest -q \
  tests/entrypoints/openai/test_continuous_duplex.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/model_executor/models/personaplex/duplex \
  tests/model_executor/stage_input_processors/test_personaplex.py \
  tests/model_executor/models/nemotron_voicechat/duplex \
  -m 'core_model and cpu' --run-level=core_model
```

Use the same verified ModelScope assets and ownership-contained H200 launcher
as the acquisition report, with source directory
`/root/duplex-official-alignment.2uKOQy/source-personaplex-drain-worktree` and a
fresh result label. The launcher checks GPU occupancy before starting and
never kills unrelated GPU jobs. No new environment/container, PR or commit
was created. H20/H100/NPU, live KV migration, stress/concurrency expansion,
and the independent browser backend are not signed off by these checks.
