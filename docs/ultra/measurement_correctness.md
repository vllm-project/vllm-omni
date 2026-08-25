# MiniCPM-o 4.5 measurement and correctness closure

Status: implemented for CPU/static validation; A3 execution is pending.

## Scope

`VLLM_OMNI_ULTRA_TIMELINE=1` now records process-local server boundaries in
addition to the M0 client timeline:

- `first_codec_token` at the Stage 1 codec bridge;
- `connector_put` and `connector_get` around inter-stage connector calls;
- `cfm_setup_begin/end`, `cfm_begin/end`, and `hift_begin/end` inside the
  request-owned Token2Wav backend;
- `first_pcm_ready`, `pcm_ready`, and `last_audio_ready` before Stage 2 output
  publication.

Server records contain only host timestamps and tensor metadata already
available without materialization: shape, dtype, `numel * element_size`, stage,
request/turn/chunk identifiers, and bounded error/status details. The hooks do
not call `.cpu()`, `.numpy()`, `.item()`, or an accelerator synchronization API.

Each server event is appended immediately to `events.<pid>.jsonl`. Unlike the
client recorder, a worker process does not have one reliable terminal callback
for every connector and model-stage request. Immediate diagnostic flushing
prevents timeout, abort, or process-failure evidence from being lost. This mode
is diagnostic only and must remain disabled during formal score collection.

## Accuracy gate

The MiniCPM-o accuracy test uses the conservative union of submission-guide
revision 12 and repository thresholds:

- Daily-Omni accuracy >= 0.78;
- Video-MME accuracy >= 0.68;
- Seed-TTS mean WavLM similarity >= 0.689;
- Seed-TTS mean WER <= 0.0156.

The full Seed-TTS Chinese split remains 2020 rows at concurrency 4 for quality
validation. Competition performance remains a separate concurrency-1 run.

## Correctness expectations

The existing async-chunk regression suite remains authoritative for 25-frame
emission, three-frame left context, exact-boundary and short-tail flushes,
staggered request isolation, abort/replacement epochs, duplicate terminal
suppression, and request-owned Code2Wav state. Timeline failures are swallowed
and cannot change connector fallback or inference results.
