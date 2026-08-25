# MiniCPM-o 4.5 Code2Wav batch=1 low-copy path

Status: implemented and statically validated; A3 numerical and score evidence
are pending.

## Hypothesis

The official Seed-TTS score workload uses `max_concurrency=1`. The generic
batch path nevertheless launches singleton `split`, `stack`, and `cat`
operations around codec tokens, Flow/DiT caches, and HiFT caches. Reusing
request-owned storage for those singleton boundaries should reduce Stage 2
launch and memory-copy overhead without changing model math.

## Ownership boundary

- A one-segment request returns the original flattened token view; a one-item
  codec bucket adds only a batch-dimension view.
- A batch=1 Flow cache transfers newly produced tensors directly into the
  request state, and the next decode reads them through a shallow container
  copy. Flow/DiT return new cache tensors, so the previous state remains
  unchanged if the decode succeeds.
- Batch sizes greater than one retain the original cat/split implementation.
- HiFT history inputs reuse their single request tensors, but the small next
  mel/source/speech tails still clone. Those clones prevent a small state view
  from retaining an entire generated waveform or mel allocation and keep the
  emitted audio storage isolated from future state.

No dtype, step count, chunk size, sampling, public API, SSE, or deploy YAML is
changed. The path activates only when the actual exact-shape bucket size is one.

## Promotion gate

- CPU ownership tests and batch>1 regression tests pass.
- A3 batch=1 eager/Graph token, shape, dtype, audio, and next-state outputs are
  equal to the parent commit.
- Official Chinese c=1 mean RTF improves by at least 2% with TTFP/TTFT no more
  than 1% worse, or RTF remains within 0.5% while TTFP improves by at least 5%.
- Peak HBM growth remains at or below 5%, all requests/audio decode, and the
  four complete quality gates pass before promotion.
