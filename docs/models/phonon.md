# Phonon 31M EN

Initial single-stage MRV2 integration for `gradium/phonon-31m-en`.
The default deployment uses one request, TP=1, BF16 AR/Flow and FP32 Mimi.
The default pretrained voice produces 24 kHz mono audio in 1920-sample frames.

## Serve

Install the repository and its TTS dependencies, including `sentencepiece` and
`safetensors`, in a CUDA-enabled vLLM environment. The model ID uses the default HF cache and downloads missing files as needed.
Only `config.json`, `model.safetensors`, `tokenizer.model` and
`default-voice.safetensors` are required for these model tests.

```bash
vllm serve gradium/phonon-31m-en --omni \
  --deploy-config vllm_omni/deploy/phonon.yaml
```

Use `/v1/audio/speech` with `voice="default"` and English text. The adapter
prepares the official SentencePiece text IDs and precomputed speaker latents.
The model checks the speaker file's `sig@epoch` against `embeddings_ext`.
Whitespace, capitalization and sentence-final punctuation are normalized.
Full official English number/date/currency normalization is not ported.

## Execution and state

AR uses vLLM paged attention and runner CUDA graphs. The stateless LSD flow
head has a separate main-stream graph. Its normalized latent supplies the
next AR embedding through `model_intermediate_buffer`.

Mimi consumes a denormalized latent through its quantizer output projection,
depthwise upsampling, two Transformer layers and SEANet decoder. Mimi owns
fixed input buffers and slot-indexed KV/conv pools. Attention writes into the
ring in place and applies the causal context mask in physical ring order.
Convolution updates read from separately materialized history/input storage.

The side stream waits for latent production, copies into the codec graph's
input, and decodes there. The main stream waits for completion of that copy
before it may reuse Flow output storage. All codec work is serialized on one
side stream. PCM is copied into owned storage before the next replay. The
existing MRV2 streaming-audio copy interface waits for its producer event.
Slot reset/reuse waits for the last codec operation before clearing state.

AR and Mimi share `input_batch.idx_mapping`. B>1-shaped tensor interfaces are
retained, while admission explicitly requires B=1. The pipeline fails a
request on recompute preemption. Cancellation releases its shared slot.
EOS includes the triggering frame plus three tail frames for short text or
one for longer text, following the official prompt policy. Every frame is
emitted immediately, including the final frame.

## Local regression

Run the lightweight model-state, sampler and kernel regressions from the
repository root. These tests use synthetic inputs or fake modules and require
no checkpoint download.

```bash
pytest -v --noconftest -o addopts='' tests/model_executor/models/phonon
```

Full-weight compiled decoder and CUDA Graph output-lifecycle
checks are in the offline e2e suite. They resolve the four required files from
`gradium/phonon-31m-en` through the standard HF cache. An optional
`PHONON_CHECKPOINT_DIR` overrides the model source.

```bash
pytest -v --noconftest -o addopts='' tests/e2e/offline_inference/test_phonon.py
```

In a fully provisioned CI environment, use the advanced-model routing.

```bash
pytest -v --run-level advanced_model -m 'advanced_model and tts' \
  tests/e2e/offline_inference/test_phonon.py
```

The CUDA e2e case requires a CUDA GPU and the compile/Triton dependencies.
`--noconftest` supports isolated local runs without global diffusion fixtures.

## Validation status

Local macOS CPU validation passed seven focused tests with official weights.
The CUDA graph test was skipped because CUDA is unavailable locally. Config,
pipeline and TTS adapter registration were checked locally. Full GPU serving,
BF16 audio quality, graph capture/replay, cross-stream execution and performance
remain server validation items. The AR backbone uses vLLM paged KV exclusively.
The tests establish finite audio, EOS completion and codec chunk consistency.
They do not establish numerical parity against a separately executed official
runtime or a speech intelligibility score.

The implementation was checked against the official `gradium-ai/xn-ptts`
source at `e5110afd3ead725c940b6411eec4985fb8b111aa`, including its actual
Mimi configuration, conditioner formulas, speaker projection and EOS policy.

## Remaining implementation work

- Admission and request RNG currently support B=1, TP=1. Batch-shaped tensor
  interfaces still need lifecycle and RNG work before enabling B>1.
- Ring attention uses PyTorch indexed writes, gathers and reductions. Dedicated
  ring-attention kernels have not been implemented.
- Convolution history is alias-safe using separate concatenated storage.
  Preallocated ping-pong scratch and temporary-allocation reduction remain.
- Feedback is stored in both `model_intermediate_buffer` and a slot embedding
  pool. Decode currently reads the pool, leaving duplicate feedback state.
- Recompute preemption fails the request. Saving/restoring a suspended request's
  codec and feedback state remains unimplemented.
- Only the default pretrained speaker is exposed. Online voice cloning and
  additional checkpoint variants remain outside this first implementation.

This implementation and its local validation were AI-assisted with Codex.
Human review and CUDA/server validation remain required before merge readiness.

## Default-voice benchmark

Start the Phonon service, then run the existing universal TTS benchmark against
it using the bundled English smoke dataset. Start with concurrency 1 because
the initial deployment admits one request at a time.

```bash
python benchmarks/tts/bench_tts.py \
  --model gradium/phonon-31m-en \
  --task default_voice --locale en \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --concurrency 1 --num-prompts 20 --num-warmups 2 \
  --host 127.0.0.1 --port 8000 \
  --output-dir ./results/phonon
```

If the server was started with `--served-model-name phonon`, also pass
`--served-model-name phonon` to the benchmark. The registry sends `voice=default`
and English conditioning. Live benchmark performance remains unvalidated.

BF16 AR eager and full-graph Dynamo capture are covered by a CPU regression.
RoPE computes phases in FP32 and casts rotated Q/K back to their input dtype.
CUDA backend compilation and service warmup still require server validation.
