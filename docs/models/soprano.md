# Soprano native inference

Soprano-1.1-80M supports English, the checkpoint’s fixed voice and native eager
FP32 execution on one GPU (TP1). AR uses native vLLM Qwen3 layers and KV cache;
the second stage runs the checkpoint’s decoder. Workers do not call
Transformers generation or LMDeploy.

For pinned checkpoint/software revisions, the actual RTX 4090 Laptop 16GB
validation environment, offline/online/playback/benchmark commands and untested
combinations, use the [model recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/ekwek/Soprano-1.1-80M-RTX4090-Laptop.md).
Shared entrypoints are documented in the [offline TTS hub](../user_guide/examples/offline_inference/text_to_speech.md),
[online TTS hub](../user_guide/examples/online_serving/text_to_speech.md) and
[Speech API](../serving/speech_api.md#soprano-11-80m).

## Production contracts

- The unmodified checkpoint has ordinary Qwen3 metadata. Explicit
  `--deploy-config vllm_omni/deploy/soprano.yaml` selects complete WAV;
  `soprano_streaming.yaml` adds async PCM/WAV and SSE. The served model alias
  names the API endpoint model; it does not identify the pipeline.
- Set `TRITON_F32_DEFAULT=ieee` before Python starts. Both packaged profiles
  use eager FP32, v1 runner, one sequence per stage on GPU0, a 64 MiB AR KV
  cache, no CUDA graph pool, no prefix caching and no chunked prefill.
- The pinned checkpoint uses untied input/output embeddings. Stage 0 audits
  the native AR weights; stage 1 loads `decoder.pth` strictly without reading
  the AR tensor iterator. Re-exported tied-weight variants are unqualified.
- The production `offline.synthesize` frontend cleans/splits whole English
  utterances, validates all units before submission, and concatenates audio
  in sentence order. Low-level `Omni.generate` and the Speech API each accept
  one unit of 1–256 prompt tokens. No silent truncation or hallucination retry
  is performed. The per-unit generation budget is 1–512 tokens.
- Predictor zero is the prompt’s last hidden state (`h_prompt`, not `h_y0`).
  EOS excludes the EOS predictor; length stop retains the last valid vector.
  Fewer than two valid predictors is an explicit synthesis failure, never a
  successful empty WAV. The offline frontend raises without returning partial
  utterance audio; low-level consumers must check `audio_synthesis_error`.
- Streaming emits deltas at 32000 Hz. Default chunk size 4 is configured in
  the connector, with a supported range of 1–64. Windows retain at most
  `8 + chunk_size` hidden rows, first regular decode needs `4 + chunk_size`
  predictors, and flush ends at `2048 * (T - 1)` samples. Short generations
  can finish before a regular chunk. Audio-window state is released at finish,
  error and cancellation, including worker-slot reuse.
- A failed decoder stream stays failed for subsequent chunks of that request.
  Its failure marker is retained until the runner's finish/abort callback;
  late chunks cannot create a new window in the middle of the utterance.
- Streaming keeps the native offline predictor sequence. The original HF
  streaming path consumes a different sequence; natural sampled waveforms
  are not a numerical parity oracle. Even identical predictors can yield
  numerical differences between full and windowed decoding. Complete WAV
  on the streaming profile concatenates that profile’s emitted deltas.
- Concurrent HTTP requests do not establish GPU batching support or continuous
  playback. The generic output queues still lack complete hard-capacity
  backpressure; model/lifecycle correctness tests do not resolve this limit.

The adapted text utilities and decoder retain the original Soprano attribution
and Apache-2.0 license in `vllm_omni/model_executor/models/soprano/NOTICE` and
`LICENSE`; see the recipe for pinned upstream links.

## Tests and CI collection

Deterministic predictor indexing, EOS/length ownership, short-output errors,
window cropping/flush, borrowed-buffer ownership, cancellation cleanup and slot
reuse are CPU contract tests (`core_model and cpu`). The decoder interleaving
test uses nonzero inputs and the actual CPU decoder; its weights are synthetic.
Mocks in adapter/transport tests validate control flow, not real inference.

The fixed-token GPU integration uses the original checkpoint and real native
Qwen3 forward/KV execution. A test-owned worker extension constrains sampling
IDs after computing real logits and observes both stages through existing RPC
hooks. It checks exact position/handoff ownership, including `h_prompt` versus
`h_y0`, EOS exclusion and the last length-stop vector. It does not establish HF
numerical parity; that needs an independently calibrated reference experiment.

Online integration uses the standard `omni_server` / `online_client` fixtures
and the public clients exported by `tests/helpers/runtime.py`. The client
implementation lives in `tests/helpers/client.py`. Complete WAV and PCM/SSE
protocol smoke tests carry both `core_model` and `advanced_model`, plus `tts`.
Cancellation/peer completion/slot reuse and partial-block length flush are
`advanced_model` only. Malformed/unsupported HTTP requests and short-generation
recovery live in the existing endpoint reliability module, marked `slow and tts`.

GPU parameters collect **separate** L4 and RTX4090Laptop items via
`hardware_marks`; select the actual machine. A local RTX 4090 Laptop run does
not validate L4 hardware. These are TP1/single-slot engine tests, including
multiple HTTP requests, without a multi-row GPU batching claim.

Run from the repository root with the matching vLLM/Omni environment, pytest
plugins and the `soprano` extra installed. Set `SOPRANO_MODEL_PATH` to the
unmodified local checkpoint (otherwise the repository mechanism resolves the
HF model). Cached ASR dependencies/weights, OpenCC and an `ffmpeg` executable on PATH are additionally needed for the
standard `advanced_model` complete-speech content assertions.

```bash
pytest -sv tests/model_executor/models/soprano \
  tests/entrypoints/openai_api/test_soprano_adapter.py \
  tests/helpers/tests/test_speech_cancellation_client.py \
  -m 'core_model and cpu' --run-level core_model

SOPRANO_MODEL_PATH="$SOPRANO_MODEL_DIR" pytest -sv \
  tests/e2e/online_serving/test_soprano_1_1.py \
  -m 'core_model and tts and cuda and RTX4090Laptop and cards_1' \
  --run-level core_model

SOPRANO_MODEL_PATH="$SOPRANO_MODEL_DIR" pytest -sv \
  tests/e2e/offline_inference/test_soprano_1_1.py \
  tests/e2e/online_serving/test_soprano_1_1.py \
  -m 'advanced_model and tts and cuda and RTX4090Laptop and cards_1' \
  --run-level advanced_model

SOPRANO_MODEL_PATH="$SOPRANO_MODEL_DIR" pytest -sv \
  tests/dfx/reliability/invalid_param_test/test_invalid_audio_speech.py \
  -m 'slow and tts and cuda and RTX4090Laptop and cards_1' -k soprano
```

CI-like GPU commands select `L4` instead of `RTX4090Laptop`; they require an
actual L4 worker. L1 reuses the existing model-executor/entrypoints/other CPU
shards. L2 keeps two basic smoke scenarios in the existing Soprano ready job;
L3 reuses its merge job for the offline contract and online scenarios. HTTP
validation reuses the weekly invalid-audio-speech shard. No extra nightly or
performance job, 200-request CI requirement, or remote CI trigger is added.
Maintainers control admission and matching-head CI results.

TTFA, RTF, resource trends and long-run behavior belong to recorded performance
or reliability experiments; there is no acceleration claim or performance gate
here. Human listening is a separate, explicitly recorded acceptance step. The
current generic output queues lack explicit hard capacities/complete downstream
backpressure; successful model/lifecycle tests do not resolve that limitation.

### Fixed protocol-case HNR calibration

The PCM/SSE equality scenario uses the fixed text "Hello, this is a simple
speech synthesis test.", seed 44, temperature 0.3, top-p 0.95, top-k 50 and
repetition penalty 1.2. Its HNR floor is **0.5 dB for those two requests only**.
Other scenarios and the shared default retain 1.0 dB. ASR and exact sample
comparison across raw PCM, SSE and complete WAV remain required.

The same 32 kHz, libsndfile PCM16 measurement on the original checkpoint
(`27b5a5f5f541a1db3a51d6fd1b0fc7147b92cd01`) with official reference source
`12fac06eb8fa53bad8b3941d3cb11e9c869477c4` gave:

| Path | Measured HNR (dB) |
| --- | ---: |
| Official Transformers, FP32 AR/decoder | 0.55369 |
| Official Transformers, default BF16 AR / FP32 decoder | 0.67694 |
| Native FP32, chunk size 4 | 0.55378 |

The official FP32 value and native PCM bytes were each repeatable across three
runs. The floor rounds the independent official diagnostic reference down to
0.1 dB; it is not a statistical confidence interval. The original ten reference
texts all exceeded 1.0 dB, so that corpus alone did not cover this fixed-case
false rejection. This input is a diagnostic anchor, not an unseen holdout.

The reference uses its original `infer` path in a separate Transformers 4.57.1
environment. Its BF16 checkpoint parameters are converted in memory to FP32
for the matching-precision comparison; original files and sampling semantics
are unchanged. Natural official streaming states are a different contract and
were not assumed waveform-equivalent. CPU regressions continue to reject
silence, constant signals and white noise at the calibrated floor, including
32 kHz noise. These measurements calibrate this protocol case; they do not
establish a universal Soprano quality floor or a subjective quality score.
