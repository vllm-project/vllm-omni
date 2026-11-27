# CSM-1B text-to-speech — RTX 4090

> Sesame CSM-1B behind `/v1/audio/speech` on one RTX 4090, with the packaged
> two-stage deploy config and no overrides. The talker stage replays CUDA
> graphs and generates faster than real time on this card.

## Summary

- Vendor: Sesame
- Model: `sesame/csm-1b` (gated), revision `c92a71e1c419772e25be7dc14d952c2521a740ab`
- Task: text-to-speech, optionally continuing one reference turn
- Mode: online serving through `/v1/audio/speech`, complete and streamed responses
- Hardware: 1x NVIDIA GeForce RTX 4090 24GB
- Maintainer: Community

## When to use this recipe

Use this recipe to serve CSM-1B on one 24 GB GPU. The packaged deploy config
runs unmodified and uses about 8 GiB of the card. A request takes about 0.27 to
0.39 s of wall time per second of audio with up to 8 requests in flight, so
complete responses return quickly and a stream arrives faster than it plays.

## Supported model contract

| Item | Contract |
| --- | --- |
| Endpoints | `POST /v1/audio/speech`; `POST /v1/audio/voices` stores a reference recording with its transcript |
| Text | `input` is required and is spoken as one turn. Text that contains the tokenizer's special tokens returns 400. Qualified with English text only |
| Speaker label | `extra_params.speaker_id`, a non-negative integer, default `0`. It labels the turn; it does not select a fixed voice, and there are no preset voices |
| Reference turn | Optional. One `ref_audio` with its `ref_text` (both required), or the name of an uploaded voice in `voice`. The recording must be 1 to 30 s long (the serving layer's limit) and is resampled to 24 kHz. The model continues that speaker. The reference frames (12.5 per second) and both texts must fit in 2,047 prompt tokens |
| Output | 24 kHz mono. Complete responses: `wav` (default), `pcm`, `flac`, `mp3`, `opus`. Streams were qualified with `pcm`, as raw bytes and as SSE |
| Length | One token per 80 ms frame. The default budget is 125 tokens, at most 10 s; `max_new_tokens` accepts 1 to 2048. A complete response that stops at the budget sets `x-audio-truncated: true` |
| Sampling | First codebook: temperature 0.9 and top-k 50, which a request can override through `extra_params`. Codebooks 1-31: `subtalker_sampling_params` of the deploy YAML, the same for every request |

The packaged profile, [`vllm_omni/deploy/csm.yaml`](../../vllm_omni/deploy/csm.yaml):

| Setting | Stage 0 (talker) | Stage 1 (Mimi decoder) |
| --- | --- | --- |
| Device | GPU 0 | GPU 0 |
| Precision | BF16 | FP32 |
| `max_num_seqs` | 8 | 8 |
| `gpu_memory_utilization` | 0.3 | 0.1 |
| Execution | CUDA graphs: the backbone's decode steps (`FULL_DECODE_ONLY`) and the depth decoder's frame step, which the model captures for batches of 1, 2, 4 and 8. Prompts run eagerly | eager |
| Hand-off | 5-frame chunks (0.4 s), the first after 1 frame | one decoder state per request |

## References

- Model card: <https://huggingface.co/sesame/csm-1b>
- Reference implementation: `CsmForConditionalGeneration` in Transformers
- Launch and request examples: [`examples/online_serving/text_to_speech/README.md`](../../examples/online_serving/text_to_speech/README.md#csm-1b)
- Request fields: [`docs/serving/speech_api.md`](../../docs/serving/speech_api.md#csm-1b)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- End-to-end test: `tests/e2e/online_serving/test_csm.py`

## Checkpoint

`sesame/csm-1b` is gated. Accept its license on Hugging Face and log in
(`hf auth login`, or set `HF_TOKEN`) before the first launch.

The first launch downloads the whole repository, 19.6 GB (173 s on the
measured host). That includes two files in the original checkpoint format,
`ckpt.pt` and `model.safetensors` at 6.2 GB each, which the pipeline does not
read; it loads the two `transformers-*.safetensors` shards.

## Hardware

- Accelerator model and per-device memory: NVIDIA GeForce RTX 4090, 24,564 MiB
- Number of devices: 1
- Device interconnect: not applicable, single device
- Host memory: not material, no CPU offload
- Host CPU: AMD Ryzen 9 9950X, 15.36 cores available to the container. It
  matters for stages that run eagerly; see [Timing](#timing)
- Qualification scope: runtime-qualified for online serving with the packaged
  deploy config, English text, 1 to 16 concurrent requests

Other accelerators are not covered by this recipe. The weekly CI step targets
an L4, and it had not run when this recipe was written.

## Software environment

- OS: Ubuntu 24.04.3 LTS, kernel 7.0.0-34-generic, in a container started
  from `vllm/vllm-openai:v0.34.0`
- Python: 3.12.3
- Driver / runtime: NVIDIA driver 595.91.07, CUDA 13.0 (PyTorch 2.14.0+cu130)
- vLLM version: 0.34.0 (Transformers 5.22.0)
- vLLM-Omni version or commit: `8aece1ce0c9314fc340b8d387eb7ab74a9022c15`, run from source
- Model runner: V2

## Command

```bash
vllm serve sesame/csm-1b --omni --port 8091
```

The model selects the packaged deploy config, so `--deploy-config` is not
needed. With the weights cached the server was ready after 63 to 80 s (four
starts), which includes capturing the graphs.
The two stages are always connected by chunks: `--no-async-chunk` is not
supported.

## Verification

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8091/health
```

A complete response:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello from Sesame.",
        "response_format": "wav",
        "seed": 7
    }' --output hello.wav
```

Expected: HTTP 200 and a 24 kHz mono 16-bit WAV, with the response header
`x-audio-truncated: false`. On the measured setup this request returned the
same 1.44 s file (18 frames, 20 output tokens) on every run, also after
restarting the server. The same seed can give a different sample on another
GPU or attention backend, and it gave a different one (1.20 s) with stage 0
set to `enforce_eager: true`.

A streamed response:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello from Sesame.",
        "stream": true,
        "stream_format": "audio",
        "response_format": "pcm"
    }' --no-buffer --output hello.pcm
```

Expected: HTTP 200 and raw signed 16-bit little-endian mono PCM at 24 kHz.
The first bytes arrived after about 0.06 s. Without a seed the length varies
from run to run; three runs returned 1.28 to 1.52 s in 0.39 to 0.46 s.

The end-to-end test, with the command of the weekly CI step:

```bash
VLLM_OMNI_CI_MODEL_RUNNER=v2 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    pytest -s -v tests/e2e/online_serving/test_csm.py -m "slow and L4 and tts" --run-level "full_model"
```

Expected: 5 passed. The test starts its own server. The `L4` marker names the
CI card; the run recorded below was on the RTX 4090.

## Qualification evidence

Everything below was measured on 2026-11-26 (UTC) on the hardware and software
above, at the commit above. No other process used the GPU during the timed
runs. These are bounded runs.

### Correctness

- End-to-end test: 5 passed in 178 s. It covers a text request, a streamed PCM
  request, two concurrent requests, `speaker_id` 1 and a reference turn.
  Whisper small transcripts scored 1.000 against the input for four of the six
  responses, 0.915 for one of the concurrent requests and 0.930 for the
  `speaker_id` 1 request (the test requires more than 0.9). The streamed PCM
  scored 9.6 dB on the shared harmonic-to-noise check, whose floor is 1.0 dB;
  33 seeded streams of three sentences scored 3.6 to 18.1 dB.
- Sampled output across seeds: the five requests of the test with seeds 0 to 9
  each, 50 requests with default sampling. No response was truncated. With
  Whisper small, 49 transcripts scored above 0.9 and one scored 0.729;
  Whisper large-v3, which the test falls back to, scored that one 1.000.
- Greedy comparison with Transformers 5.22.0
  (`CsmForConditionalGeneration.generate`): four short text prompts, one
  453-token text and one prompt with a reference recording. All 976 compared
  frames (31,232 codes) were identical and the waveforms differed by at most
  1.09 steps of 16-bit PCM, with stage 0 replaying its graphs and with stage 0
  eager. The long text and the prompt with the recording reached their limits
  (300 and 500 frames) on both sides, so those frames were compared but not
  the ends of those turns. This used a diagnostic configuration, not the
  packaged profile: FP32 talker on the `FLEX_ATTENTION` backend
  (`--stage-overrides '{"0":{"attention_backend":"FLEX_ATTENTION"}}'`),
  sampling turned off in the request and in the deploy YAML, and cuDNN TF32
  turned off on both sides. In BF16 the codes differ from the reference
  implementation's BF16 output from the first frame, as that implementation's
  own BF16 and FP32 outputs do, so the packaged profile rests on the
  transcript checks and on the next item.
- Sampled comparison with the same implementation: eight sampled 80 s
  responses of an FP32 eager talker (8,000 frames) were fed back to it frame
  by frame, with the served codes forced in. Under its logits the served
  codes were as likely as under the server's own: mean log-likelihood -0.597
  on both sides for the first codebook and -2.760 for codebooks 1-31. No
  first-codebook code ranked outside its top 50, and 48 of the 248,000 other
  codes fell one or two places outside it. Under both sets of logits the
  sampled codes fitted temperature 0.9 better than 0.7, 0.8, 1.0 or 1.1.
- Streamed and complete responses: for 11 seeded requests, and for one seeded
  40 s request, the streamed PCM samples equalled the WAV response for that
  seed.
- Per-request sampling: with `extra_params.temperature` 0, `top_k` 1 or
  `top_p` 0.01 the first codebook of the first frame was the same code in 12
  of 12 unseeded runs. It took two values with `top_k` 2, seven with the
  defaults and twelve with temperature 1.5. Codebooks 1-31 of that frame
  still differed from run to run, as the deploy YAML's depth-decoder sampling
  prescribes.
- Response formats: `wav`, `pcm`, `flac`, `mp3` and `opus` each returned
  1.44 s at 24 kHz for the seeded request above. The SSE stream returned five
  `speech.audio.delta` events and `speech.audio.done`.
- Reference recordings: 120 recordings of 1 to 8 s, cut at and around codec
  frame boundaries, at 8 to 48 kHz and one of them stereo, each returned 200.
- Aborted requests: after closing a stream (one, eight at once, and eight
  with a reference turn) and after client timeouts of a complete response and
  of a reference request, the next seeded request returned the same bytes as
  before, eight concurrent requests completed, and the two stage processes
  held 7,644 + 1,024 MiB against 7,642 + 1,022 MiB before the aborts. With
  counters added to a diagnostic copy of the server, both stages had released
  the state of every aborted request.
- Long responses: streams of 140, 250, 500 and 1,000 frames (11 to 80 s) ran
  at an RTF of 0.25 with first audio after 0.06 s, and their chunks arrived at
  most 0.10 s apart. A 514-word text that fills the context (581 prompt
  tokens and 1,467 output tokens, 2,048 together) returned 200 with
  `x-audio-truncated: true` and 117 s of audio. Eight 500-frame streams at
  once ran at an RTF of 0.36.
- Long texts: a sampled response can leave out a phrase. For a 401-word text
  (453 prompt tokens), transcripts of 300- and 400-frame responses lacked
  three or more consecutive words among the first 60, most often the same six
  words near the 31st, in 24 of 144 served responses and in 14 of 64
  responses of the reference implementation.

Quality was checked by transcription and by the word error rates below. No
MOS, UTMOS or speaker-similarity evaluation was run.

### Standard TTS benchmark

Scope: `benchmarks/tts/bench_tts.py` (it runs `vllm bench serve --omni`), task
`default_voice`, the bundled `seed_tts_smoke` sentences, 20 requests after 2
warm-up requests per level, streamed PCM, default sampling without a seed and
without `max_new_tokens`, so the 10 s default budget applies. The word error
rate is Whisper large-v3 on the same GPU after each run; speaker similarity
and UTMOS were turned off.

```bash
SEED_TTS_SIM_EVAL=0 SEED_TTS_UTMOS_EVAL=0 SEED_TTS_EVAL_DEVICE=cuda:0 \
    python benchmarks/tts/bench_tts.py --model sesame/csm-1b --task default_voice \
    --dataset-path benchmarks/build_dataset/seed_tts_smoke --port 8091 \
    --concurrency 1 4 8 --num-prompts 20 --num-warmups 2 --wer-eval
```

| Concurrency | Completed | RTF, mean / P99 | First audio, median / P99 (ms) | Underrun, mean / P99 (s) | Continuity OK | Audio seconds per wall second | WER, mean / median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 / 20 | 0.27 / 0.28 | 59 / 63 | 0.007 / 0.008 | 100% | 3.71 | 0.060 / 0 |
| 4 | 20 / 20 | 0.35 / 0.37 | 99 / 132 | 0.029 / 0.036 | 100% | 10.79 | 0.050 / 0 |
| 8 | 20 / 20 | 0.37 / 0.39 | 129 / 230 | 0.028 / 0.108 | 95% | 18.25 | 0.065 / 0 |

Underrun is a request's largest buffer deficit under a player that plays at
real-time rate, and "Continuity OK" is the share of requests that stayed below
100 ms. Of the 20 transcripts per level, 14 or 15 matched the text exactly.
Four sentences scored above zero at every level because Whisper wrote their
numbers as digits ("gate 23 at 1115") where the text spells them out. The
other differences were single words. One request at concurrency 8 reached the
10 s budget. Sampling is unseeded, so the error rate varies from run to run.

### 256 sentences, with the reference implementation

Scope: the same benchmark command (`vllm bench serve --omni`) over the first
256 distinct English sentences of the Seed-TTS evaluation list, text only, in
list order, after 16 warm-up requests, with `max_new_tokens` 250. Each level
ran two or three times; the first run of a level was discarded and the table
gives the mean of the others. The word error rate is Whisper large-v3 on the
same GPU.

| Concurrency | Runs kept | RTF, mean / P99 | Per frame (ms) | First audio, median / P99 (ms) | Audio seconds per wall second | WER, mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.27 / 0.34 | 21.9 | 65 / 80 | 3.69 | 0.041, 0.038 |
| 2 | 1 | 0.34 / 0.43 | 27.1 | 85 / 93 | 5.90 | not scored |
| 4 | 2 | 0.36 / 0.46 | 29.1 | 100 / 132 | 11.02 | 0.036 |
| 8 | 2 | 0.39 / 0.50 | 31.0 | 113 / 233 | 20.22 | 0.034 |
| 12 | 1 | 0.60 / 1.25 | 47.7 | 780 / 1,353 | 21.18 | not scored |
| 16 | 1 | 0.80 / 1.97 | 64.3 | 1,450 / 2,111 | 20.64 | not scored |

All 3,840 requests of the 15 runs completed. Between 78.5% and 79.7% of the
transcripts matched the text exactly, and the median error rate was 0.

The reference implementation on the same GPU and the same sentences, one
request at a time, with the same sampling settings and the same 250-frame
limit. Its rows time `generate()` and the codec decode in one process and
return complete audio; the pipeline's rows above are client-side times of
streamed requests.

| Implementation | Sentences | Per frame (ms) | RTF, mean / P99 | Audio seconds per wall second | WER, mean / median | Exact transcripts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformers, FP32 | 256 | 56.6 | 0.73 / 0.78 | 1.39 | 0.049 / 0 | 75.8% |
| Transformers, BF16 | first 32 | 58.7 | 0.75 / 0.81 | 1.33 | 0.026 / 0 | 78.1% |
| Transformers, FP32, static cache (compiled) | first 32 | 52.3 | 0.67 / 0.75 | 1.49 | 0.036 / 0 | 75.0% |
| This pipeline, 1 request | 256 | 21.9 | 0.27 / 0.34 | 3.69 | 0.041, 0.038 / 0 | 78.9%, 79.7% |
| This pipeline, 8 requests | 256 | 31.0 | 0.39 / 0.50 | 20.22 | 0.034 / 0 | 78.5% |

Sentence by sentence, one pipeline run at 1 request and the FP32 reference
run both matched the text exactly for 174 of the 256 sentences, only the
pipeline for 28, only the reference for 20 and neither for 34.

### Memory

- The server's two stage processes, read with `nvidia-smi`: 7,632 MiB after
  startup (6,794 + 838) and 8,214 MiB (7,108 + 1,106) at the end of the
  benchmark, timing and long-response runs, which was also the highest
  reading in a log taken every 0.5 s over those 48 minutes. A server that
  handled the verification requests above, which include a reference
  recording of 29 s, held 8,668 MiB (7,644 + 1,024).
- Reported by the engine at startup: talker weights with the depth decoder's
  graphs 3.28 GiB, the backbone's graphs 0.05 GiB, and a KV cache of 84,992
  tokens; Mimi decoder weights 0.36 GiB. With stage 0 set to
  `enforce_eager: true` the KV cache had 102,400 tokens within the same
  memory share.
- Each stream in flight held up to about 15 MiB of allocated memory in stage
  1: 8 MiB after 2 s of audio, 14 MiB after 8 s and at most 15.5 MiB from 12 s
  on, measured to 32 s. Eight streams at once peaked at 118 MiB together. The
  largest part is the decoder's attention cache, which the stage declares at
  8.1 MiB per request.

### Timing

Scope: packaged deploy config, one warm-up request, streamed PCM, the sentence
"The quick brown fox jumps over the lazy dog near the river." (about 3.2 to
3.6 s of audio), default sampling without a seed, `max_new_tokens` 250. Each
round sent all its requests at once from the same host; a level ran 2 to 4
measured rounds after one discarded round.

| Concurrency | Requests | First audio, median / max (s) | RTF, median / max | Audio seconds per wall second |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.055 / 0.056 | 0.27 / 0.28 | 3.68 |
| 2 | 6 | 0.069 / 0.084 | 0.34 / 0.34 | 5.50 |
| 4 | 12 | 0.128 / 0.130 | 0.36 / 0.37 | 10.27 |
| 8 | 24 | 0.224 / 0.247 | 0.38 / 0.40 | 18.81 |
| 12 | 24 | 0.235 / 1.517 | 0.39 / 0.83 | 15.24 |
| 16 | 32 | 0.731 / 2.070 | 0.44 / 0.84 | 16.58 |

First audio is the time to the first byte read by the client. RTF is a
request's wall time divided by the duration of its audio, so values above 1
are slower than real time. The last column divides the audio of a level's
rounds by their wall time, from a round's first request to the end of its
last, so it depends on each round's longest request. All 102 requests
returned 200. A second run of the same script differed by at most 0.003 in
median RTF up to 12 concurrent requests and 0.055 at 16 and, in median first
audio, by at most 0.008 s. Probes of 20 to 22 single requests at the start,
in the middle and at the end of the 48 minutes of timed runs had median RTFs
of 0.272, 0.272 and 0.270. With an 8 s reference recording in the request,
median first audio was 0.069 s with one request and 0.344 s with eight.

With stage 0 set to `enforce_eager: true` on the same host, the RTF was 0.58
with one request and 0.61 with eight, and first audio took 0.058 s and
0.160 s. The depth decoder's frame step alone took 36 ms per frame eagerly
and 16 ms from its graph (37 to 38 ms and 20 ms for batches of 2 to 8). The
eager figures depend on the host CPU.

## Notes

- Memory usage: about 8 GiB of 24 GiB; see [Memory](#memory).
- Key flags: `--omni` is required. No other flag was used.
- Known limitations:
    - Stage 0 runs 8 requests at a time. Further requests wait: the longest
      wait for first audio was 1.5 s at 12 concurrent requests and 2.1 s at
      16, where the slowest request took 0.84 s per second of audio. In the
      256-sentence benchmark the mean RTF was 0.60 at 12 and 0.80 at 16, with
      a P99 of 1.25 and 1.97.
    - A long text can lose a phrase in a sampled response; see
      [Correctness](#correctness). The reference implementation shows the
      same behavior.
    - The depth decoder's graphs record the deploy YAML's depth sampling.
      Setting stage 0 back to `enforce_eager: true` turns its graphs off.
      Stage 1 runs eagerly: a decode call takes about 3.5 ms whatever its
      chunk size.
    - The deploy config's batch sizes, memory shares and chunk sizes are
      starting values and have not been tuned.
    - Some outputs clip. The reference implementation's own floating-point
      waveform exceeded full scale for two of the six compared prompts
      (peaks 1.69 and 2.39), and the server clips to the 16-bit range.
    - Not qualified: other accelerators, text in other languages, streamed
      `wav`, `speed`, the WebSocket text-input endpoint,
      `/v1/audio/speech/batch`, offline `Omni` inference and long-running
      load.

## Supported features

| Feature | Status on this profile | Guide |
| --- | --- | --- |
| Complete responses (`wav`, `pcm`, `flac`, `mp3`, `opus`) | supported | [Speech API](../../docs/serving/speech_api.md#response-format) |
| Streamed PCM, raw bytes or SSE | supported; first audio after about 0.06 s, generation faster than real time | [Speech API](../../docs/serving/speech_api.md#response-format) |
| Reference turn (`ref_audio` + `ref_text`) | supported, one recording of 1 to 30 s | [CSM-1B fields](../../docs/serving/speech_api.md#csm-1b) |
| Uploaded voice (`POST /v1/audio/voices`, then `voice`) | supported when the voice was uploaded with its transcript; without one the request returns 400 | [Voices endpoint](../../docs/serving/speech_api.md#voices-endpoint) |
| Preset voices, `instructions`, `language`, `sample_rate`, `speaker_embedding`, `ref_audio_2`, `x_vector_only_mode` | not supported; the fields return 400 and an unknown `voice` name is ignored | [CSM-1B fields](../../docs/serving/speech_api.md#csm-1b) |
| Per-request sampling (`extra_params.temperature`, `top_k`, `top_p`) | first codebook only | [CSM-1B fields](../../docs/serving/speech_api.md#csm-1b) |
| `seed` | supported; the same request returned the same bytes on the same server | [CSM-1B fields](../../docs/serving/speech_api.md#csm-1b) |
| Concurrent requests | 8 in flight, further requests wait | — |
| Truncation report (`x-audio-truncated`) | supported | [Truncated audio](../../docs/serving/speech_api.md#truncated-audio) |
| CUDA graphs | stage 0: backbone decode steps and the depth decoder's frame step; stage 1 runs eagerly (`enforce_eager: true`) | — |
| Prefix caching | disabled in both stages; stage 0 does not start with it enabled | — |
| `--no-async-chunk` | not supported | — |
| WebSocket text input, batch endpoint, offline `Omni` inference | not qualified | — |
