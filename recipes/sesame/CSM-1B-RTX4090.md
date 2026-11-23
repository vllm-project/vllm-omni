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
runs unmodified and uses about 8 GiB of the card. A request takes about 0.3 to
0.4 s of wall time per second of audio with up to 8 requests in flight, so
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

The first launch downloads the whole repository, about 19 GB (374 s on the
measured host). That includes two files in the original checkpoint format,
`ckpt.pt` and `model.safetensors` at 6.2 GB each, which the pipeline does not
read; it loads the two `transformers-*.safetensors` shards.

## Hardware

- Accelerator model and per-device memory: NVIDIA GeForce RTX 4090, 24,564 MiB
- Number of devices: 1
- Device interconnect: not applicable, single device
- Host memory: not material, no CPU offload
- Qualification scope: runtime-qualified for online serving with the packaged
  deploy config, English text, 1 to 16 concurrent requests

Other accelerators are not covered by this recipe. The weekly CI step targets
an L4, and it had not run when this recipe was written.

## Software environment

- OS: Ubuntu 24.04.3 LTS, kernel 6.8.0-149-generic, in a container started
  from `vllm/vllm-openai:v0.34.0`
- Python: 3.12.3
- Driver / runtime: NVIDIA driver 615.71.09, CUDA 13.0 (PyTorch 2.14.0+cu130)
- vLLM version: 0.34.0 (Transformers 5.22.0)
- vLLM-Omni version or commit: `05552893ce107e2e8c79241edc0f41fe9dbacec5`, run from source
- Model runner: V2

## Command

```bash
vllm serve sesame/csm-1b --omni --port 8091
```

The model selects the packaged deploy config, so `--deploy-config` is not
needed. With the weights cached the server was ready after about 100 to 155 s,
which includes capturing the graphs.
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
GPU or attention backend, and it gave a different one (1.20 s) before CUDA
graphs were enabled.

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
The first bytes arrived after about 0.08 s. Without a seed the length varies
from run to run; three runs returned 1.28 to 1.52 s in 0.41 to 0.49 s.

The end-to-end test, with the command of the weekly CI step:

```bash
VLLM_OMNI_CI_MODEL_RUNNER=v2 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    pytest -s -v tests/e2e/online_serving/test_csm.py -m "slow and L4 and tts" --run-level "full_model"
```

Expected: 5 passed. The test starts its own server. The `L4` marker names the
CI card; the run recorded below was on the RTX 4090.

## Qualification evidence

Everything below was measured on 2026-11-22 and 2026-11-23 (UTC) on the
hardware and software above, at the commit above unless an item names an
earlier revision. The earlier revisions are this change before it was rebased
onto main `4dfc20301`: `6e4427a0`, where both stages still ran eagerly, and the
first revision with the CUDA graphs. Their items were not repeated. These are
bounded runs.

### Correctness

- End-to-end test: 5 passed in 824 s. It covers a text request, a streamed PCM
  request, two concurrent requests, `speaker_id` 1 and a reference turn.
  Whisper small transcripts scored 1.000 against the input for four of the six
  responses, 0.915 for one of the concurrent requests and 0.930 for the
  `speaker_id` 1 request (the test requires more than 0.9). The streamed PCM
  scored 9.6 dB on the shared harmonic-to-noise check, whose floor is 1.0 dB;
  33 seeded streams of three sentences scored 3.6 to 18.1 dB.
- Sampled output across seeds: the five requests of the test with seeds 0 to 9
  each, 50 requests with default sampling. No response was truncated. With
  Whisper small, 49 transcripts scored above 0.9 and one scored 0.729; at the
  first revision with the CUDA graphs, where the same request gave the same
  transcript, Whisper large-v3, which the test falls back to, scored it 1.000.
  An earlier sentence for the streaming test scored below 0.9 for 2 of 10
  seeds and was replaced, so intelligibility depends on the sentence and on
  the sample.
- Greedy comparison with Transformers 5.22.0
  (`CsmForConditionalGeneration.generate`): four text prompts and one prompt
  with a reference recording. All 336 compared frames (10,752 codes) were
  identical and the waveforms differed by at most 1.09 steps of 16-bit PCM;
  the codes also equalled those of the earlier revisions, eager and with the
  graphs. This used a diagnostic configuration, not the packaged profile: FP32
  talker on the `FLEX_ATTENTION` backend
  (`--stage-overrides '{"0":{"attention_backend":"FLEX_ATTENTION"}}'`),
  sampling turned off in the request and in the deploy YAML, and cuDNN TF32
  turned off on both sides. In BF16 the
  codes differ from the reference implementation's BF16 output from the first
  frame, as that implementation's own BF16 and FP32 outputs do, so the
  packaged profile rests on the transcript checks above.
- Streamed and complete responses: for 11 seeded requests the streamed PCM
  samples equalled the WAV response for that seed.
- Per-request sampling (at `6e4427a0`): with `extra_params.temperature` 0 or
  `top_k` 1 the first codebook of the first frame was the same code in 12 of
  12 unseeded runs, and with `top_k` 2 it took two values. Codebooks 1-31 of
  that frame still differed from run to run, as the deploy YAML's
  depth-decoder sampling prescribes.
- Response formats: `wav`, `pcm`, `flac`, `mp3` and `opus` each returned
  1.44 s at 24 kHz for the seeded request above. The SSE stream returned five
  `speech.audio.delta` events and `speech.audio.done`.
- Aborted requests: after closing a stream (one, eight at once, and eight
  with a reference turn) and after client timeouts of a complete response and
  of a reference request, the next seeded request returned the same bytes as
  before, eight concurrent requests completed, and the two stage processes
  held 7,084 + 1,034 MiB against 7,082 + 1,034 MiB before. At `6e4427a0`,
  with counters added to the server, both stages had released the state of
  every aborted request.

Quality was checked by transcription and by the word error rate below. No MOS,
UTMOS or speaker-similarity evaluation was run.

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
| 1 | 20 / 20 | 0.27 / 0.28 | 81 / 87 | 0.009 / 0.014 | 100% | 3.66 | 0.050 / 0 |
| 4 | 20 / 20 | 0.37 / 0.39 | 122 / 176 | 0.033 / 0.076 | 100% | 10.48 | 0.077 / 0 |
| 8 | 20 / 20 | 0.40 / 0.44 | 163 / 327 | 0.061 / 0.198 | 95% | 17.79 | 0.065 / 0 |

Underrun is a request's largest buffer deficit under a player that plays at
real-time rate, and "Continuity OK" is the share of requests that stayed below
100 ms. Of the 20 transcripts per level, 14 or 15 matched the text exactly.
Four sentences scored above zero at every level because Whisper wrote their
numbers as digits ("gate 23 at 1115") where the text spells them out. One
request at concurrency 4 reached the 10 s budget and lost its last words. The
other differences were single words. Sampling is unseeded, so the error rate
varies from run to run: the same benchmark one commit earlier gave 0.065,
0.064 and 0.060.

### Memory

- The server's two stage processes, read with `nvidia-smi`: 7,672 MiB after
  startup (6,830 + 842), 8,116 MiB (7,082 + 1,034) after the verification
  requests above, which include reference recordings, and the timing runs
  below, and 8,122 MiB (7,084 + 1,038) after the benchmark. The whole-device
  reading rose by at most 68 MiB during the timing runs; it was taken once
  per round, so short peaks can be missed.
- Reported by the engine at startup: talker weights with the depth decoder's
  graphs 3.28 GiB, the backbone's graphs 0.05 GiB, and a 2.62 GiB KV cache
  (86,016 tokens); Mimi decoder weights 0.36 GiB. Before the graphs the KV
  cache had 102,400 tokens within the same memory share.
- Each request in flight held about 6.9 MiB of decoder state in stage 1 (short
  requests, at `6e4427a0`). The decoder's attention cache, the largest part,
  is bounded at 8.1 MiB per request and reaches that after 10 s of audio.

### Timing

Scope: packaged deploy config, one warm-up request, streamed PCM, the sentence
"The quick brown fox jumps over the lazy dog near the river." (about 3.2 to
3.6 s of audio), default sampling without a seed, `max_new_tokens` 250. Each
round sent all its requests at once from the same host; a level ran 2 to 4
rounds. A second, idle server held 8.1 GiB of the same GPU and received no
request during the runs.

| Concurrency | Requests | First audio, median / max (s) | RTF, median / max | Audio seconds per wall second |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.086 / 0.091 | 0.28 / 0.28 | 2.27 |
| 2 | 6 | 0.103 / 0.125 | 0.35 / 0.36 | 4.75 |
| 4 | 12 | 0.166 / 0.178 | 0.38 / 0.40 | 8.94 |
| 8 | 24 | 0.304 / 0.322 | 0.41 / 0.43 | 16.93 |
| 12 | 24 | 0.355 / 1.600 | 0.43 / 0.82 | 15.23 |
| 16 | 32 | 0.776 / 2.109 | 0.55 / 0.91 | 18.61 |

First audio is the time to the first byte read by the client. RTF is a
request's wall time divided by the duration of its audio, so values above 1
are slower than real time. The last column divides a round's audio by the
round's wall time, so it depends on the round's longest request. All 102
requests returned 200. A second run of the same script differed by at most
0.01 in median RTF up to 12 concurrent requests and 0.04 at 16 and, in median
first audio, by at most 0.003 s up to 8 concurrent requests, 0.04 s at 12 and
0.05 s at 16. Two runs one commit earlier measured the same median RTF and a
median first audio of 0.072 and 0.081 s with one request.

With both stages eager (`6e4427a0`, the same script and GPU, no other
process), the RTF was 1.76 with one request and 2.03 with eight, and first
audio took 0.151 s and 0.375 s. The depth decoder's frame step alone took
about 127 ms per frame eagerly and 16 ms from its graph (20 ms for batches of
2 to 8), measured at the first revision with the graphs.

## Notes

- Memory usage: about 8 GiB of 24 GiB; see [Memory](#memory).
- Key flags: `--omni` is required. No other flag was used.
- Known limitations:
    - Stage 0 runs 8 requests at a time. Further requests wait: the longest
      wait for first audio was 1.6 s at 12 concurrent requests and 2.1 s at
      16, where the slowest request took 0.91 s per second of audio (1.04 in
      the second run).
    - The depth decoder's graphs record the deploy YAML's depth sampling.
      Setting stage 0 back to `enforce_eager: true` turns its graphs off.
      Stage 1 runs eagerly: a decode call takes about 12 ms whatever its
      chunk size.
    - The deploy config's batch sizes, memory shares and chunk sizes are
      starting values and have not been tuned.
    - Some outputs clip. The reference implementation's own floating-point
      waveform exceeded full scale for two of the five compared prompts
      (peaks 1.69 and 2.39), and the server clips to the 16-bit range.
    - Not qualified: other accelerators, text in other languages, streamed
      `wav`, `speed`, the WebSocket text-input endpoint,
      `/v1/audio/speech/batch`, offline `Omni` inference and long-running
      load.

## Supported features

| Feature | Status on this profile | Guide |
| --- | --- | --- |
| Complete responses (`wav`, `pcm`, `flac`, `mp3`, `opus`) | supported | [Speech API](../../docs/serving/speech_api.md#response-format) |
| Streamed PCM, raw bytes or SSE | supported; first audio after about 0.08 s, generation faster than real time | [Speech API](../../docs/serving/speech_api.md#response-format) |
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
