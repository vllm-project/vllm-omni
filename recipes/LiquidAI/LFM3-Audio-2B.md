# LFM3-Audio-2B on CUDA

## Model and execution

[LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
accepts text and 16 kHz audio and returns text, speech, or interleaved text and
speech. It uses the [LFM3-Omni](./LFM3-Omni-2B.md) integration: the temporal
model and MaskGIT depth decoder run in one autoregressive stage, and a second
stage runs LiquidCodec and returns 24 kHz audio. Text-only deployment omits the
codec stage. Only the speech encoder and the checkpoint's module names differ
from LFM3-Omni; there is no image or video input.

The checkpoint contains custom Transformers code and requires
`--trust-remote-code`. The reference used during integration is revision
`2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1` with Transformers 5.20.2 and
`attn_implementation="eager"`. Serving runs with Transformers 5.19.0.

## Deployment

Validated on one NVIDIA RTX 4090 (24 GB). The weights take 3.69 GiB.

Speech output, including TTS and interleaved speech-to-speech:

```bash
vllm serve LiquidAI/LFM3-Audio-2B --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/lfm3_audio.yaml --async-chunk --port 8091
```

```bash
curl http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"LiquidAI/LFM3-Audio-2B","input":"Hello from LFM3.","instructions":"Perform TTS in American English using a female voice.","response_format":"wav"}' \
  --output speech.wav
```

Text output, for ASR and audio understanding:

```bash
vllm serve LiquidAI/LFM3-Audio-2B --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/lfm3_audio_text.yaml --port 8091
```

Send the clip to `/v1/chat/completions` as an `audio_url` content part with a
system prompt such as `Perform ASR in English.`. Add
`"vllm_xargs": {"generation_mode": "text"}` to keep the reply in text; this
matches the reference's `suppress_mixed` and `suppress_audio_out`.

## Configuration and limits

The LFM3-Omni runtime limits apply: tensor and pipeline parallel size one,
eager execution, and no prefix caching, chunked prefill, or recompute
preemption. Both stages share device 0, with at most four concurrent requests
and a 4,096-token context.

- Speech settings that a request omits come from the checkpoint's
  `generation_config.json`: audio temperature 1.0, top-k 128, repetition
  penalty 1.1, and a 4,096-frame repetition window. The deploy profile sets the
  same values. `SamplingParams.extra_args` overrides them per request.
- On CUDA, the width-7 convolutions use the checkpoint's own Triton kernels,
  with its seven-input cache, as the reference does.
- Concurrent requests change the backbone GEMM shapes. Their greedy speech codes
  can therefore differ from a single request's. With `VLLM_BATCH_INVARIANT=1`,
  each request in a mixed concurrent batch reproduced its single-request output.
- Reference-audio voice cloning (`audio_out` prompts), live transcription,
  quantization, and distributed execution are outside this integration's scope.

## Validation

All comparisons used the pinned reference, seed 42, and identical prompts and
16 kHz input samples:

| Case | Text tokens | Speech |
| --- | --- | --- |
| ASR, audio question answering | Identical | n/a |
| Greedy TTS (42, 25, and 24 frames) | Identical | Bitwise-identical waveform |
| Sampled TTS, temperature 1.0, top-k 128 (39 frames) | Identical | Bitwise-identical waveform |
| Greedy interleaved speech-to-speech (55 frames) | Identical | Bitwise-identical waveform |

- These results held for both full payloads and `--async-chunk` streaming. The
  streaming comparison used the reference's four-frame streaming decode.
- ASR and audio question answering through `/v1/chat/completions` returned the
  reference text.
- In batch-invariant mode, a concurrent mix of eight speech, two sampled, and
  four text requests matched the single-request outputs.

When a stream's frame count is a multiple of four, its last chunk ends at a codec
boundary. The final empty payload then does not flush the overlap-add tail,
and 480 samples (20 ms) are missing. This is an open LFM3 streaming issue
shared with LFM3-Omni.

## Measured performance

These are smoke measurements on one RTX 4090 for one English sentence
(`benchmarks/lfm3_omni/benchmark.py --deploy-config
vllm_omni/deploy/lfm3_audio.yaml`). Setup: BF16, four-frame streaming codec
chunks, warmed engine, and three measured batches per concurrency. Timing
covers the offline Omni API, including orchestration and the codec.

| Audio temperature | Concurrency | First audio, s | Latency, s | RTF | Requests/s |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 1 | 0.252 | 2.404 | 0.715 | 0.416 |
| 0.0 | 2 | 0.391 | 3.330 | 1.095 | 0.601 |
| 0.0 | 4 | 0.651 | 6.088 | 1.845 | 0.656 |
| 1.0 | 1 | 0.261 | 2.364 | 0.758 | 0.423 |
| 1.0 | 2 | 0.393 | 3.369 | 1.144 | 0.581 |
| 1.0 | 4 | 0.681 | 5.785 | 1.940 | 0.686 |

These numbers describe the eager bring-up runtime. They do not show its
limits for longer speech or other hardware.
