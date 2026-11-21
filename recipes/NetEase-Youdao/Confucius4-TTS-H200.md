# Confucius4-TTS for voice-cloned TTS on H200

## Summary

- Vendor: NetEase Youdao
- Model: `netease-youdao/Confucius4-TTS`
- Task: Multilingual text-to-speech with voice cloning
- Mode: Online serving with `/v1/audio/speech`, or offline with `Omni`
- Hardware: 1x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Use this recipe to run Confucius4-TTS as a two-stage voice-cloning pipeline.
The native vLLM talker generates semantic tokens. Stage 1 then recomputes the
aligned latents from the finished token sequence in float32, exactly as the
canonical implementation does, and an S2A diffusion model and BigVGAN
synthesize 22.05 kHz mono audio. Every request
requires exactly one reference audio clip.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | Text plus exactly one reference audio clip |
| Languages | `zh`, `en`, `ja`, `ko`, `de`, `fr`, `es`, `id`, `it`, `th`, `pt`, `ru`, `ms`, `vi` |
| Output | 22.05 kHz mono audio |
| Streaming | Not supported; use `stream=false` |
| Reference audio | Exactly one clip, truncated to at most 30 seconds |
| Prompt limit | 520 formatted tokens, including the language instruction |
| Semantic limit | At most 1518 output semantic tokens, about 30 seconds of audio |
| Deployment | Both stages share one H200; this recipe is qualified for single-request use |

The integration does not normalize or segment long input automatically. The
model emits about 50 semantic tokens per second of audio, so one request yields
at most about 30 seconds. Longer text is cut off without an error; the stage
processor logs a truncation warning when Stage 0 stops on the token budget
instead of EOS. Split long text into suitable utterances before sending
requests.

## References

- Canonical repository:
  [netease-youdao/Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS)
- Model weights:
  [netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS)
- Online serving example:
  [`examples/online_serving/text_to_speech/README.md#confucius4-tts`](../../examples/online_serving/text_to_speech/README.md#confucius4-tts)
- Offline inference example:
  [`examples/offline_inference/text_to_speech/README.md#confucius4-tts`](../../examples/offline_inference/text_to_speech/README.md#confucius4-tts)
- Deploy config:
  [`vllm_omni/deploy/confucius4_tts.yaml`](../../vllm_omni/deploy/confucius4_tts.yaml)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141GB
- Number of devices: 1
- Qualification scope: offline generation and OpenAI-compatible speech serving

## Software requirements

- Python: `>=3.11,<3.14`, as declared by the current project
- vLLM: `0.34.0`, the version the current checkout requires. vLLM-Omni now runs
  every LLM stage on Model Runner V2, and both stages of this pipeline declare
  `supports_native_mrv2_data_plane`.
- vLLM-Omni: install or load the current checkout rather than pinning the
  original qualification commit

Everything in the Verification and Latency sections below was run on one H200
with Python 3.12.13, PyTorch 2.14.0 (CUDA 13.0), transformers 5.22.2 and the
released vLLM 0.34.0 wheel, with this integration installed from the pull
request tree rebased on upstream `main` at `dc8214f41`. The canonical
implementation was run in its own environment (see its comparison below).

## Model layout

On a connected host, the auxiliary models can be downloaded from their
original repositories on first use. For an offline host, assemble a
self-contained directory:

```text
Confucius4-TTS/
├── config.json
├── configuration.json
├── s2a_model.pt
├── t2s_model.safetensors
├── wav2vec2bert_stats.pt
├── tokenizer.json
├── tokenizer.model
├── tokenizer_config.json
├── pretrained/
│   ├── w2v-bert-2.0/
│   └── campplus/
│       └── campplus_cn_common.bin
└── bigvgan/
```

One way to prepare it is:

```bash
MODEL_DIR=/models/Confucius4-TTS
hf download netease-youdao/Confucius4-TTS --local-dir "$MODEL_DIR"
hf download facebook/w2v-bert-2.0 \
  --local-dir "$MODEL_DIR/pretrained/w2v-bert-2.0"
hf download funasr/campplus campplus_cn_common.bin \
  --local-dir "$MODEL_DIR/pretrained/campplus"
hf download nvidia/bigvgan_v2_22khz_80band_256x \
  --local-dir "$MODEL_DIR/bigvgan"
```

## Command

Serve the Hugging Face model:

```bash
vllm serve netease-youdao/Confucius4-TTS \
  --omni \
  --port 8091
```

For an offline deployment, replace the model ID with `MODEL_DIR`. The packaged
`vllm_omni/deploy/confucius4_tts.yaml` is selected automatically.

## Verification

The server accepts a language through either `language` or
`extra_params.lang`:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "netease-youdao/Confucius4-TTS",
    "input": "Hello, this is a Confucius4 text to speech test.",
    "stream": false,
    "response_format": "wav",
    "ref_audio": "data:audio/wav;base64,<BASE64_ENCODED_REFERENCE_AUDIO>",
    "extra_params": {"lang": "en"}
  }' \
  --output confucius4.wav
```

Validation of the current revision, all on the stack described above:

- `tests/e2e/online_serving/test_confucius4_tts_expansion.py` passed (2 tests)
  against a real `vllm serve --omni` server with the packaged deploy config. A
  non-streaming English request returned a WAV of more than 1 KiB at 22050 Hz,
  and a `max_new_tokens=1` request returned within a 60 s client timeout. Such
  a request ends on Stage 0's prefill step; when its conditioning tensors were
  only published on the first decode step, Stage 1 waited for the 600 s
  stage-input deadline.
- The 8 quality prompts and 40 tail-word seeds below ran through the offline
  `Omni.generate()` path without a failure.
- The 218 Confucius4-related unit tests passed (`tests/model_executor/models/confucius4_tts`,
  the stage-input processor, extras, adapter, TTS detection and IndexTTS2 codec
  tests).

On November 19, 2026, the current safe-deserialization path was checked against
the packaged 398 MiB `s2a_model.pt`. Loading it on CPU with
`torch.load(..., weights_only=True)` succeeded and returned 274 tensor entries,
and the same file loaded through the engine on H200 in the end-to-end runs
above.

## Notes

- Stage 0 and Stage 1 share GPU 0. The packaged config reserves 0.50 and 0.35
  GPU memory utilization respectively.
- Both stages set `max_num_seqs: 1`. Treat this as a single-request functional
  configuration; concurrent throughput and latency have not been qualified.
- Stage 0 leaves `dtype` unspecified so vLLM selects it automatically
  (bfloat16 for this checkpoint); Stage 1 remains in `float32`. Forcing Stage 0
  to `float32` or `float16` did not change the tail-word behavior in the
  comparisons below, so the packaged config does not force a precision.
- Stage 1 uses 25 diffusion steps and an inference CFG rate of 0.7.
- Stage 1 also holds a float32 copy of the talker backbone (about 2.6 GB) and
  runs one extra full-sequence forward per request to rebuild the latent. The
  canonical implementation does the same, which keeps the latent independent
  of Stage 0 precision and of decode-step bookkeeping. Stage 0 therefore
  budgets exactly the public 1518 semantic tokens and publishes no hidden
  states.
- The talker's `forward()` returns the hidden-state tensor and builds its
  multimodal payload afterwards in `make_omni_output`, so it declares
  `mrv2_forward_returns_tensor = True`. Without that declaration MRv2 treats it
  as a structured-output model, drops FULL CUDA graph capture, and Stage 0
  falls back to PIECEWISE graphs: single-request latency was about twice as
  high (1.5 s instead of 0.9 s for the first prompt below, 3.8 s instead of
  1.9 s for the last). Both stages also declare `supports_native_mrv2_data_plane`;
  without it the stages use the legacy transport and the request never leaves
  Stage 0.
- Stage 0 runs with the default async scheduling. A request that stops on its
  token budget (a 160-word input) and a `max_new_tokens=1` request both
  completed with async scheduling on and off, with identical audio lengths and
  latencies within 1%.
- Stage 0 samples from the unmodified logits, including BOS, and strips BOS
  afterwards as the canonical implementation does. It keeps the stored
  semantic position embedding row 0, which the BOS prefix embedding includes.
- The reference clip is trimmed to 30 seconds, and its 16 kHz and 22.05 kHz
  signals are resampled independently from the source in float32, matching the
  canonical front-end.
- The first request can be slower because reference encoders and auxiliary
  models are loaded and cached.

### Prompt format and transformers versions

The checkpoint was trained on prompts of the form `<s> ... </s>`. Whether the
tokenizer appends `</s>` for this checkpoint's `tokenizer_config.json`
(`add_eos_token: true`) depends on the transformers release: 4.x does, 5.x does
not. A prompt without `</s>` is one token shorter, so the semantic BOS and every
later position shift by one, and the talker drifts away from the canonical
distribution towards the end of the utterance. The integration therefore adds
`<s>` and `</s>` explicitly and does not rely on the library default. Any code
that builds Confucius4 prompts itself must do the same.

### Comparison with the canonical implementation

On one H200 with the packaged config, the prompt `战斗战斗，只做第一` was
synthesized for 40 seeds (2000 to 2039) with one reference clip, and the output
was transcribed with Whisper-tiny with the language fixed to Chinese. A
transcript counts when it ends with 第一 after stripping whitespace and
punctuation:

| Implementation | Transcript ends with 第一 |
| --- | --- |
| Canonical (beam 1) | 28 / 40 |
| This integration, before restoring `</s>` | 0 / 10 |
| This integration, after restoring `</s>`, earlier revision on vLLM 0.33.0 | 36 / 40 |
| This integration, current revision, vLLM 0.34.0 with Model Runner V2 | 31 / 40 |

Across the whole 8-prompt quality set Whisper-tiny's micro error rate on the
current revision was 18% for English and 21% for Chinese. Whisper-tiny is a
coarse intelligibility proxy. Only one sentence and one reference voice were
checked, and with 40 samples a hit rate has a standard deviation of about 2.5
files, so 36 and 31 are not distinguishable and neither is a firm difference
from the canonical 28. This does not establish general quality or latency
parity.

### Latency on H200

Offline `Omni.generate()` with the packaged deploy config on one H200, one
reference clip, seed 42, one warm-up request, then 5 repetitions per prompt.
Latency is wall time per request; `stage_0` and `stage_1` are the per-stage
generation times reported in `stage_durations`. Stage 0 includes the reference
front-end (Wav2Vec2-BERT, CAMPPlus, reference mel) and semantic decoding;
Stage 1 includes the latent recompute, 25 S2A diffusion steps and BigVGAN.

| Prompt | Audio | Latency (median, min–max) | RTF | Stage 0 | Stage 1 |
| --- | --- | --- | --- | --- | --- |
| `en`, 9 words | 3.67 s | 0.90 s (0.89–0.92) | 0.25 | 553 ms | 343 ms |
| `zh`, 9 characters | 1.89 s | 0.70 s (0.68–0.76) | 0.37 | 350 ms | 344 ms |
| `zh`, 3 sentences | 11.33 s | 1.93 s (1.92–1.95) | 0.17 | 1457 ms | 468 ms |
| `en`, 2 sentences | 10.98 s | 1.87 s (1.85–1.88) | 0.17 | 1411 ms | 454 ms |

A request with `max_new_tokens=1` returned in 0.50 s, and a 160-word English
input stopped on the 1518-token budget, returned 30.3 s of audio in 4.9 s and
logged the truncation warning. Peak GPU memory as reported by `nvidia-smi` was
about 80 GiB, which is dominated by the two stages' `gpu_memory_utilization`
reservations rather than by the model.

### Latency against the canonical implementation

The canonical `ConfuciusTTSVLLM` path (official repository
`0267ef80fa29366dfa5a18edd93f0771e145f8ed`, `example_vllm.py` flow) was run on
the same H200 with the same reference clip, the same four prompts, and the same
sampling settings (temperature 0.8, top-p 0.8, top-k 30, repetition penalty
10.0, 25 diffusion steps, CFG 0.7). Text normalization and segmentation were
turned off (`raw=True`, one segment) so both sides synthesize the same text. It
ran in its own environment, as the official requirements demand: Python 3.10,
vLLM 0.16.0, PyTorch 2.9.1, transformers 4.57.6, `gpu_memory_utilization` 0.4.
One warm-up request, then 5 repetitions per prompt; per-component times are
medians with CUDA synchronization around each call.

| Prompt | vLLM-Omni latency (RTF) | Canonical latency (RTF) |
| --- | --- | --- |
| `en`, 9 words | 0.90 s (0.25) | 1.01 s (0.27) |
| `zh`, 9 characters | 0.70 s (0.37) | 0.93 s (0.39) |
| `zh`, 3 sentences | 1.93 s (0.17) | 1.62 s (0.17) |
| `en`, 2 sentences | 1.87 s (0.17) | 1.64 s (0.16) |

Canonical per-component medians, in ms, for the first and last prompts:

| Prompt | Semantic front-end | Semantic decode | Latent recompute | S2A | BigVGAN |
| --- | --- | --- | --- | --- | --- |
| `en`, 9 words | 152 | 368 | 17 | 324 | 138 |
| `en`, 2 sentences | 155 | 934 | 34 | 359 | 127 |

Reading this table:

- Short prompts are at parity or faster on vLLM-Omni; the long prompts are
  within about 15% of the canonical latency (RTF 0.17 against 0.16 to 0.17).
  After removing the roughly 150 ms front-end, vLLM-Omni's Stage 0 spends about
  2.3 ms per semantic token on the longest prompt against about 1.9 ms in the
  canonical path. The cause has not been investigated; the stacks also differ
  (vLLM 0.34 and PyTorch 2.14 against vLLM 0.16 and PyTorch 2.9, bfloat16
  against float16 for the talker), so the gap cannot be attributed to the
  integration alone.
- The two sides do not synthesize identical audio lengths, because the canonical
  path does not take a per-request seed. RTF is therefore the fair column; the
  canonical audio lengths were 3.73 s, 2.41 s, 10.05 s and 8.42 s against 3.67 s,
  1.89 s, 11.33 s and 10.98 s here.
- vLLM-Omni reserves more GPU memory (about 80 GiB `nvidia-smi` peak against
  about 66 GiB), because its two stages hold 0.50 and 0.35 of the device while
  the canonical path holds 0.40 for one engine plus the S2A model.
- This is single-request latency only. Concurrent throughput was not measured
  for either side.

## Supported features

| Feature | Status | Notes |
| --- | --- | --- |
| Offline `Omni.generate()` | Supported | Requires text and one reference clip |
| OpenAI-compatible speech API | Supported | Non-streaming WAV verified on H200 |
| Uploaded audio voices | Supported | Alternative to inline `ref_audio` |
| Async chunk streaming | Not supported | [`async_chunk`](../../docs/design/feature/async_chunk.md) is disabled |
| Multi-GPU execution | Not qualified | This recipe covers one H200 only |
| Concurrent serving | Not qualified | Packaged stages use `max_num_seqs: 1` |
