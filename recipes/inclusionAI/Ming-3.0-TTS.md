# Ming-3.0-TTS Tiny

> Voice cloning, speech control, voice design, and multi-speaker Podcast.

## Summary

- Vendor: inclusionAI
- Model: [`inclusionAI/Ming-3.0-tts-tiny`](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny)
- Runtime: vLLM-Omni single-stage serving with Model Runner V2
- API: OpenAI-compatible `/v1/audio/speech`

## Architecture

The default pipeline runs the Ling V3 MoE backbone (18 KDA / 6 MLA layers),
Flow decoder and AudioVAE in one stage. Model Runner V2 keeps request state
on the device and returns streaming PCM directly. The default deployment is
[ming3_tts.yaml](../../vllm_omni/deploy/ming3_tts.yaml).

## CUDA

### Environment

- OS: Linux
- Python: 3.12.3
- CUDA: 13.0
- PyTorch: 2.13.0+cu130; Transformers: 5.19.0
- vLLM version: 0.32.0
- GPU: 1x H100 / H200

### Reference audio

Run from the repository root. These examples use two
[Seed-TTS evaluation](https://github.com/BytedanceSpeech/seed-tts-eval) clips from
a [pinned HF mirror](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval/tree/8f5e1aa2a35d42f42e940074c1983358b9491f89).
Download both WAVs before starting the server; requests use the local copies.
Commands require `curl` and `jq`.

```bash
REF_DIR="$PWD/.cache/ming3_tts/references"
BASE=https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval/resolve/8f5e1aa2a35d42f42e940074c1983358b9491f89/zh/prompt-wavs
mkdir -p "$REF_DIR"
curl --fail --location --show-error "$BASE/10002287-00000094.wav" --output "$REF_DIR/speaker_1.wav"
curl --fail --location --show-error "$BASE/10002290-00000094.wav" --output "$REF_DIR/speaker_2.wav"
```

| File | Exact transcript |
| --- | --- |
| `speaker_1.wav` | 在此奉劝大家别乱打美白针。 |
| `speaker_2.wav` | 对此，乔宝云认为有夸大的成分。 |

### Commands

Start the server in one terminal using the uv `.venv`. The HF ID selects the
pipeline automatically. Run the request examples in another terminal from the
same repository root.

```bash
MODEL=inclusionAI/Ming-3.0-tts-tiny
REF_DIR="$PWD/.cache/ming3_tts/references"
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve "$MODEL" \
  --omni --port 8000 --allowed-local-media-path "$REF_DIR"
```

Use `--revision <HF-commit>` to pin a checkpoint. The HF model weights download
automatically on first launch.

### Default execution

The command above selects the fused Model Runner V2 deployment automatically:
one stage, up to 64 scheduled sequences, and a 4096-token context limit.
No separate throughput profile or deployment argument is required.

Under load, new arrivals are admitted together when at least 32 requests are
running: admission resumes when eight requests wait or the oldest has waited
150 ms. This wait contributes to first-audio latency. For latency-sensitive
traffic, the existing `ar_prefill_admission_min_batch` option in Stage 0's
`additional_config` can disable batching when set to 0.

The decoder flushes its held final audio frame before completing a request.
Its graph buckets cover up to 64 rows; larger row counts use eager decode.
Seeded requests preserve an independent generator per request, so their CPU
sampling cost differs from unseeded throughput workloads.

Streaming chunk sizes use Stage 0's `additional_config`. `latent_chunk_size`
must be positive; `initial_latent_chunk_size` must be between zero and that
size. Zero skips the initial ramp and uses the regular chunk size. The first
available audio is still released immediately.

## Voice cloning

Use the downloaded `speaker_1.wav` and its exact transcript. Omit `ref_text`
for speaker-only conditioning.

```bash
REF_DIR="$PWD/.cache/ming3_tts/references"
jq -n --arg ref_audio "file://$REF_DIR/speaker_1.wav" \
  '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "傍晚的风从山口吹过来，远处的云慢慢染成了金色。",
    "ref_audio": $ref_audio,
    "ref_text": "在此奉劝大家别乱打美白针。",
    "language": "Chinese",
    "response_format": "wav",
    "max_new_tokens": 200,
    "extra_params": {"temperature": 0.0, "ming3_cfg": 1.0},
    "seed": 1234
  }' |
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @- --output ming3_tiny_clone.wav
```

## Speech control and voice design

Pass structured controls as a JSON string in `instructions`, or use freeform
text. Speaker-only control tasks take reference audio without `ref_text`.

| Task | Control | Reference audio |
| --- | --- | --- |
| Emotion | `情感`, e.g. `高兴` | Required |
| Dialect | `方言`, e.g. `广粤话` | Required |
| Speed / pitch / volume | `语速` / `基频` / `音量` | Required |
| Freeform | Natural-language instructions | Required |
| Voice design | `风格`, e.g. `年轻女声，声音清晰自然。` | Omit |
| Built-in voice | `IP`: vendor voice name | Omit |

Dialect example:

```bash
REF_DIR="$PWD/.cache/ming3_tts/references"
jq -n --arg ref_audio "file://$REF_DIR/speaker_1.wav" \
  '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "今日天气几好，我哋食完饭一齐去海边行下啦。",
    "ref_audio": $ref_audio,
    "instructions": "{\"方言\":\"广粤话\"}",
    "response_format": "wav",
    "extra_params": {"temperature": 0.0, "ming3_cfg": 1.0},
    "seed": 1234
  }' |
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @- --output ming3_tiny_dialect.wav
```

## Podcast

Provide one reference clip and exact transcript per speaker, in speaker order.
Each dialogue line starts with `speaker_N:`. The example uses both downloaded
WAVs; replace audio and transcripts together when using your own clips.

```bash
REF_DIR="$PWD/.cache/ming3_tts/references"
jq -n \
  --arg ref_a "file://$REF_DIR/speaker_1.wav" \
  --arg ref_b "file://$REF_DIR/speaker_2.wav" \
  '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "speaker_1:你好，今天我们聊聊语音合成。\nspeaker_2:好呀，我们开始吧。",
    "ref_audio": [$ref_a, $ref_b],
    "ref_text": "speaker_1:在此奉劝大家别乱打美白针。\nspeaker_2:对此，乔宝云认为有夸大的成分。",
    "response_format": "wav",
    "extra_params": {"temperature": 0.0, "ming3_cfg": 1.0},
    "seed": 1235
  }' |
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @- --output ming3_tiny_podcast.wav
```

## Notes

- Output is mono 44.1 kHz WAV. For PCM streaming, set `response_format: "pcm"` and `stream_format: "audio"`.
- CFM CUDA Graphs follow Stage 0's configuration: `enforce_eager: true` or `compilation_config.cudagraph_mode: NONE` disables their compilation and capture.
- For chunked prefill, speaker and reference-audio conditioning is cached per
  request and sliced using the scheduled token offset, including single-token
  prefill slices. AudioVAE decodes the streams together in the same stage.
- Reference clips must be 1–30 seconds. Upload clips through `/v1/audio/voices` and reuse their names in `voice`; built-in IP names use `instructions`.
- Flow temperature defaults to 1.5; the examples explicitly use 0.0.
  `ming3_cfg=1` corresponds to native CFG scale 2. The legacy `lingv3_cfg` alias and checkpoint `lingv3_tts` metadata remain supported.
- Podcast infers language from the dialogue and does not accept `instructions`. Decode bounds are derived from the turn text; an explicit `max_new_tokens` must not be below the derived minimum.
- The Podcast example has a second-speaker similarity warning. With the same references, temperature 1.5 / seed 1234 / `language: "Chinese"` matched both intended speakers in the sampling control. Voice quality still needs listening.
