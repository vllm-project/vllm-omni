# MOSS-TTS

## Summary

- Vendor: OpenMOSS
- Models: `OpenMOSS-Team/MOSS-TTS` (8B), `OpenMOSS-Team/MOSS-TTS-v1.5` (8B),
  `OpenMOSS-Team/MOSS-TTS-Realtime` (1.7B), `OpenMOSS-Team/MOSS-TTSD-v1.0` (8B),
  `OpenMOSS-Team/MOSS-SoundEffect` (8B), `OpenMOSS-Team/MOSS-VoiceGenerator` (1.7B)
- Task: Text-to-speech synthesis, sound effect generation, zero-shot voice design
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API; offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe for 24 kHz multilingual TTS with voice cloning (20 languages including
Chinese and English). Choose a variant based on your latency and quality requirements:

| Model | Params | Use case |
| --- | --- | --- |
| MOSS-TTS | 8B | General TTS, highest quality |
| MOSS-TTS-v1.5 | 8B | General TTS upgrade of 1.0: 31 languages, steadier cloning, `[pause Xs]` markers (set `language` for best results); same `MossTTSDelay` API |
| MOSS-TTS-Realtime | 1.7B | Lowest latency (TTFB ~180 ms), streaming-first |
| MOSS-TTSD-v1.0 | 8B | Multi-turn dialogue TTS |
| MOSS-SoundEffect | 8B | Sound effect synthesis from text description |
| MOSS-VoiceGenerator | 1.7B | Zero-shot voice design |

The variants above share the same codec (`OpenMOSS-Team/MOSS-Audio-Tokenizer`, ~7 GB) and
output 24 kHz mono audio.

MOSS-TTS-Local-Transformer-v1.5 uses MOSS-Audio-Tokenizer-v2 and outputs 48 kHz
stereo audio. For Local voice cloning through `/v1/audio/speech`, provide an
accurate `ref_text` transcript alongside `ref_audio`. The adapter uses the
reference transcript and audio as a continuation prefix, then generates only
the requested `input` speech. Without a nonblank `ref_text`, Local uses
audio-reference generation. Reference transcripts must match the reference
audio; they are not style instructions.

## References

- Offline inference example: [`examples/offline_inference/text_to_speech/moss_tts/`](../../examples/offline_inference/text_to_speech/moss_tts/)
- Deploy configs: [`vllm_omni/deploy/moss_tts.yaml`](../../vllm_omni/deploy/moss_tts.yaml) and variants
- HuggingFace org: <https://huggingface.co/OpenMOSS-Team>

## Hardware Support

### GPU

#### 1x H100 80GB — MOSS-TTS (8B)

##### Environment

- OS: Linux
- Python: 3.11+
- CUDA 12.8
- vLLM-Omni version: see `vllm_omni/__version__.py`

##### Command

```bash
# The codec is loaded automatically from OpenMOSS-Team/MOSS-Audio-Tokenizer.
# Override the path with MOSS_TTS_CODEC_PATH if you have a local copy.
vllm serve OpenMOSS-Team/MOSS-TTS --omni --port 8091
```

##### Verification

Voice cloning (provide a reference audio clip):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-TTS",
        "input": "Hello, this is a voice cloning test.",
        "voice": "default",
        "ref_audio": "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/zh_1.wav",
        "response_format": "wav"
    }' --output output.wav
```

##### Notes

- Peak GPU memory: ~18 GB for the talker (8B) + ~8 GB for the codec decoder on the same device.
  Use `gpu_memory_utilization: 0.85` in `moss_tts.yaml` (default).
- Output: 24 kHz mono WAV.
- The `MOSS_TTS_CODEC_PATH` environment variable overrides the codec checkpoint location.

---

#### 1x A10G 24GB — MOSS-TTS-Realtime (1.7B)

##### Environment

- OS: Linux
- Python: 3.11+
- CUDA 12.8

##### Command

```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Realtime --omni --port 8091
```

##### Verification

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
        "input": "This is a low-latency streaming TTS test.",
        "voice": "default",
        "ref_audio": "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/zh_1.wav",
        "response_format": "wav",
        "stream": true,
        "stream_format": "audio"
    }' --output output.wav
```

##### Notes

- Peak GPU memory: ~6 GB for the talker (1.7B) + ~8 GB for the codec decoder.
- First-audio latency (TTFB): ~180 ms on A10G.
- `codec_chunk_frames: 15` in `moss_tts_realtime.yaml` for lower TTFA than the 8B variant.

---

#### 1x A10G 24GB — MOSS-SoundEffect (8B, sound effect synthesis)

##### Command

```bash
vllm serve OpenMOSS-Team/MOSS-SoundEffect --omni --port 8091
```

##### Verification

Sound effect synthesis takes a text description instead of reference audio:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenMOSS-Team/MOSS-SoundEffect",
        "input": "Thunder rumbling, rain pattering on a tin roof.",
        "response_format": "wav"
    }' --output thunder.wav
```

##### Notes

- No `ref_audio` required or accepted for MOSS-SoundEffect.
- Input field maps to the `ambient_sound` parameter in the upstream processor.
- Rate: ~12.5 tokens per second; longer descriptions produce longer audio.

## MOSS Local v1.5 full BF16 pipeline on H200

The opt-in [`moss_tts_local_h200_full.yaml`](../../vllm_omni/deploy/moss_tts_local_h200_full.yaml)
combines native MRV2, FULL backbone graphs, CUDA MPS, Local QKV lookup and
fused sampling, slot codec attention, asynchronous PCM output and early first-audio
delivery. It requires one large-memory NVIDIA H200, TP/PP=1, CUDA/Triton and the
MOSS Audio Tokenizer v2 checkpoint. It uses BF16 without quantization and disables
prefix caching. Keep the standard Local profile for other deployment targets.

The Talker owns a private first-frame codec and publishes its PCM through the
existing first-audio sender. The regular codec receives the same first code to
establish streaming state, primes at 15 frames, and removes the already-delivered
PCM frame. Clients receive chunks of 1, 14, 15, ... frames; shorter terminal chunks
are flushed. This does not share persistent codec state across processes.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --omni --host 0.0.0.0 --port 8124 --trust-remote-code \
  --deploy-config vllm_omni/deploy/moss_tts_local_h200_full.yaml \
  --api-server-count 1 --init-timeout 3600 --stage-init-timeout 3600 \
  --allowed-local-media-path /path/to/workspace
```

The YAML explicitly enables `cuda_mps: true`; stage-specific priorities alone do
not start MPS. Cold compilation and graph capture can take several minutes.
The early decoder adds a second decoder copy and private graph buffers in the
Talker process, so this preset is not a memory sizing recommendation for smaller
GPUs. Backbone tile64 attention is installed only on eligible MOSS model
instances, without replacing an upstream global operator. Other attention
configurations retain the upstream implementation.

Reproduce the original client workload with the existing benchmark:

```bash
VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  python benchmarks/tts/bench_tts.py --host 127.0.0.1 --port 8124 \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --task voice_clone --locale en --dataset-path /path/to/workspace/seedtts_testset \
  --num-prompts 1088 --concurrency 64 64 128 64 128 --output-len 256 \
  --output-dir /path/to/results/moss-local-full
```

Historical retained-source results on one H200 (September 27, 2026), not a new
measurement of the submitted integration:

| Concurrency | Audio seconds / wall second | Client mean TTFP (ms) |
| --- | ---: | ---: |
| 64 | 422.5 | 87.7 |
| 128 | 521.1 | 201.7 |

Each of five rounds completed 1088 requests without errors. The first C64 round
is treated as cold and excluded. Throughput pools audio seconds and wall time
across the two remaining rounds per concurrency; TTFP is a request-weighted mean,
not P50 or an unloaded single-request latency. A later same-configuration H200
rerun measured 418.2 / 86.2 ms at C64 and 517.6 / 198.2 ms at C128.

These measurements cover the whole combination and cannot be assigned to one
kernel or added to other PR speedups. The Local sampler changes the RNG mapping
relative to `torch.multinomial`; identical speech at a fixed seed is not claimed.
The empty-history first decoder is experimental: inherited cross-service waveform
differences remain, and protocol checks do not establish speech-quality equivalence.
No new WER, SIM or UTMOS evaluation is included in this integration.
