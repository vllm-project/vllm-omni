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

For the optional CUDA MRV2 runner, see the
[Local 1.5 deployment profile](../../docs/configuration/stage_configs.md#moss-tts-local-15-with-model-runner-v2).
The default Local deployment continues to use V1.

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

## Local 1.5 MRV2 and slot attention

`MOSS-TTS-Local-Transformer-v1.5` supports the native CUDA MRV2 pipeline with
an explicit deploy profile. The default Local profile continues to use V1.
Both profiles below preserve 1-frame initial and 15-frame steady codec chunks
and the model's sampling defaults.

```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --omni \
  --deploy-config vllm_omni/deploy/moss_tts_local_mrv2.yaml
```

This C64 profile bounds Talker prefill to 512 tokens and retains codec-owned
CUDA graphs without compiling the codec with Inductor.

For sustained high concurrency on a large-memory CUDA GPU, use:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --omni \
  --deploy-config vllm_omni/deploy/moss_tts_local_mrv2_high_concurrency.yaml \
  --stage-init-timeout 1200 --init-timeout 1500
```

The high-concurrency profile places both stages on one GPU, sets each stage's
capacity to 256 and fixes the Talker KV budget at 32 GiB. It selects
`codec_attention_backend: triton_slot`, Inductor mode 3 with combo kernels
disabled, and codec CUDA graph buckets through 256. The configuration was
validated on one H200; the compiled profile used about 118.3 GiB of sampled
peak GPU memory, including loading and warmup. Reduce the stage capacities
and graph buckets together when adapting it to smaller GPUs. Cold codec
compilation can take several minutes; subsequent starts can reuse the AOT
cache. Graph capture is still performed at startup.

The codec backends differ in state access:

- `sdpa` uses PyTorch attention after gathering and updating the ring cache.
- `triton` replaces the attention calculation, retaining the ring-cache
  gather/copy and explicit mask construction.
- `triton_slot` writes surviving K/V directly into request slots, attends
  directly to the ring, and advances the active slot offsets in three
  ordered kernels. Graph-padding rows do not advance persistent state.

The slot path preserves the existing chunk-complete ring semantics, including
retaining the final cache-capacity tokens when a chunk exceeds the ring.
It does not introduce another cache owner or change request-slot lifetime.
The slot kernel is specific to the CUDA tokenizer-v2 decoder; other paths
retain their existing attention implementation. Event-driven orchestration
remains independently selectable and its default is unchanged.

### Reproduce the serving benchmark

Use the complete Seed-TTS English test set, including reference audio and text.
Set `SEED_TTS_DATA` to the directory containing `en/meta.lst` and its 1088
entries. Run the native benchmark after the server is ready:

```bash
export VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000
export VLLM_OMNI_BENCH_AUDIO_CHANNELS=2
export SEED_TTS_WER_EVAL=0

for concurrency in 128 256; do
  for phase in warm r1 r2; do
    vllm bench serve --omni \
      --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
      --backend openai-audio-speech --endpoint /v1/audio/speech \
      --dataset-name seed-tts --dataset-path "$SEED_TTS_DATA" \
      --seed-tts-locale en --disable-shuffle --num-prompts 1088 \
      --num-warmups 0 --ready-check-timeout-sec 0 \
      --output-len 256 --max-concurrency "$concurrency" \
      --request-rate inf --seed 42 \
      --extra-body '{"task_type":"Base","max_new_tokens":256}' \
      --save-result --save-detailed --result-dir results/moss-local \
      --result-filename "c${concurrency}-${phase}.json"
  done
done
```

Discard the complete `warm` pass and combine measured runs as total generated
audio seconds divided by total benchmark duration. Require 1088 successes,
zero failures and 1088 nonempty-audio metric samples in each run. The output
cap matches the benchmark protocol; success alone does not establish speech
quality or that every sentence ended before the cap. Dataset seed 42 does
not fix an independent sampling seed for every request.

For an attention-only comparison, copy the high-concurrency YAML beside the
original to preserve relative `base_config` resolution. Change its
codec `compilation_config.mode` to `0` while retaining `cudagraph_mode: FULL`,
and compare `codec_attention_backend: triton` against `triton_slot`. Keep all
other settings, warmup and client concurrency identical. Restart the server
between configurations and use separate result directories. Comparing the
compiled slot profile to uncompiled Triton includes both changes.

Kernel and MHA regression tests require CUDA, compatible vLLM/Triton packages,
and no model weights:

```bash
python -m pytest -q tests/model_executor/models/moss_tts/test_slot_attention.py \
  tests/model_executor/models/moss_tts/test_streaming_attention.py \
  -m 'core_model and cuda' --run-level=core_model
```
