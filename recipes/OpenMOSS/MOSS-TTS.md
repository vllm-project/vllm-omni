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

For the CUDA MRV2 runner and platform fallbacks, see the
[Local 1.5 deployment profile](../../docs/configuration/stage_configs.md#moss-tts-local-15-with-model-runner-v2).
Local 1.5 defaults to MRV2 on CUDA and retains V1 on other platforms.

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

## Local 1.5 unified deployment and slot attention

`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` uses one single-GPU deploy
file, `moss_tts_local.yaml`. The CUDA settings use MRV2, 128 stream slots per
stage, full-quota private MPS, prefix caching and Triton backbone attention.
These settings replace the former C64, high-concurrency, low-latency and
optimized presets. Pipeline selection no longer probes GPU memory or MPS.
Non-CUDA platforms retain V1 through the same file's platform overrides;
the separate 2/3-NPU files describe different device placements.

Both stages share one GPU, with a 32 GiB Talker KV budget. The CUDA settings
are validated on H200 and require room for codec state and graphs, plus
`nvidia-cuda-mps-control` on `PATH`. For a smaller GPU or a service without
MPS, supply a custom deployment override for the stage capacities, graph
buckets, KV budget and `platforms.cuda.cuda_mps`.

With CUDA MRV2 GPU slot state and asynchronous chunks, Stage0 prepares the first
MTP frame immediately after prefill, decodes it locally and sends it directly.
This requires a `UniProcExecutor` (or subclass) with TP=1 and PP=1. Other executors
retain regular Stage1 delivery without loading the extra Stage0 decoder.
Requests whose stop conditions or sampling constraints prevent safe early
publication also retain regular delivery. Stage1 primes its streaming state
with the same codes and sends subsequent audio without duplicating the first
frame. Its regular dispatch target is 16 with a maximum wait of 6 ms.
`local_compile_audio_sampler: true` in the Talker HF overrides compiles the native
Torch sampler; explicit request generators use the original helper. Set the
override to `false` in a deployment file to disable that compilation. The setting is enabled in the unified CUDA profile.
The Stage0 decoder adds approximately 2 GiB of parameter weights for this
checkpoint, plus buffers and CUDA Graph memory, all included in model memory
profiling. Its T=1 graph buckets follow the Talker's capture sizes up to the
request capacity.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --omni \
  --deploy-config vllm_omni/deploy/moss_tts_local.yaml \
  --stage-init-timeout 1200 --init-timeout 1500
```

Evaluate WER, speaker similarity and streaming latency for your workload;
throughput gains do not establish quality equivalence. Do not apply partial
MPS SM quotas: BF16 GEMM outputs were observed incomplete in that configuration
on the validation environment. MPS uses an owned control socket or an
explicitly supplied operator socket; only the owned daemon is stopped at shutdown.

The Talker uses mixed FULL CUDA graphs with a 512-token prefill budget and
capture limit. This keeps new prompt chunks within graph coverage instead
of letting large prefills monopolize streaming decode. GPU request slots
retain Local hidden states, audio codes and continuation control. Published
codes own their storage, so slot reuse cannot overwrite an in-flight output.
The codec uses `triton_slot` attention, Inductor mode 3 with combo kernels
disabled, and codec-owned CUDA graphs through batch 128. Cold compilation
can take several minutes; subsequent starts can reuse the AOT cache.

The codec coalesces ready streams with
`connectors.shm.extra.generation_min_batch_size: 16` and
`generation_max_wait_ms: 6`. These are a dispatch target and bounded wait,
not an execution cap. Pending outputs retire before waiting; cancellation
and input notifications keep their scheduler bookkeeping. Set the wait to
`0` in a custom deployment to disable coalescing.

Two Talker HF overrides reduce CPU and sampler work:

- `mrv2_batch_prefill: true` reuses the batch's text embeddings and combines
  CPU reference-code slices and positions into one pinned upload. Host
  staging is reused only after the upload event completes.
- `mrv2_direct_tokens: true` returns the determined text token after the
  Local audio/stop decision, avoiding a full text-vocabulary sampling pass.
  Audio-code and binary-stop sampling retain their original algorithm;
  requests needing distribution metadata or token constraints use the normal
  sampler.

When editing a stage's `hf_overrides`, retain the other required entries:
deployment inheritance replaces this mapping as a whole. Measure first-packet
latency and throughput together when changing these settings.

The optional connector setting `generation_coalescing_policy: idle_wait`
waits for an inbox notification when no stream is runnable, all receivers
are parked and registered, and no output needs retirement. The first ready
arrival starts a fresh coalescing window, preserving the batch budget.
Control messages, including cancellations, can wait up to two windows
instead of one (12 ms at the 6 ms setting). The default policy remains
`fixed`; neither policy changes the global orchestration default.

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
Zero-length padding rows skip the attention computation and emit zeros.

The CUDA profile includes dense codec batch buckets, including 3, 5, 6, 7,
10, 12, 14 and 24. To compare with power-of-two buckets, change the stage-1
`cudagraph_capture_sizes` to `[1, 2, 4, 8, 16, 32, 64, 128]`.
Keep the maximum bucket equal to the state capacity to retain terminal-tail
coalescing. Smaller buckets reduce padding work but require more graphs;
measure complete serving runs before selecting them for a deployment.

For a codec output-path comparison, set
`connectors.shm.extra.codec_gpu_stream_output: 0` to select the synchronous
codec output path; `1` selects GPU output snapshots on a private codec
stream. Both configurations use MRV2 and keep streaming audio responses.
The option changes input ownership, metadata staging and snapshot handling
as well as transfer scheduling, so its timing difference is not just D2H
copy time.

Those chunk-complete semantics truncate the causal window: every ring holds
exactly `context` entries and a whole chunk is written before attention runs,
so token `i` of a `T`-token chunk sees `capacity - T + i + 1` keys. With the
tokenizer-v2 decoder's per-layer contexts (10/10/8/4/2/1 s, i.e. 125/250/400/
400/400/400 tokens) a 15-frame chunk is 480 tokens at the last transformer, so
its first 80 tokens attend to nothing and no chunk sees the previous one there.
Padding a terminal tail to 15 frames therefore decodes it differently from an
exact-length decode; the difference is deterministic (identical in fp32 and
fp64), not bf16 noise. The connector option `codec_ring_headroom: 1` sizes each
ring as `context + max_chunk_frames * tokens_per_frame` (including ramp shapes); chunked streaming
then reproduces whole-sequence decoding bit-for-bit in fp64 and padded tails
equal exact tails. It increases state/graph memory and capture time; historical C256 experiments
observed about a 2% throughput cost. It is off by default and its perceptual
quality benefit has not been established. Terminal tails are padded
into the regular 15-frame graph bucket in either mode.
The slot kernel is specific to the CUDA tokenizer-v2 decoder; other paths
retain their existing attention implementation. Event-driven orchestration
remains independently selectable and its default is unchanged.

### Optional progressive chunks

Set `codec_chunk_ramp` under `connectors.shm.extra` to insert smaller chunks
before steady decoding, for example:

```yaml
connectors:
  shm:
    extra:
      codec_chunk_ramp: [1, 4, 15]
```

The ramp's first entry overrides `initial_codec_chunk_frames`. After the last
entry, chunks use `codec_chunk_frames`. Each request advances independently;
the final partial chunk is flushed and an empty terminal carries no codec
tokens. The codec captures every ramp length, sizes the maximum execution
step for the largest entry, and uses the ramp's first entry for its dedicated
first-chunk graph. Additional shapes increase startup and graph memory.

Ramps are opt-in. They can reduce early playback underrun while increasing
codec calls, first-audio latency or total generation time. The measured C128
throughput presets retain 1→15; their published figures do not establish an
additional benefit from enabling ramps. The processor and graph-shape tests
run without GPU or model weights:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q \
  tests/model_executor/stage_input_processors/test_moss_tts_async_chunk.py \
  -m 'core_model and cpu' --run-level core_model
```

### Reference-encoding options

Reference encoding runs in the API layer, independently of MRV2. Enable
reference graphs explicitly with `VLLM_OMNI_MOSS_REF_GRAPHS=1`; the default
eager encoder is retained when the option is absent. The graph path uses the
loaded tokenizer's encoder/quantizer, length buckets and optional compilation
(`VLLM_OMNI_MOSS_REF_COMPILE=0` disables compilation). It also uses windowed
attention; `VLLM_OMNI_MOSS_REF_ATTN=sdpa` retains the original attention.
Compilation and attention changes need not produce bit-identical codes.

For multiple API processes, `VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR` enables shared
reference-code storage and, by default, a single encoder host with four
workers. Use a dedicated directory per service, checkpoint and encoding
configuration. Workers have their own graph resources; all graph captures
complete before serving begins. `VLLM_OMNI_MOSS_REF_SHARED_ENCODER=0` retains
separate encoders while sharing codes. More workers/graphs consume memory;
they do not imply more GPUs. Example after selecting an available GPU and
configuring any operator-managed MPS daemon:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_EVENT_DRIVEN_ORCH=1 \
VLLM_OMNI_CONNECTOR_RECV_POLL_MS=1 \
VLLM_OMNI_MOSS_REF_GRAPHS=1 \
VLLM_OMNI_MOSS_REF_ENCODER_WORKERS=4 \
VLLM_OMNI_MOSS_REF_BATCH_WINDOW_MS=0 \
VLLM_OMNI_MOSS_REF_INFLIGHT=4 \
VLLM_OMNI_MOSS_REF_HOST_WINDOW_MS=2 \
VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR=/dev/shm/moss-local-service \
vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --omni \
  --api-server-count 4 \
  --deploy-config vllm_omni/deploy/moss_tts_local.yaml \
  --stage-init-timeout 1200 --init-timeout 1500
```

These reference-encoder options are independent of the unified deployment.
Evaluate cold and hot reference requests separately when enabling them.

### Cold versus hot reference measurements

A cold reference misses the encoded-reference cache: the request pays for
reference parsing/preparation and GPU encoding before speech generation.
A hot reference reuses reference codes but still synthesizes new speech;
generated audio is not cached. Pre-registering a reference moves encoding
into registration and can make subsequent synthesis hot, but registration
latency is still part of the first-use cost.

Model startup/compilation is separate from reference coldness. EN1088 contains
repeated references, so its first pass is not uniformly cold. For an all-unique
cold pass, use its 666 first-occurrence references, keep order, warm execution
with disjoint references, then repeat the same 666 requests. A second pass
through independent API-local caches is not guaranteed hot; verify cache
coverage before comparing it with a shared-cache result. Report requests/s
alongside audio-s/s and retain failed, empty, long and near-silent outputs.

### Reproduce the serving benchmark

Use the complete Seed-TTS English test set, including reference audio and text.
Set `SEED_TTS_DATA` to the directory containing `en/meta.lst` and its 1088
entries. Run the native benchmark after the server is ready:

```bash
export VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000
export VLLM_OMNI_BENCH_AUDIO_CHANNELS=2
export SEED_TTS_WER_EVAL=0

for concurrency in 128; do
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
