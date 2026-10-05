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
capacity to 128 and fixes the Talker KV budget at 32 GiB. The Talker uses
mixed FULL CUDA graphs with both its token budget and capture limit set to
512, so new prompt chunks remain within graph coverage. It also enables
`hf_overrides.mrv2_gpu_slot_state`: Local hidden states, audio codes and
continuation control stay in GPU request slots, gathered through MRV2's
batch mapping. Prefill conditioning and explicit per-request sampling seeds
retain their original behavior. Published audio codes own their storage, so
slot reuse cannot overwrite an in-flight output. Set the override to `false`
to compare with the generic Omni model state. The profile selects
`codec_attention_backend: triton_slot`, Inductor mode 3 with combo kernels
disabled, and codec CUDA graph buckets through 128. The configuration was
validated on one H200. The earlier capacity-256 variant used about 110 GiB of
sampled peak GPU memory, including loading and warmup. Reduce the stage capacities,
graph buckets and coalescing target together when adapting it to smaller GPUs. Cold codec
compilation can take several minutes; subsequent starts can reuse the AOT
cache. Graph capture is still performed at startup.

The high-concurrency profile also coalesces ready codec streams with
`connectors.shm.extra.generation_min_batch_size: 32` and
`generation_max_wait_ms: 12`. The target controls when to stop waiting; it
neither caps execution at 32 nor guarantees every codec group has 32 rows.
First and final chunks share the bounded window at high concurrency. When
fewer than 32 requests remain, dispatch is immediate. Pending outputs are
retired before waiting, with the original deadline preserved, so waiting does
not prevent the previous batch from releasing its in-flight state. Cancellation
and input notifications keep their ordinary scheduler bookkeeping. This option
requires a stateful native MRV2 generation stage with TP1/PP1. Set the wait to
`0` to disable coalescing; other profiles keep the immediate-dispatch default.

The high-concurrency profile enables two additional Local model-state
optimizations under the CUDA stage-0 `hf_overrides`, alongside
`mrv2_gpu_slot_state: true`:

- `mrv2_batch_prefill: true` reuses the batch's text embeddings and combines
  CPU reference-code slices and their destination positions into one pinned
  upload. Host staging is reused only after the upload event completes;
  GPU-resident references keep the original path.
- `mrv2_direct_tokens: true` returns the already determined text token after
  the Local audio/stop decision. It avoids constructing and sampling the
  full text vocabulary. Audio-code and binary-stop sampling are unchanged.
  Requests requiring distribution metadata or token constraints, and
  unsupported execution modes, retain the normal sampler.

Other profiles leave both flags disabled. Set both flags explicitly when
comparing their combined effect. Deployment
inheritance replaces a stage's `hf_overrides` mapping, so an overlay must also
retain `mrv2_gpu_slot_state: true`. Measure first-packet latency as well as
throughput: faster prompt preparation can change competition between the
Talker and codec sharing the GPU.

For the single-H200 capacity-128 throughput baseline, keep both switches on.
In complete Seed-TTS EN1088 runs at client concurrency 128, four combined-switch
rounds pooled 344.11 audio-s/s versus 335.78 for batch prefill alone; direct
tokens alone had no reliable gain. The combined path had 685 ms mean first
audio versus 679 ms across six control rounds, with lower mean completion time.
A fixed-seed 128-row audio sample showed mean Whisper WER of 3.68% versus
3.78% for the control. This sample does not establish speaker similarity or
exclude small quality changes. At concurrency 256, the combined path had
raised mean first-audio latency by about 0.33 s, so other profiles leave the
switches off until their latency and quality are checked.

The optional connector setting `generation_coalescing_policy: idle_wait`
waits for an inbox notification when no stream is runnable, all receivers
are parked and registered, and no output needs retirement. The first ready
arrival starts a fresh coalescing window, preserving the batch budget.
Control messages, including cancellations, can wait up to two windows
instead of one (24 ms at the 12 ms setting). The default policy remains
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

The high-concurrency profile includes intermediate codec batch buckets 6, 12
and 24. To compare with power-of-two buckets, change the stage-1
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

### Low-latency and reference-encoding options

`moss_tts_local_mrv2_low_latency.yaml` keeps both stage capacities at 128 and
uses prefill 2048, GPU slot state, batch prefill, direct tokens and eager MTP.
Its codec first-chunk fast path uses dedicated graphs and a stream handoff
before the regular decoder reuses the request slot. A gate limits contention
with regular codec work. Slot waits are bounded at 30 seconds. A failed or
stalled decode raises an error rather than replaying a partially advanced
slot; a failed output enqueue retains owned PCM for regular delivery.
Closing the decoder rejects new jobs. Its regular-batch limit of 16 is internal dispatch
policy; it does not change the client concurrency or stage capacities.

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
  --deploy-config vllm_omni/deploy/moss_tts_local_mrv2_low_latency.yaml \
  --stage-init-timeout 1200 --init-timeout 1500
```

This example is a configurable serving profile, not the complete launch
configuration of a historical benchmark. In particular, the low-latency YAML
uses synchronous stage-1 scheduling; the retained source46 experiment used
asynchronous stage-1 scheduling, FP8 backbone, MPS and additional encoder
environment settings. Do not infer a throughput result from the YAML alone.

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
