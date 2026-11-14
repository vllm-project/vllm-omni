# Haru-4B — H200

> Live full-duplex spoken dialogue (the model listens while it speaks), with a
> character card and a voice reference per session, on the native vLLM-Omni engine.

## Summary

- Vendor: higurashi-ai
- Model: [`higurashi-ai/Haru-4B`](https://huggingface.co/higurashi-ai/Haru-4B)
  (Apache-2.0, a pretrained base model released for research)
- Task: Full-duplex speech-to-speech. Every 133 ms frame (7.5 Hz) the model takes
  3200 samples of the user's 24 kHz audio and returns 3200 samples of its own audio
  plus its text stream.
- Mode: Live duplex WebSocket serving through the unified duplex stack (`/v1/realtime?duplex=1`)
- Hardware: 1x NVIDIA H200 141GB
- Maintainer: [`@linyueqian`](https://github.com/linyueqian)

## When to use this recipe

Use this recipe to run Haru-4B as a phone-call style voice agent: many concurrent
conversations on one GPU, each one a resumable engine request whose prompt grows by
one position per frame. The Qwen3.5 backbone runs on vLLM's engine (paged attention
cache, Gated DeltaNet state, CUDA graphs); the MeanFlow audio head and the VibeVoice
codec run in model hooks of the same engine step, so a frame's audio leaves in the
step that produced it.

## Supported model contract

| Item | Contract |
| --- | --- |
| Endpoint | `ws://<host>:<port>/v1/realtime?duplex=1` (OpenAI Realtime events) |
| Session start | `session.update` with `instructions` (the character card text) and optional `ref_audio` |
| Voice reference | `ref_audio`: a `data:audio/...;base64,` URI (WAV, FLAC, OGG or MP3), one speaker, 3 s to 3 min of clean speech; silence is trimmed and the last 180 s are kept |
| Audio in | `input_audio_buffer.append`, `pcm_f32le` at 24 kHz, exactly 3200 samples per append |
| Audio out | one `response.output_audio.delta` per append: PCM16, 24 kHz, 3200 samples |
| Text out | `response.output_audio_transcript.delta` on frames that carry text |
| Frame alignment | the first append starts the session (its audio is ignored; send silence) and returns model frame 0; append `k` returns model frame `k + 1` |
| Session options | `temperature` (default 0.7); `extra_body`: `guidance_scale` 1 to 2 (default 1.5), `top_k` (25), `top_p` (1.0), `eop_bias` (0), `speak_first` (false), `dry` (true), `seed` (random) |
| Fixed after creation | `instructions`, `ref_audio` and the `extra_body` options |
| Languages | English, Spanish, Portuguese, German, French, Chinese, Japanese, Russian, Korean |

## References

- Upstream model card: [`higurashi-ai/Haru-4B`](https://huggingface.co/higurashi-ai/Haru-4B)
- Authors' code, web demo and character-card format: [higurashi-ai/Haru](https://github.com/higurashi-ai/Haru)
- Integration package: [`vllm_omni/model_executor/models/haru/`](../../vllm_omni/model_executor/models/haru/)
- Deploy config: [`vllm_omni/deploy/haru.yaml`](../../vllm_omni/deploy/haru.yaml)
- Related recipe (the other lockstep duplex model): [`NVIDIA/PersonaPlex.md`](../NVIDIA/PersonaPlex.md)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB
- Number of devices: 1
- Device interconnect: not used (single GPU)
- Qualification scope: online duplex serving with the deploy config below

## Software environment

- OS: Linux x86_64
- Python: 3.12
- Driver / runtime: CUDA 13.0 (PyTorch 2.14.0+cu130), Triton 3.8.0
- vLLM version: 0.33.0
- vLLM-Omni version or commit: `272c28c5c` (main) plus the Haru integration

## Command

```bash
vllm serve higurashi-ai/Haru-4B --omni \
  --deploy-config vllm_omni/deploy/haru.yaml \
  --host 0.0.0.0 --port 8000
```

The authors' web demo can use this server as its backend:

```bash
git clone https://github.com/higurashi-ai/Haru && cd Haru && pip install -e ./haru
python -m haru.server --backend vllm --vllm-url http://localhost:8000
```

## Verification

`tests/e2e/online_serving/test_haru_duplex.py` drives two lockstep sessions at once
and checks that every append returns exactly one 3200-sample PCM16 frame and that the
two identical sessions return identical audio and text:

```bash
pytest tests/e2e/online_serving/test_haru_duplex.py --run-level advanced_model -m advanced_model -s
```

## Performance

Measured on 1x H200 with `max_sessions` = `max_num_seqs` = N, `max_model_len: 8192` and a 16 GiB
KV cache. Each session runs 300 frames (40 s) in lockstep: a client appends its next frame as soon
as the previous model frame arrives, so the turnaround is one engine step with N rows. A session
stays realtime while that step is shorter than the 133 ms frame period.

| Sessions (N) | Frame turnaround p50 / p95 | First frame (prompt, about 290 positions) |
| --- | --- | --- |
| 1 | 11.7 / 14.2 ms | 128 ms |
| 4 | 21.6 / 24.0 ms | 278 ms |
| 8 | 28.5 / 30.9 ms | 314 ms |
| 16 | 44.6 / 47.6 ms | 661 ms |

## Notes

- Memory usage: 10.1 GiB of weights (bf16) plus the KV cache set by `kv_cache_memory_bytes`
  (8 GiB in the default config: 2 sessions x 32768 positions). The first start compiles and captures
  CUDA graphs; the server reported healthy after about 200 s.
- Key flags: raise `duplex_session.max_sessions` and the stage's `max_num_seqs` together
  to serve more conversations, and grow `kv_cache_memory_bytes` with them;
  `max_model_len: 32768` holds 60 minutes of conversation plus the prompt.
- Engine options the per-step audio hooks do not support, refused at startup:
  `async_scheduling`, prefix caching, speculative decoding, tensor or pipeline parallelism.
- Known limitations: Haru-4B is a pretrained base model; it holds a natural conversation
  but does not follow instructions like an assistant. Conversation state lives in the
  worker, so a preempted conversation fails instead of being recomputed.

## Supported features

| Feature | Status |
| --- | --- |
| Full-duplex realtime serving (`/v1/realtime?duplex=1`) | ✅ |
| Multiple concurrent sessions on one GPU | ✅ (`duplex_session.max_sessions`) |
| CUDA graphs for the backbone | ✅ |
| Triton codec kernels replayed from CUDA graphs | ✅ (falls back to PyTorch without Triton) |
| Offline `Omni.generate` | ❌ (served as a duplex session only) |
| Tensor / pipeline parallelism | ❌ |
| Prefix caching, async scheduling | ❌ |
