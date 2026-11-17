# AURA_v2 full-duplex audio-visual assistant — H200

> Proactive speech and video chat on `/v1/realtime?duplex=1`: AURA_v2 watches
> the camera, decides on every turn whether to speak, and answers in 24 kHz
> speech.

## Summary

- Vendor: AURA team
- Model: `aurateam/AURA_2` (Qwen3.5 hybrid thinker), with
  [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and
  [`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- Task: streaming audio-visual assistant. Speech and video in; text and
  speech out; silent turns, proactive replies and tool calls.
- Mode: online full-duplex Realtime serving only
- Hardware: 1x or 2x NVIDIA H200 (141 GB)
- Recommended deployment: two GPUs, `vllm_omni/deploy/aura_omni_v2_2gpu.yaml`
- Maintainer: [`@NumberWan`](https://github.com/NumberWan)

## When to use this recipe

Use this recipe to serve AURA_v2 behind a live camera and microphone. The
client pushes speech and video frames over the Realtime API and commits each
utterance (push-to-talk), or commits frames alone so the model can speak up
when something it was asked to watch for appears. Every commit is one
four-stage request:

```text
Qwen3-ASR -> AURA_v2 thinker -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

The thinker starts each reply with `<|silent|>` (no speech for this turn) or
`<|respond|>` followed by the reply. Design details:
[Full-Duplex Runtime (AURA_v2)](../../docs/design/aura_v2_duplex.md).

## Supported model contract

| Entry point | Status |
| --- | --- |
| `/v1/realtime?duplex=1` | supported |
| `/v1/chat/completions`, offline `Omni.generate` | not supported for AURA_v2 |

| Input | Contract |
| --- | --- |
| Video | JPEG frames on `input_audio_buffer.append` (`video_frames`); required on every commit |
| Audio | PCM16 mono, 16 kHz, `is_speech: true` for user speech; optional (frames-only commits are vision turns) |
| Text | Not a turn input. `instructions` sets the system prompt |
| Tools | Realtime `session.tools`; results come back as `function_call_output` |

| Output | Contract |
| --- | --- |
| Text | `response.output_audio_transcript.delta`; grounding boxes stay in the text, object labels are unwrapped |
| Speech | `response.output_audio.delta`, PCM16 mono 24 kHz, Qwen3-TTS CustomVoice (default speaker `Vivian`) |
| Silence | `response.listen`, no audio |
| Tool call | `response.function_call_arguments.done` |

| Profile | Deploy file | GPUs | Stage placement | Graphs | Sessions |
| --- | --- | --- | --- | --- | --- |
| 2-GPU (recommended) | `vllm_omni/deploy/aura_omni_v2_2gpu.yaml` | 2 | Stage1 on GPU 1; Stage0/2/3 on GPU 0 | CUDA graphs on all stages | up to 16 |
| 1-GPU | `vllm_omni/deploy/aura_omni_v2.yaml` | 1 | all stages on GPU 0 | eager | 1 |

## References

- Design: [Full-Duplex Runtime (AURA_v2)](../../docs/design/aura_v2_duplex.md)
- Online example and web demo: [`examples/online_serving/aura_omni/`](../../examples/online_serving/aura_omni/README.md)
- Shared browser client: [`examples/online_serving/realtime_web/`](../../examples/online_serving/realtime_web/)
- Realtime duplex API: [`docs/serving/realtime_duplex_api.md`](../../docs/serving/realtime_duplex_api.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Benchmark dataset: [OmniInteract](https://github.com/Lucky-Lance/OmniInteract) `1q1a`

## Hardware

- Accelerator: NVIDIA H200, 141 GB per device (`nvidia-smi` reports 143771 MiB)
- Number of devices: 1 (1-GPU profile) or 2 (2-GPU profile)
- Interconnect: PCIe between the two devices; stages exchange data through
  host shared memory
- Qualification scope: both profiles runtime-qualified (see Qualification
  evidence)

## Software environment

- OS: Linux 5.10
- Python: 3.12.13
- Driver / runtime: NVIDIA 570.133.20, CUDA 13.0
- PyTorch: 2.14.0+cu130
- vLLM: 0.33.0
- vLLM-Omni: qualified at `c77f5222` (on `main` `813618b5`)

## Command

Run from the repository root. Each stage loads its own checkpoint from the
deploy file; the positional model is the served model name.

### 2-GPU (recommended)

```bash
vllm serve aurateam/AURA_2 --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/aura_omni_v2_2gpu.yaml \
  --host 0.0.0.0 --port 8091
```

### 1-GPU

```bash
vllm serve aurateam/AURA_2 --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/aura_omni_v2.yaml \
  --host 0.0.0.0 --port 8091
```

For local checkpoints, copy the deploy file and edit each stage's `model`.

The server is ready when the log prints `Duplex warmup finished`. Startup runs
one spoken warmup turn through all four stages.

### Web demo (push-to-talk, tools)

```bash
python -m examples.online_serving.aura_omni.realtime_web \
  --ws-backend ws://127.0.0.1:8091 --model aurateam/AURA_2 --port 7863
```

Open `http://<host>:7863/`. Hold the button to talk; release to commit. The
demo proxy runs the vision tool set (`VisionToolExecutor`) in
`examples/online_serving/realtime_web/tools.py`; its search tools need
`DEEPSEEK_API_KEY`. Set `AURA_DEMO_TOOLS=safe` for the allowlisted set, or
`AURA_DEMO_TOOLS=0` to serve without function calling.

## Verification

```bash
python examples/online_serving/aura_omni/smoke_duplex_realtime_client.py \
  --url ws://127.0.0.1:8091/v1/realtime --model aurateam/AURA_2
```

Expected: `session.created`, then for the spoken question a
`response.created`, `response.output_audio_transcript.delta` events,
`response.output_audio.delta` events (24 kHz PCM16) and `response.done`.

## Qualification evidence

Measured with Stage1 prefix caching off, before the shipped profiles turned it on. OmniInteract `1q1a` videos `0002`–`0004` (12 questions, injected
at `question_time`), after a full `0001` session as warmup that is not scored.
Fresh server per row, three rounds unless noted. TTFT and TTFP are client-side
from the commit to the first text and the first audio of spoken turns, p50 /
p90.

| Profile | Sessions | TTFT (ms) | TTFP (ms) | Stage1 TPOT (ms) | Peak memory |
| --- | ---: | --- | --- | --- | --- |
| 2-GPU | 1 | 286 / 397 | 515 / 619 | 8.4 / 8.8 | 39.3 GiB (GPU 0) / 131.8 GiB (GPU 1) |
| 2-GPU | 8 | 678 / 2293 | 1671 / 6582 | 69.5 / 298 | 39.3 / 132.2 GiB |
| 2-GPU | 16 | 1145 / 5210 | 3745 / 9065 | 151 / 296 | 39.3 / 132.7 GiB |
| 1-GPU (1 round) | 1 | 536 / 706 | 1541 / 1682 | 35.0 / 35.3 | 109.9 GiB |

Peak memory is the `nvidia-smi` reservation set by `gpu_memory_utilization`.

- Accuracy: a keyword check of each answer (not the IA-QTF1 judge) scores 10/12
  in each of three 2-GPU rounds, the same as the reference AURA_v2 streaming
  inference (10/12 in each of two rounds); 1-GPU scored 7/12 in one round.
- Concurrency: at 8 and 16 sessions no reply names an object from another
  session's video (384 spoken turns).
- Long session: a 303 s video (294 turns) ran without a session error.
- Tools: 27 push-to-talk lines through the web demo proxy, 11 with a tool call;
  all 27 spoken.
- Startup: warmup takes 45–55 s; the first user turn after it reaches first
  audio in 528 ms.

## Notes

- Memory usage: see Qualification evidence. The 1-GPU profile reserves
  `0.08 + 0.55 + 0.12 + 0.12` of GPU 0.
- Sizing: with Stage1 prefix caching off, `max_num_seqs: 8` serves 16
  sessions by queueing turns (p90 TTFP 9 s). Lower `max_sessions` for a
  tighter latency target.
- Key flags: Stage1 `async_scheduling: false` (the two-token logits processor
  reads the live output list), `enable_prefix_caching: true` and
  `mamba_cache_mode: align`. `async_chunk: true` is required for Talker to
  Code2Wav audio.
- Known limitations: no server VAD (the client commits each utterance); no
  typed text turns; `response.cancel` is not used by the reference client.

## Supported features

| Feature | AURA_v2 status |
| --- | --- |
| [Realtime full duplex](../../docs/serving/realtime_duplex_api.md) | supported (`turn_commit` adapter) |
| Concurrent sessions | 2-GPU profile, up to 16 |
| Next turn while the prior reply drains | supported (`supports_concurrent_turn_requests`) |
| CUDA graphs | supported on all stages (2-GPU profile) |
| [Async chunk](../../docs/design/feature/async_chunk.md) | required (Talker to Code2Wav) |
| [Prefix caching](../../docs/design/feature/prefix_caching.md) | Stage0 on; Stage1 on via native hybrid KV (`mamba_cache_mode: align`) |
| Tool calls | supported (client-executed, spoken follow-up) |
| Chat completions / offline | not supported |
| AMD, Ascend, Intel | not qualified |
