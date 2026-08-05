# PersonaPlex

> Live full-duplex speech-to-speech (talk to the model in your browser, it listens
> while it speaks) with persona and voice control, on the native vLLM-Omni engine.

## Summary

- Vendor: NVIDIA
- Model: [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
  (gated; a Moshi finetune)
- Task: Full-duplex speech-to-speech. 24 kHz mic audio in, agent speech +
  inner-monologue text out, in 80 ms lockstep. Persona (role text) and voice
  (zero-shot clone) are set per session.
- Mode: Live duplex WebSocket server with the official browser client, single or
  multi-session (`--batch-size`), plus an offline WAV-in / WAV-out example
- Maintainer: [`@linyueqian`](https://github.com/linyueqian)

## When to use this recipe

Use this recipe to run `nvidia/personaplex-7b-v1` as a live voice agent on one GPU.
The integration is **moshi-free**: the temporal transformer, depformer, and
streaming Mimi codec are all native vLLM-Omni modules (Mimi runs on
`transformers.MimiModel`), so no vendored fork is installed. Decoding is greedy,
matching the reference implementation frame for frame on the golden replays used
as the acceptance gate.

PersonaPlex is the first Moshi-class (pure-lockstep) model in the experimental
full-duplex framework (`vllm_omni/experimental/fullduplex/`), and the first with
elastic multi-session batching: one engine hosts `--batch-size` concurrent
conversations, callers join or leave on any 80 ms tick without disturbing live
slots (greedy isolation is bit-exact).

## References

- Serving package:
  [`vllm_omni/experimental/fullduplex/personaplex/`](../../vllm_omni/experimental/fullduplex/personaplex/)
- Native model modules (temporal / depformer / streaming Mimi):
  [`vllm_omni/model_executor/models/personaplex/`](../../vllm_omni/model_executor/models/personaplex/)
- Online example (server + headless client):
  [`examples/online_serving/personaplex/`](../../examples/online_serving/personaplex/)
- Offline example:
  [`examples/offline_inference/personaplex/personaplex_offline.py`](../../examples/offline_inference/personaplex/personaplex_offline.py)
- Staged offline pipeline (talker -> Mimi code2wav, async-chunk streaming):
  [`vllm_omni/deploy/personaplex.yaml`](../../vllm_omni/deploy/personaplex.yaml)
- Integration PR: [vllm-project/vllm-omni#4771](https://github.com/vllm-project/vllm-omni/pull/4771)
- Framework context: [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745)
  (duplex adapter patterns), [#1335](https://github.com/vllm-project/vllm-omni/issues/1335)
  (full-duplex target), [PR #3907](https://github.com/vllm-project/vllm-omni/pull/3907)
  (fullduplex core contracts this adapter conforms to)
- Upstream model card:
  [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)

## Why it is different from MiniCPM-o / JoyVL duplex

| | MiniCPM-o 4.5 (#3907) | JoyVL | **PersonaPlex (this)** |
|---|---|---|---|
| Cadence | 1 s chunk groups | ~1 fps frames | **80 ms lockstep (12.5 Hz)** |
| Turn control | learned `⟨listen⟩`/`⟨speak⟩` | `</silence>`/`</response>` | **none, pure lockstep** |
| Per step | variable-length token group | text decision | **1 user frame in -> 1 agent frame + 1 text token** |
| Barge-in | at chunk boundary | n/a | **native (model always hears the user)** |
| Session state | chunk-group KV | per-tick HTTP | **persistent ring KV + streaming Mimi state** |

This is the lockstep ("parallel-frame joint") shape of the duplex adapter
patterns: the adapter declares `DuplexCapability.continuous = True`, and
`core.DuplexRuntime` runs ONE eternal response that consumes input frames as they
arrive and drains on close. The flag is a small, default-off lifecycle mode, so
turn-based adapters (JoyVL, MiniCPM-o) are unaffected.

## Architecture (Moshi RQ-Transformer, all native)

- **Mimi codec** 24 kHz, 12.5 Hz, 1920 samples/frame, 8 active codebooks
  (card 2048). Streaming encode + decode via `transformers.MimiModel` with the
  PersonaPlex reference checkpoint weights.
- **Temporal transformer** (Helium backbone) 4096-d / 32 layers / 32 heads,
  sliding window 3000 frames (a 4-minute rolling context) held in a ring KV cache
  with per-slot recycle.
- **Depformer** 1024-d / 6 layers / 16 heads, autoregressive over 16 codebooks
  per frame (8 vocoded), reset each frame.
- Per-frame token column `[B, 17, 1]`: row 0 = inner-monologue text,
  rows 1-8 = agent audio, rows 9-16 = user audio.
- Persona + voice are injected once at session open through the same lockstep
  step: voice clone forces the agent stream from reference-audio Mimi codes;
  persona forces the text stream from `<system> ... <system>` tokens.

## Hardware Support

Verified on GPU (Hopper-class). The serving path is plain PyTorch eager plus the
native modules, so other CUDA GPUs with enough memory are expected to work.

## GPU

### 1x 141 GB Hopper-class GPU (H20-class, verified)

#### Environment

- OS: Linux
- Python: 3.10+ (CI covers 3.11 / 3.12)
- vLLM-Omni: PR #4771 branch or current `main` once merged
- Extra dependency for the live server: `pip install sphn` (Opus framing;
  `aiohttp` already ships with vLLM)
- `export HF_TOKEN=...` with access to the gated
  `nvidia/personaplex-7b-v1` repo (accept the license on the model page)

First run auto-downloads from the model repo: `model.safetensors`,
`tokenizer_spm_32k_3.model`, `voices.tgz`, and the Mimi reference checkpoint;
`kyutai/mimi` is fetched via `transformers` for the codec module graph.

#### Command

Live duplex server (official browser client served at `/`):

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 \
python -m vllm_omni.experimental.fullduplex.personaplex.serving.server \
    --port 8124 --voice NATF2.pt
```

Multi-session on the same GPU (elastic batching, callers join/leave any tick):

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 \
python -m vllm_omni.experimental.fullduplex.personaplex.serving.server \
    --port 8124 --batch-size 4
```

Then open `http://localhost:8124/` and allow the microphone (use headphones so
the agent does not hear itself). Headless alternative:

```bash
python examples/online_serving/personaplex/duplex_client.py \
    --url ws://localhost:8124/api/chat --input user.wav --out reply.wav
```

Offline WAV-in / WAV-out (no server):

```bash
python examples/offline_inference/personaplex/personaplex_offline.py \
    --input-wav question.wav --output-wav reply.wav --output-text reply.json \
    --voice-prompt NATF2.pt --persona "You are a wise and friendly teacher."
```

#### Verification

```bash
# GPU-free contract tests (stubbed FrameStepper)
pytest tests/e2e/features/fullduplex/test_personaplex_adapter.py \
       tests/e2e/features/fullduplex/test_personaplex_session.py \
       tests/e2e/features/fullduplex/test_personaplex_batched.py -q

# GPU e2e: batched isolation + slot recycle (coherence check needs a real
# question WAV via PPLEX_QUESTION_WAV)
pytest tests/e2e/features/fullduplex/test_personaplex_elastic.py -q
```

A green offline run producing a coherent spoken reply (and inner-monologue text
in `reply.json`) is the quickest end-to-end check.

#### Notes

- Realtime budget is 80 ms/frame. Verified eager per-tick latency at
  `--batch-size 4` is ~70-74 ms on this hardware class, i.e. all four
  conversations stay realtime; single-session has comfortable headroom.
- Decoding is greedy only (temperature/top-k knobs are intentionally not
  exposed): greedy is what the parity gates pin against the reference
  implementation, and sampling amplifies sub-bit numeric drift into divergence.
- With `--batch-size N`, connections beyond N are rejected until a slot frees;
  a recycled slot is bit-exact with a fresh engine for the same inputs.
- Voice prompts: bundled `voices.tgz` names (`NATF*`/`NATM*` natural,
  `VARF*`/`VARM*` varied) or a path to your own `.pt`/`.wav`.
- Run the browser client near the server. The 80 ms cadence is sensitive to
  network jitter; over a high-latency link playback can stutter regardless of
  engine speed. On localhost it is smooth.
- Sliding window is 3000 frames (~4 min). The ring KV recycles beyond that;
  very long sessions keep running with a rolling context.
