# Duo 1.0 on vLLM-Omni

[Duo](https://github.com/inclusionAI/Duo) is a *reciprocal world model*: it
perceives audio and video continuously and answers in the same medium, instead of
finishing one turn before it starts listening again. This page covers the
`inclusionAI/Duo-1.0-causal-base` checkpoint.

## What this integration is

Duo is not a new diffusion backbone. Its **Realization** module is the
MiniMax-H3 DiT (`MiniMaxH3DiTModel`, 50 layers, hidden size 5376), so the
Realization and Codec stages run on the existing H3 blocks, attention
backends, VAE fast paths and LoRA plumbing in this repository. What Duo adds on
top is:

| Piece | Role |
| --- | --- |
| **Intent** (`DuoIntentForConditionalGeneration`) | Qwen3-VL 8B prompted once per chunk; emits `<speak>` + text or `<wait>`, and its selected hidden states condition Realization. |
| **Realization** (`DuoPipeline`) | H3 DiT denoising the assistant media latents, conditioned on Intent states and the committed (clean) history. |
| **Codec** (`DuoCodecPipeline`) | Streaming video VAE + audio VAE turning latents into publishable 24 fps / 32 kHz chunks. |
| **ASR** (`DuoASRForConditionalGeneration`) | Optional Qwen3-ASR front stage for the `duo_asr` topology. |

The three stages are separate pipelines over one checkpoint so each can be
placed, tensor-parallelized and budgeted independently.

## Chunk timeline

An interaction is cut into fixed chunks: **17 video frames at 24 fps**
(≈708 ms), mapping to **5 video latents** and **29 stereo audio latents**.
At chunk `k`, Intent only sees evidence available before the chunk's causal
boundary. See the upstream [architecture note](https://github.com/inclusionAI/Duo/blob/main/docs/architecture.md)
for the visibility contract.

## Execution and configuration

The `duo` topology runs Intent, Realization and Codec in that order; the `codec`
stage is the only `final_output` because a chunk publishes an MP4 carrying both
tracks. Online serving is opt-in through a deploy config that selects
`session_mode: duplex` — `vllm_omni/deploy/duo_dmd.yaml` (4 DMD evaluations,
flow shift 5) and `vllm_omni/deploy/duo_tp8_shared.yaml` (ASR + Intent TP4 +
Realization TP8 + four codec tile lanes). Check the device assignments and
memory budgets against your own system before using them.

The default offline sampler for the causal-base checkpoint is Euler, 50 steps,
flow shift 1.0 — that is what `realization/complete.json` records.

## Required artifacts

The checkpoint is a *bundle*, not a single model directory:

```text
Duo-1.0-causal-base/
├── realization/            # H3-format backbone shards + DCP
│   ├── backbone/           # config.json + model-0000*-of-00013.safetensors
│   ├── dcp/                # distributed checkpoint
│   └── complete.json       # bundle identity: codec/intent/realization hashes
├── intent/                 # Qwen3-VL weights, tokenizer, preprocessors
├── intent_adapter/         # LoRA adapter + protocol manifest
├── codec/                  # video VAE + audio VAE
└── speech/                 # Qwen3-ASR recognizer + forced aligner
```

`realization/complete.json` is the source of truth for the packaged model family
and weight format; read it before assuming a layout. The Realization weights are
H3-initialized and carry their own license — check the model provider's terms
separately from this integration's Apache-2.0 code.

## Client protocol

Duplex clients use `WS /v1/realtime?duplex=1` (or `WS /v1/duplex`) with the
`vllm_omni.clients.duo` preset. Input is 16 kHz float32 PCM plus `video_frames`;
output is 32 kHz PCM16 stereo with a duplex `video` extension carrying JPEG
frames. The payload is not a standard Realtime video-output event; audio and
video share one playback acknowledgement.

## Validation and limitations

- The upstream `inference/vllm-omni-duo` branch ships additional scheduler and
  output-processor hooks (bounded Realization window trimming, H50 scheduled-row
  handoff) that are **not** part of this integration yet. Long sessions depend on
  those windows; treat this as a preview port of the model, not of the full
  streaming scheduler.
- Online duplex serving requires the duplex plugin and a deploy config; the
  plain generation APIs do not expose reply-boundary decisions.
- Direct user-audiovisual conditioning of Realization is not enabled by the
  Duo 1.0 offline checkpoint; user input reaches Realization through Intent.
