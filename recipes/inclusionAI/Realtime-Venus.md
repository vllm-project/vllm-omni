# Realtime-Venus Omni

[Realtime-Venus](https://huggingface.co/inclusionAI/Realtime-Venus) adapts
MiniCPM-o 4.5 with a Qwen3-8B language backbone, SigLIP vision encoder,
Whisper audio encoder and S3 speech generation. vLLM-Omni reuses the existing
MiniCPM-o Thinker → Talker → Code2Wav implementation.

## Prepare the checkpoint

The repository contains two checkpoints. Use the **Omni subdirectory**, not
the repository root, as the model path. Its small `config.json` refers to a
separately pinned root YAML; download both together:

```bash
hf download inclusionAI/Realtime-Venus \
  --include 'Realtime-Venus-Omni/*' --include config.yaml \
  --local-dir ./Realtime-Venus
```

For reproducibility, add `--revision` with the model revision you selected.
Keep `config.yaml` beside `Realtime-Venus-Omni/`, including when moving to an
offline host. The native config loader verifies its SHA-256 and model identity
before constructing the nested configs. A missing or mismatched YAML fails
explicitly; the adapter does not download a different configuration at startup.
The checkpoint's tokenizer and processor still require `--trust-remote-code`.

## Serve (full duplex by default)

```bash
vllm serve ./Realtime-Venus/Realtime-Venus-Omni \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/realtime_venus_omni.yaml \
  --host 127.0.0.1 --port 8091
```

The default profile serves continuous audio/video sessions over WebSocket.
Omitting `--deploy-config` selects this same full-duplex profile automatically.
`realtime_venus_omni_duplex.yaml` remains an equivalent explicit alias.

The profile retains the checkpoint's 10 flow-matching steps and FP32 Flow
weights. It does not inherit the MiniCPM Seed-TTS recipe's 5-step/FP16 Flow
changes. CUDA graph execution and the existing Talker multi-frame decode are
reused without duplicating model kernels.

Connect to `ws://127.0.0.1:8091/v1/realtime?duplex=1` using the existing
[DuplexClient](../../docs/serving/realtime_duplex_api.md) or
[MiniCPM-o duplex client](../../examples/online_serving/minicpmo/realtime_duplex_demo.py)
with the Venus model path. Use mono PCM16 at 16 kHz for input and 24 kHz for
output, `overlap_policy="listen_only"`, and
`playback_commit_policy="ack_only"`. Report audio actually played with
`playback.ack`; attach camera frames to their corresponding audio units.
Incoming audio continues while the model speaks. The model owns listen/speak
decisions; client-forced destructive cancellation remains subject to the
session's advertised capabilities.

Delegated text remains available in the text stream but is excluded from
speech. A fully delegated unit drains preceding speech once, without closing
the dialogue turn. Further delegated units stay silent; resumed speech starts
with fresh synthesis state, and a terminator-only unit cannot replay the tail.
The duplex profile preserves the same 10-step FP32 Flow settings as turn mode.

## Turn-based HTTP serving

Select the turn profile explicitly for ordinary `/v1/chat/completions` requests:

```bash
vllm serve ./Realtime-Venus/Realtime-Venus-Omni \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/realtime_venus_omni_turn.yaml \
  --host 127.0.0.1 --port 8091
```

Text, image, audio and video inputs use the shared multimodal processor.
Request `modalities: ["text", "audio"]` for speech as well as text. Use the
existing [chat-completions clients](../../examples/online_serving/) and
[MiniCPM-o recipe](../OpenBMB/MiniCPM-o-4_5.md) for request formats. Engine mode
is selected at startup; WebSocket query parameters do not switch engines.

## Tool execution

The released model can emit `<delegate>...</delegate>` requests. The external
[Realtime-Venus-Harness](https://github.com/inclusionAI/Realtime-Venus/tree/main/harness)
executes these tasks asynchronously and returns `<backend>...</backend>` feedback
to the originating conversation. This adaptation preserves delegate text and
excludes it from speech, but does not integrate the Harness or map delegate spans
to OpenAI `tools` / `tool_calls`. Tool execution and asynchronous result injection
are therefore not available in this serving profile.

## Scope

- Both full-duplex serving (the default) and the explicit turn-based profile
  use the shared Thinker, Talker and Code2Wav stages.
- The external Realtime-Venus delegation harness, asynchronous tool-result
  orchestration and long-video memory adapter are not integrated. Backbone
  compatibility does not establish those external features.
- The Audio-only checkpoint is a separate configuration and is not covered by
  this recipe. Other devices and distributed layouts need separate validation.
- Validation coverage and measured performance are reported in the linked PR
  evidence; basic generation smoke is not a comprehensive quality evaluation.
