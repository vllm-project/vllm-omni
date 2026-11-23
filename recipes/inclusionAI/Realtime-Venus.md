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

## Serve

```bash
vllm serve ./Realtime-Venus/Realtime-Venus-Omni \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/realtime_venus_omni.yaml \
  --host 127.0.0.1 --port 8091
```

The profile serves ordinary `/v1/chat/completions` requests, with text, image,
audio and video inputs routed through the shared multimodal processor. Request
`modalities: ["text", "audio"]` for speech as well as text. Use the existing
[chat-completions clients](../../examples/online_serving/) and
[MiniCPM-o recipe](../OpenBMB/MiniCPM-o-4_5.md) for request formats.

The profile retains the checkpoint's 10 flow-matching steps and FP32 Flow
weights. It does not inherit the MiniCPM Seed-TTS recipe's 5-step/FP16 Flow
changes. CUDA graph execution and the existing Talker multi-frame decode are
reused without duplicating model kernels.

## Scope

- The default is turn-based serving. The shared duplex bridge additionally
  filters `<delegate>…</delegate>` token/hidden rows from speech conditioning,
  carries the mask across units, and clears it at the turn boundary.
- The external Realtime-Venus delegation harness, asynchronous tool-result
  orchestration and long-video memory adapter are not integrated. Backbone
  compatibility does not establish those features or full duplex parity.
- The Audio-only checkpoint is a separate configuration and is not covered by
  this recipe. Other devices and distributed layouts need separate validation.
- Validation coverage and measured performance are reported in the linked PR
  evidence; basic generation smoke is not a comprehensive quality evaluation.
