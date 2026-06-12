# IndexTTS2

## Summary

- Vendor: IndexTeam
- Model: `IndexTeam/IndexTTS-2`
- Task: Text-to-Speech with voice cloning and emotion control
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe for running IndexTTS2, a two-stage TTS model that produces
high-quality 22050 Hz speech with voice cloning from a reference audio clip.
Supports Chinese, English, Japanese and mixed-language text with optional
emotion conditioning.

## References

- Offline example:
  [`examples/offline_inference/text_to_speech/indextts2/end2end.py`](../../examples/offline_inference/text_to_speech/indextts2/end2end.py)
- Online client:
  [`examples/online_serving/text_to_speech/indextts2/speech_client.py`](../../examples/online_serving/text_to_speech/indextts2/speech_client.py)

## Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: use versions from your current checkout
- GPU: 1x L40/A100/H100 (model fits in ~16 GB; Stage 0 and Stage 1 share one GPU)

## Start server (single command)

From repository root:

```bash
vllm serve IndexTeam/IndexTTS-2 --omni --port 8092
```

Notes:

- `--omni` is required.
- The default deploy config `vllm_omni/deploy/indextts2.yaml` is loaded
  automatically by model registry.
- `async_chunk` is disabled — S2Mel flow matching requires the full mel code
  sequence from Stage 0 before Stage 1 can run.

#### Runtime tuning

```bash
# Increase Stage 0 concurrency
vllm serve IndexTeam/IndexTTS-2 --omni --port 8092 \
  --stage-overrides '{"0": {"max_num_seqs": 8}}'

# Adjust GPU memory allocation
vllm serve IndexTeam/IndexTTS-2 --omni --port 8092 \
  --stage-overrides '{
    "0": {"gpu_memory_utilization": 0.5},
    "1": {"gpu_memory_utilization": 0.3}
  }'
```

## Verification

### Client test

```bash
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
  --text "你好，这是IndexTTS2语音合成测试。" \
  --ref-audio /path/to/reference.wav \
  --api-base http://localhost:8092
```

### curl smoke test

```bash
# Basic TTS (requires ref_audio for voice cloning)
curl -s http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IndexTeam/IndexTTS-2",
    "input": "你好，世界！",
    "voice": "default",
    "response_format": "wav",
    "ref_audio": "data:audio/wav;base64,<BASE64_ENCODED_AUDIO>"
  }' \
  -o output.wav
```

## Architecture

```
Stage 0 (GPT-2 AR, PagedAttention)     Stage 1 (S2Mel + BigVGAN)
┌──────────────────────────────┐       ┌──────────────────────────┐
│ preprocess():                │       │ forward():               │
│   Wav2Vec2-BERT → Conformer  │       │   gpt_layer(latent)      │
│   → Perceiver → 32 latents   │  →→→  │   + vq2emb(mel_codes)   │
│   + CAMPPlus style + emotion │       │   → length_regulator     │
│   + BPE text + mel position  │       │   → CFM (25 Euler steps) │
│                              │       │   → BigVGAN → 22kHz wav  │
│ forward(): GPT-2 (24L/20H)  │       └──────────────────────────┘
│ compute_logits(): mel_head   │
└──────────────────────────────┘
```

## Features

- **Voice cloning**: Pass `ref_audio` (base64 data URL, file path, or HTTP URL)
- **Emotion control**: Pass `emo_audio` (emotion reference audio) or `emo_vector`
  (8-dim emotion distribution) via `extra_params`. `emo_vector` order is
  `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
- **Emotion text**: Pass both `use_emo_text: true` and `emo_text` via
  `extra_params`. Matching upstream IndexTTS2, `emo_text` is only used to
  predict an emotion vector; it does not replace the request `input` text.
  `emo_audio`, `emo_vector`, and `use_emo_text` are mutually exclusive request
  modes.
- **Multi-language**: Chinese, English, Japanese, and mixed-language text
- **Non-streaming**: S2Mel flow matching produces the full spectrogram at once
