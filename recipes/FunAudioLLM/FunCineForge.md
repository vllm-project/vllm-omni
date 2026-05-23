# FunCineForge for zero-shot movie dubbing on 1x A100 80GB

## Summary

- Vendor: FunAudioLLM (Tongyi Lab)
- Model: `FunAudioLLM/Fun-CineForge`
- Task: Zero-shot movie dubbing and TTS with face/speaker conditioning
- Mode: Online serving with the OpenAI-compatible speech API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve FunCineForge for zero-shot dubbing or
text-to-speech generation. FunCineForge extends CosyVoice3 with face embedding
conditioning, structured dialogue metadata (time/speaker/gender/age tags), and
natural language clue descriptions for emotion and style control. It supports
monologue, narration, dialogue, and multi-speaker cinematic scenes.

## References

- Upstream or canonical docs:
  [FunCineForge GitHub](https://github.com/FunAudioLLM/FunCineForge)
- Related example under `examples/`:
  [`examples/offline_inference/text_to_speech/funcineforge/`](../../examples/offline_inference/text_to_speech/funcineforge/)
- Related issue or discussion:
  [arXiv 2601.14777](https://arxiv.org/abs/2601.14777)

## Hardware Support

This recipe documents one tested reference configuration for CUDA GPU serving.
Both pipeline stages (talker + code2wav) fit on a single GPU.

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the server from the repository root:

```bash
vllm serve FunAudioLLM/Fun-CineForge --omni --port 8091
```

To use a custom deploy config:

```bash
vllm serve FunAudioLLM/Fun-CineForge \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/deploy/funcineforge.yaml
```

#### Verification

Run the offline inference example to validate the model loads correctly and
optionally verify the generated speech with ASR:

```bash
python examples/offline_inference/text_to_speech/funcineforge/end2end.py \
  --model FunAudioLLM/Fun-CineForge \
  --ref-audio /path/to/reference.wav \
  --ref-text "Reference speaker text." \
  --output outputs/funcineforge_offline_sync.wav \
  --verify-asr
```

Use `--async-chunk` for offline streaming-mode orchestration:

```bash
python examples/offline_inference/text_to_speech/funcineforge/end2end.py \
  --mode offline \
  --async-chunk \
  --model FunAudioLLM/Fun-CineForge \
  --ref-audio /path/to/reference.wav \
  --ref-text "Reference speaker text." \
  --output outputs/funcineforge_offline_async.wav \
  --verify-asr
```

For a quick API smoke test via the speech endpoint:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CineForge",
    "input": "Hello, this is a dubbing test.",
    "voice": "default",
    "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/FunCineForge/main/exps/data/ref.wav",
    "ref_text": "Reference speaker text."
  }' \
  --output test_output.wav
```

The same verifier can exercise the online endpoint. Start the server with
`--no-async-chunk` for sync mode, or omit that flag for async-chunk serving,
then run:

```bash
python examples/offline_inference/text_to_speech/funcineforge/end2end.py \
  --mode online \
  --model FunAudioLLM/Fun-CineForge \
  --api-base http://127.0.0.1:8091 \
  --ref-audio /path/to/reference.wav \
  --ref-text "Reference speaker text." \
  --output outputs/funcineforge_online.wav \
  --verify-asr
```

#### Notes

- Memory usage: Both stages fit on a single GPU with ~60% memory utilization total (talker 0.4, code2wav 0.2).
- Key flags: `--omni` is required. `dtype` defaults to `float32` per the deploy config.
- Known limitations: Face embedding conditioning requires pre-extracted face features in npz format. The model checkpoint uses DeepSpeed format.
